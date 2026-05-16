from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads import (
    CategoricalHead,
    ClassificationHead,
    EventTypeHead,
    ExistenceHead,
    FutureAmountStatsHead,
    FutureCategoricalProfileHead,
    FutureCountBucketHead,
    FutureEventTypeProfileHead,
    FutureGapBucketHead,
    NumericalHead,
    TimeDeltaHead,
)
from src.models.pooling import CLSPooling, MultiPoolingProjection, get_pooling
from src.models.tokenizer import MixedEventTokenizer
from src.models.transformer_encoder import TimeAwareTransformerEncoder

_VALID_MODES = {"pretrain", "encode", "finetune", "forecast"}


class DMEEncoder(nn.Module):
    def __init__(self, config: dict, vocab_info: dict) -> None:
        super().__init__()

        m = config["model"]
        hidden_dim: int = m["hidden_dim"]

        event_type_vocab_size: int = vocab_info["event_type_vocab_size"]
        cat_vocab_sizes: list[int] = vocab_info.get("cat_vocab_sizes", [])
        num_num_features: int = vocab_info.get("num_num_features", 0)
        num_classes: int | None = vocab_info.get("num_classes")

        self.tokenizer = MixedEventTokenizer(
            event_type_vocab_size=event_type_vocab_size,
            event_type_emb_dim=m["event_type_emb_dim"],
            cat_vocab_sizes=cat_vocab_sizes,
            cat_emb_dim=m["cat_emb_dim"],
            num_num_features=num_num_features,
            num_projection_dim=m["num_projection_dim"],
            time_projection_dim=m["time_projection_dim"],
            hidden_dim=hidden_dim,
            max_seq_len=m["max_seq_len"],
            dropout=m["dropout"],
            use_profile_token=m.get("use_profile_token", False),
        )
        self.use_profile_token = bool(m.get("use_profile_token", False))
        self.diffusion_enabled = (
            config.get("pretraining", {}).get("objective", "denoising") == "diffusion"
        )
        d3pm_cfg = config.get("d3pm", {})
        self.d3pm_event_type_enabled = (
            bool(d3pm_cfg.get("enabled", False))
            and "event_type" in set(d3pm_cfg.get("apply_to", ["event_type"]))
        )
        diffusion_steps = int(config.get("diffusion", {}).get("num_steps", 100))
        self.diffusion_timestep_embedding = (
            nn.Embedding(diffusion_steps + 1, hidden_dim) if self.diffusion_enabled else None
        )

        self.encoder = TimeAwareTransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=m["num_layers"],
            num_heads=m["num_heads"],
            dim_feedforward=m["dim_feedforward"],
            dropout=m["dropout"],
            activation=m["activation"],
        )

        pooling_cfg = config.get("pooling", {})
        pooling_type = pooling_cfg.get("type", "mean")
        client_embedding_dim = int(m.get("client_embedding_dim", hidden_dim))
        self.pooling = get_pooling(
            pooling_type,
            hidden_dim,
            client_embedding_dim=client_embedding_dim,
            components=pooling_cfg.get("components"),
            dropout=m["dropout"],
            has_profile_token=self.use_profile_token,
        )
        self.representation_dim = (
            client_embedding_dim if isinstance(self.pooling, MultiPoolingProjection) else hidden_dim
        )

        self.event_type_head = EventTypeHead(hidden_dim, event_type_vocab_size)
        self.event_type_prev_head: EventTypeHead | None = (
            EventTypeHead(hidden_dim, event_type_vocab_size)
            if self.diffusion_enabled and self.d3pm_event_type_enabled
            else None
        )
        self.time_delta_head = TimeDeltaHead(hidden_dim)
        self.existence_head = ExistenceHead(hidden_dim)
        self.time_delta_eps_head: TimeDeltaHead | None = (
            TimeDeltaHead(hidden_dim) if self.diffusion_enabled else None
        )

        self.numerical_head: NumericalHead | None = (
            NumericalHead(hidden_dim, num_num_features) if num_num_features > 0 else None
        )
        self.numerical_eps_head: NumericalHead | None = (
            NumericalHead(hidden_dim, num_num_features)
            if self.diffusion_enabled and num_num_features > 0
            else None
        )
        self.cat_head: CategoricalHead | None = (
            CategoricalHead(hidden_dim, cat_vocab_sizes) if cat_vocab_sizes else None
        )

        self.classifier: ClassificationHead | None = (
            ClassificationHead(
                hidden_dim=self.representation_dim,
                num_classes=num_classes,
                dropout=m["dropout"],
            )
            if num_classes is not None
            else None
        )

        f_cfg = config.get("forecasting", {})
        self.forecasting_enabled = bool(
            f_cfg.get("enabled", False)
            or f_cfg.get("pretrain_aux_enabled", False)
            or f_cfg.get("finetune_aux_enabled", False)
        )
        if self.forecasting_enabled:
            count_buckets = int(f_cfg.get("count_num_buckets", 6))
            gap_buckets = int(f_cfg.get("gap_num_buckets", 6))
            self.future_event_type_profile_head = FutureEventTypeProfileHead(
                self.representation_dim, event_type_vocab_size
            )
            self.future_count_bucket_head = FutureCountBucketHead(
                self.representation_dim, count_buckets
            )
            self.future_amount_stats_head = FutureAmountStatsHead(self.representation_dim)
            self.future_gap_bucket_head = FutureGapBucketHead(self.representation_dim, gap_buckets)
            self.future_cat_profile_head = FutureCategoricalProfileHead(
                self.representation_dim, cat_vocab_sizes
            )
        else:
            self.future_event_type_profile_head = None
            self.future_count_bucket_head = None
            self.future_amount_stats_head = None
            self.future_gap_bucket_head = None
            self.future_cat_profile_head = None

    # ── forward ───────────────────────────────────────────────────────────────

    def _prepend_profile_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if not self.use_profile_token:
            return attention_mask
        B = attention_mask.shape[0]
        profile_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=attention_mask.device)
        return torch.cat([profile_mask, attention_mask], dim=1)

    def _pool(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if isinstance(self.pooling, MultiPoolingProjection):
            return self.pooling(h, attention_mask)
        if self.use_profile_token:
            if isinstance(self.pooling, CLSPooling):
                return self.pooling(h, attention_mask)
            return self.pooling(h[:, 1:, :], attention_mask[:, 1:])
        return self.pooling(h, attention_mask)

    def forward(self, batch: dict, mode: str = "pretrain") -> dict:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of: {_VALID_MODES}")

        event_attention_mask: torch.Tensor = batch["attention_mask"]

        x = self.tokenizer(
            event_type=batch["event_type"],
            time_delta=batch["time_delta"],
            num_features=batch.get("num_features"),
            cat_features=batch.get("cat_features"),
            attention_mask=event_attention_mask,
        )
        attention_mask = self._prepend_profile_mask(event_attention_mask)
        if self.diffusion_timestep_embedding is not None and "diffusion_t" in batch:
            t_emb = self.diffusion_timestep_embedding(batch["diffusion_t"].long())
            x = x + t_emb.unsqueeze(1)
            x = x * attention_mask.unsqueeze(-1).float()
        h = self.encoder(x, attention_mask)  # [B, L(+1), H]
        event_h = h[:, 1:, :] if self.use_profile_token else h

        if mode == "pretrain":
            out: dict = {
                "event_type_logits": self.event_type_head(event_h),  # [B, L, V]
                "time_delta_pred": self.time_delta_head(event_h),     # [B, L, 1]
                "existence_logits": self.existence_head(event_h),     # [B, L, 1]
                "hidden_states": event_h,                             # [B, L, H]
            }
            if self.numerical_head is not None:
                out["num_pred"] = self.numerical_head(event_h)  # [B, L, N]
            if self.cat_head is not None:
                out["cat_logits"] = self.cat_head(event_h)      # list[[B, L, V_j]]
            if self.time_delta_eps_head is not None:
                out["time_delta_eps_pred"] = self.time_delta_eps_head(event_h)
            if self.numerical_eps_head is not None:
                out["num_eps_pred"] = self.numerical_eps_head(event_h)
            if self.event_type_prev_head is not None:
                out["event_type_prev_logits"] = self.event_type_prev_head(event_h)
            return out

        pooled = self._pool(h, attention_mask)  # [B, H or client_embedding_dim]

        if mode == "encode":
            return {"representation": pooled}

        if mode == "forecast":
            if not self.forecasting_enabled:
                raise ValueError("DMEEncoder forecasting heads are disabled in config")
            return {
                "future_event_type_profile": self.future_event_type_profile_head(pooled),
                "future_count_bucket_logits": self.future_count_bucket_head(pooled),
                "future_amount_stats": self.future_amount_stats_head(pooled),
                "future_gap_bucket_logits": self.future_gap_bucket_head(pooled),
                "future_cat_profiles": self.future_cat_profile_head(pooled),
                "representation": pooled,
            }

        # mode == "finetune"
        if self.classifier is None:
            raise ValueError(
                "DMEEncoder was initialized without 'num_classes' in vocab_info; "
                "cannot use mode='finetune'."
            )
        return {
            "logits": self.classifier(pooled),  # [B, num_classes]
            "representation": pooled,           # [B, representation_dim]
        }

    # ── parameter helpers ─────────────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        def _n(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        result: dict[str, int] = {
            "tokenizer": _n(self.tokenizer),
            "encoder": _n(self.encoder),
            "pooling": _n(self.pooling),
            "event_type_head": _n(self.event_type_head),
            "time_delta_head": _n(self.time_delta_head),
            "existence_head": _n(self.existence_head),
        }
        if self.classifier is not None:
            result["classifier"] = _n(self.classifier)
        if self.numerical_head is not None:
            result["numerical_head"] = _n(self.numerical_head)
        if self.time_delta_eps_head is not None:
            result["time_delta_eps_head"] = _n(self.time_delta_eps_head)
        if self.numerical_eps_head is not None:
            result["numerical_eps_head"] = _n(self.numerical_eps_head)
        if self.diffusion_timestep_embedding is not None:
            result["diffusion_timestep_embedding"] = _n(self.diffusion_timestep_embedding)
        if self.event_type_prev_head is not None:
            result["event_type_prev_head"] = _n(self.event_type_prev_head)
        if self.cat_head is not None:
            result["cat_head"] = _n(self.cat_head)
        if self.forecasting_enabled:
            result["forecast_heads"] = (
                _n(self.future_event_type_profile_head)
                + _n(self.future_count_bucket_head)
                + _n(self.future_amount_stats_head)
                + _n(self.future_gap_bucket_head)
                + _n(self.future_cat_profile_head)
            )
        result["total"] = _n(self)
        return result

    def get_encoder_params(self) -> list[nn.Parameter]:
        """Параметры tokenizer + encoder + pooling для оптимизатора с малым lr."""
        return (
            list(self.tokenizer.parameters())
            + list(self.encoder.parameters())
            + list(self.pooling.parameters())
        )

    def get_head_params(self) -> list[nn.Parameter]:
        """Параметры classifier для основного lr при fine-tuning."""
        if self.classifier is None:
            return []
        return list(self.classifier.parameters())
