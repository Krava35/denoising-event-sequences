from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads import (
    CategoricalHead,
    ClassificationHead,
    EventTypeHead,
    ExistenceHead,
    NumericalHead,
    TimeDeltaHead,
)
from src.models.pooling import get_pooling
from src.models.tokenizer import MixedEventTokenizer
from src.models.transformer_encoder import TimeAwareTransformerEncoder

_VALID_MODES = {"pretrain", "encode", "finetune"}


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
        )

        self.encoder = TimeAwareTransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=m["num_layers"],
            num_heads=m["num_heads"],
            dim_feedforward=m["dim_feedforward"],
            dropout=m["dropout"],
            activation=m["activation"],
        )

        self.pooling = get_pooling(config["pooling"]["type"], hidden_dim)

        self.event_type_head = EventTypeHead(hidden_dim, event_type_vocab_size)
        self.time_delta_head = TimeDeltaHead(hidden_dim)
        self.existence_head = ExistenceHead(hidden_dim)

        self.numerical_head: NumericalHead | None = (
            NumericalHead(hidden_dim, num_num_features) if num_num_features > 0 else None
        )
        self.cat_head: CategoricalHead | None = (
            CategoricalHead(hidden_dim, cat_vocab_sizes) if cat_vocab_sizes else None
        )

        self.classifier: ClassificationHead | None = (
            ClassificationHead(hidden_dim=hidden_dim, num_classes=num_classes, dropout=m["dropout"])
            if num_classes is not None
            else None
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: dict, mode: str = "pretrain") -> dict:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of: {_VALID_MODES}")

        attention_mask: torch.Tensor = batch["attention_mask"]

        x = self.tokenizer(
            event_type=batch["event_type"],
            time_delta=batch["time_delta"],
            num_features=batch.get("num_features"),
            cat_features=batch.get("cat_features"),
            attention_mask=attention_mask,
        )
        h = self.encoder(x, attention_mask)  # [B, L, H]

        if mode == "pretrain":
            out: dict = {
                "event_type_logits": self.event_type_head(h),  # [B, L, V]
                "time_delta_pred": self.time_delta_head(h),     # [B, L, 1]
                "existence_logits": self.existence_head(h),     # [B, L, 1]
                "hidden_states": h,                             # [B, L, H]
            }
            if self.numerical_head is not None:
                out["num_pred"] = self.numerical_head(h)        # [B, L, N]
            if self.cat_head is not None:
                out["cat_logits"] = self.cat_head(h)            # list[[B, L, V_j]]
            return out

        pooled = self.pooling(h, attention_mask)  # [B, H]

        if mode == "encode":
            return {"representation": pooled}

        # mode == "finetune"
        if self.classifier is None:
            raise ValueError(
                "DMEEncoder was initialized without 'num_classes' in vocab_info; "
                "cannot use mode='finetune'."
            )
        return {
            "logits": self.classifier(pooled),  # [B, num_classes]
            "representation": pooled,           # [B, H]
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
        if self.cat_head is not None:
            result["cat_head"] = _n(self.cat_head)
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
