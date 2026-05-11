from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from src.corruption.categorical import _NUM_SPECIAL
from src.corruption.diffusion import DiffusionSchedule
from src.data.forecasting import get_derived_time_feature_names

if TYPE_CHECKING:
    from src.data.preprocessing import EventPreprocessor
    from src.models.dme_encoder import DMEEncoder


def _generation_cfg(config: dict) -> dict:
    return config.get("generation", {})


def _midpoint_prefix_lengths(attention_mask: torch.Tensor, config: dict, suffix_len: int) -> torch.Tensor:
    g_cfg = _generation_cfg(config)
    min_ratio = float(g_cfg.get("prefix_min_ratio", 0.50))
    max_ratio = float(g_cfg.get("prefix_max_ratio", 0.80))
    ratio = 0.5 * (min_ratio + max_ratio)
    lengths = attention_mask.sum(dim=1).long()
    result = torch.zeros_like(lengths)
    for i, n_raw in enumerate(lengths.tolist()):
        n = int(n_raw)
        if n <= 1:
            result[i] = n
            continue
        cut = int(np.floor(n * ratio))
        cut = max(1, min(cut, n - 1))
        if n - cut < 1:
            cut = max(1, n - min(suffix_len, n - 1))
        result[i] = cut
    return result


def _repeat_batch(batch: dict, num_samples: int) -> tuple[dict, list[int]]:
    batch_size = int(batch["event_type"].shape[0])
    if num_samples <= 1:
        return batch, [0] * batch_size

    repeated: dict = {}
    sample_ids: list[int] = []
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            repeated[key] = value.repeat_interleave(num_samples, dim=0)
        elif key == "entity_id":
            entities: list[str] = []
            for entity_id in value:
                for sample_id in range(num_samples):
                    entities.append(entity_id)
                    sample_ids.append(sample_id)
            repeated[key] = entities
        else:
            repeated[key] = value
    if not sample_ids:
        sample_ids = [sample_id for _ in range(batch_size) for sample_id in range(num_samples)]
    return repeated, sample_ids


def build_conditional_generation_batch(
    clean_batch: dict,
    config: dict,
    *,
    suffix_len: int | None = None,
    num_samples: int = 1,
    prefix_lengths: torch.Tensor | None = None,
) -> dict:
    """Build a clean-prefix/noisy-suffix seed batch for conditional generation."""
    g_cfg = _generation_cfg(config)
    suffix_len = int(suffix_len if suffix_len is not None else g_cfg.get("suffix_len", 16))
    if suffix_len < 1:
        raise ValueError("suffix_len must be >= 1")

    clean_batch, sample_ids = _repeat_batch(clean_batch, int(num_samples))
    event_type_src = clean_batch["event_type"]
    time_src = clean_batch["time_delta"]
    num_src = clean_batch.get("num_features")
    cat_src = clean_batch.get("cat_features")
    attention_mask = clean_batch["attention_mask"]
    device = event_type_src.device
    max_seq_len = int(config.get("model", {}).get("max_seq_len", event_type_src.shape[1]))
    if suffix_len > max_seq_len:
        raise ValueError("suffix_len must be <= model.max_seq_len")

    if prefix_lengths is None:
        source_prefix_lengths = _midpoint_prefix_lengths(attention_mask, config, suffix_len)
    else:
        source_prefix_lengths = prefix_lengths.to(device=device).long()
        if num_samples > 1 and source_prefix_lengths.shape[0] != event_type_src.shape[0]:
            source_prefix_lengths = source_prefix_lengths.repeat_interleave(num_samples, dim=0)

    source_prefix_lengths = torch.minimum(source_prefix_lengths, attention_mask.sum(dim=1).long())
    max_prefix_len = max(0, max_seq_len - suffix_len)
    kept_prefix_lengths = torch.clamp(source_prefix_lengths, min=0, max=max_prefix_len)
    total_len = int(min(max_seq_len, max(kept_prefix_lengths.max().item(), 0) + suffix_len))

    B = event_type_src.shape[0]
    n_num = int(num_src.shape[-1]) if isinstance(num_src, torch.Tensor) else 0
    n_cat = int(cat_src.shape[-1]) if isinstance(cat_src, torch.Tensor) else 0

    event_type = torch.zeros(B, total_len, dtype=torch.long, device=device)
    time_delta = torch.zeros(B, total_len, dtype=time_src.dtype, device=device)
    num_features = torch.zeros(B, total_len, n_num, dtype=time_src.dtype, device=device)
    cat_features = torch.zeros(B, total_len, n_cat, dtype=torch.long, device=device)
    generated_attention = torch.zeros(B, total_len, dtype=torch.bool, device=device)
    prefix_mask = torch.zeros(B, total_len, dtype=torch.bool, device=device)
    suffix_mask = torch.zeros(B, total_len, dtype=torch.bool, device=device)

    target_event_type = torch.zeros_like(event_type)
    target_time_delta = torch.zeros_like(time_delta)
    target_num_features = torch.zeros_like(num_features)
    target_cat_features = torch.zeros_like(cat_features)
    target_mask = torch.zeros_like(suffix_mask)

    for i in range(B):
        valid_len = int(attention_mask[i].sum().item())
        source_cut = int(source_prefix_lengths[i].item())
        prefix_len = int(kept_prefix_lengths[i].item())
        prefix_start = max(0, source_cut - prefix_len)
        prefix_end = source_cut
        if prefix_len > 0:
            event_type[i, :prefix_len] = event_type_src[i, prefix_start:prefix_end]
            time_delta[i, :prefix_len] = time_src[i, prefix_start:prefix_end]
            if n_num > 0:
                num_features[i, :prefix_len] = num_src[i, prefix_start:prefix_end]
            if n_cat > 0:
                cat_features[i, :prefix_len] = cat_src[i, prefix_start:prefix_end]
            prefix_mask[i, :prefix_len] = True

        suffix_start = prefix_len
        suffix_end = min(total_len, suffix_start + suffix_len)
        generated_attention[i, :suffix_end] = True
        suffix_mask[i, suffix_start:suffix_end] = True
        event_type[i, suffix_start:suffix_end] = 2
        time_delta[i, suffix_start:suffix_end] = torch.randn(
            suffix_end - suffix_start,
            dtype=time_delta.dtype,
            device=device,
        )
        if n_num > 0:
            num_features[i, suffix_start:suffix_end] = torch.randn(
                suffix_end - suffix_start,
                n_num,
                dtype=num_features.dtype,
                device=device,
            )
        if n_cat > 0:
            cat_features[i, suffix_start:suffix_end] = 3

        true_future_len = max(0, min(suffix_end - suffix_start, valid_len - source_cut))
        if true_future_len > 0:
            src_slice = slice(source_cut, source_cut + true_future_len)
            dst_slice = slice(suffix_start, suffix_start + true_future_len)
            target_event_type[i, dst_slice] = event_type_src[i, src_slice]
            target_time_delta[i, dst_slice] = time_src[i, src_slice]
            if n_num > 0:
                target_num_features[i, dst_slice] = num_src[i, src_slice]
            if n_cat > 0:
                target_cat_features[i, dst_slice] = cat_src[i, src_slice]
            target_mask[i, dst_slice] = True

    batch = {
        "event_type": event_type,
        "time_delta": time_delta,
        "num_features": num_features,
        "cat_features": cat_features,
        "attention_mask": generated_attention,
        "entity_id": clean_batch.get("entity_id", [str(i) for i in range(B)]),
    }
    if "label" in clean_batch:
        batch["label"] = clean_batch["label"]

    targets = {
        "event_type": target_event_type,
        "time_delta": target_time_delta,
        "num_features": target_num_features,
        "cat_features": target_cat_features,
    }
    return {
        "batch": batch,
        "targets": targets,
        "prefix_mask": prefix_mask,
        "suffix_mask": suffix_mask,
        "target_mask": target_mask,
        "source_prefix_lengths": source_prefix_lengths,
        "kept_prefix_lengths": kept_prefix_lengths,
        "sample_ids": sample_ids,
    }


def _sampling_timesteps(num_steps: int, num_sampling_steps: int) -> list[int]:
    raw = torch.linspace(num_steps, 0, int(num_sampling_steps) + 1).round().long().tolist()
    steps: list[int] = []
    for step in raw:
        step = int(step)
        if not steps or step != steps[-1]:
            steps.append(step)
    if steps[-1] != 0:
        steps.append(0)
    return steps


def _sample_discrete(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    if logits.shape[-1] > _NUM_SPECIAL:
        logits = logits.clone()
        logits[..., :_NUM_SPECIAL] = -torch.inf
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        values, _ = torch.topk(logits, top_k, dim=-1)
        cutoff = values[..., -1:].expand_as(logits)
        logits = torch.where(logits >= cutoff, logits, torch.full_like(logits, -torch.inf))
    temperature = max(float(temperature), 1e-6)
    probs = torch.softmax(logits / temperature, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    denom = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(denom > 0, probs / denom.clamp_min(1e-12), torch.ones_like(probs) / probs.shape[-1])
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(logits.shape[:-1])


def _ddim_update(
    x0_pred: torch.Tensor,
    eps_pred: torch.Tensor | None,
    x_t: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_prev: torch.Tensor,
) -> torch.Tensor:
    while alpha_t.ndim < x_t.ndim:
        alpha_t = alpha_t.unsqueeze(-1)
    while alpha_prev.ndim < x_t.ndim:
        alpha_prev = alpha_prev.unsqueeze(-1)
    if eps_pred is None:
        eps_pred = (x_t - torch.sqrt(alpha_t).clamp_min(1e-12) * x0_pred) / torch.sqrt(
            (1.0 - alpha_t).clamp_min(1e-12)
        )
    return torch.sqrt(alpha_prev) * x0_pred + torch.sqrt((1.0 - alpha_prev).clamp_min(0.0)) * eps_pred


@torch.no_grad()
def generate_suffix(
    model: "DMEEncoder",
    generation_batch: dict,
    config: dict,
    vocab_info: dict,
) -> dict:
    """Run DDIM-lite conditional suffix generation."""
    g_cfg = _generation_cfg(config)
    sampler = g_cfg.get("sampler", "ddim_lite")
    if sampler != "ddim_lite":
        raise ValueError("Only generation.sampler='ddim_lite' is supported")

    batch = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in generation_batch["batch"].items()
    }
    prefix_clean = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    prefix_mask = generation_batch["prefix_mask"].to(batch["event_type"].device)
    suffix_mask = generation_batch["suffix_mask"].to(batch["event_type"].device)

    d_cfg = config.get("diffusion", {})
    schedule = DiffusionSchedule(
        num_steps=int(d_cfg.get("num_steps", 100)),
        beta_start=float(d_cfg.get("beta_start", 1e-4)),
        beta_end=float(d_cfg.get("beta_end", 2e-2)),
    )
    steps = _sampling_timesteps(
        schedule.num_steps,
        int(g_cfg.get("num_sampling_steps", min(50, schedule.num_steps))),
    )

    temperature_event = float(g_cfg.get("temperature_event_type", 1.0))
    temperature_cat = float(g_cfg.get("temperature_cat", 1.0))
    top_k_event = int(g_cfg.get("top_k_event_type", 20))
    final_outputs: dict | None = None

    model.eval()
    for t, t_prev in zip(steps[:-1], steps[1:]):
        timesteps = torch.full(
            (batch["event_type"].shape[0],),
            int(t),
            dtype=torch.long,
            device=batch["event_type"].device,
        )
        batch["diffusion_t"] = timesteps
        outputs = model(batch, mode="pretrain")
        final_outputs = outputs

        sampled_event = _sample_discrete(
            outputs["event_type_logits"],
            temperature=temperature_event,
            top_k=top_k_event,
        )
        batch["event_type"] = torch.where(suffix_mask, sampled_event, batch["event_type"])

        cat_logits = outputs.get("cat_logits") or []
        if cat_logits and batch["cat_features"].shape[-1] > 0:
            updated_cat = batch["cat_features"].clone()
            for j, logits_j in enumerate(cat_logits):
                sampled_cat = _sample_discrete(
                    logits_j,
                    temperature=temperature_cat,
                    top_k=None,
                )
                updated_cat[:, :, j] = torch.where(
                    suffix_mask,
                    sampled_cat,
                    updated_cat[:, :, j],
                )
            batch["cat_features"] = updated_cat

        alpha_t = schedule.alpha_bar(timesteps, dtype=batch["time_delta"].dtype)
        prev_steps = torch.full_like(timesteps, int(t_prev))
        alpha_prev = schedule.alpha_bar(prev_steps, dtype=batch["time_delta"].dtype)
        time_prev = _ddim_update(
            outputs["time_delta_pred"].squeeze(-1),
            outputs.get("time_delta_eps_pred", None).squeeze(-1)
            if outputs.get("time_delta_eps_pred") is not None
            else None,
            batch["time_delta"],
            alpha_t,
            alpha_prev,
        )
        batch["time_delta"] = torch.where(suffix_mask, time_prev, batch["time_delta"])

        if "num_pred" in outputs and batch["num_features"].shape[-1] > 0:
            num_prev = _ddim_update(
                outputs["num_pred"],
                outputs.get("num_eps_pred"),
                batch["num_features"],
                alpha_t,
                alpha_prev,
            )
            batch["num_features"] = torch.where(
                suffix_mask.unsqueeze(-1),
                num_prev,
                batch["num_features"],
            )

        for key in ("event_type", "time_delta"):
            batch[key] = torch.where(prefix_mask, prefix_clean[key], batch[key])
        if batch["num_features"].shape[-1] > 0:
            batch["num_features"] = torch.where(
                prefix_mask.unsqueeze(-1),
                prefix_clean["num_features"],
                batch["num_features"],
            )
        if batch["cat_features"].shape[-1] > 0:
            batch["cat_features"] = torch.where(
                prefix_mask.unsqueeze(-1),
                prefix_clean["cat_features"],
                batch["cat_features"],
            )

    return {
        "generated_batch": batch,
        "final_outputs": final_outputs or {},
        "prefix_mask": prefix_mask,
        "suffix_mask": suffix_mask,
        "target_mask": generation_batch["target_mask"].to(batch["event_type"].device),
        "targets": {
            k: (v.to(batch["event_type"].device) if isinstance(v, torch.Tensor) else v)
            for k, v in generation_batch["targets"].items()
        },
        "sample_ids": generation_batch["sample_ids"],
    }


def _invert_vocab(mapping: dict | None) -> dict[int, str]:
    if not isinstance(mapping, dict):
        return {}
    return {int(v): str(k) for k, v in mapping.items() if isinstance(v, int)}


def _unscale(value: float, params: dict) -> float:
    scale = float(params.get("scale", 1.0)) or 1.0
    center = float(params.get("center", 0.0))
    return float(value) * scale + center


def _inverse_time(value: float, preprocessor: "EventPreprocessor") -> float:
    unscaled = _unscale(value, preprocessor.scaler_params.get("time_delta", {}))
    if getattr(preprocessor, "time_transform", "log1p") == "log1p":
        return float(np.expm1(np.clip(unscaled, -50.0, 50.0)))
    return float(unscaled)


def _inverse_num(value: float, col: str, preprocessor: "EventPreprocessor") -> float:
    unscaled = _unscale(value, preprocessor.scaler_params.get(col, {}))
    amount_cols = set(getattr(preprocessor, "_amount_cols", set()))
    if col in amount_cols:
        transform = getattr(preprocessor, "amount_transform", "robust_scaler")
        if transform == "log1p":
            return float(np.expm1(np.clip(unscaled, -50.0, 50.0)))
        if transform == "sign_log1p":
            return float(np.sign(unscaled) * np.expm1(np.clip(abs(unscaled), 0.0, 50.0)))
    return float(unscaled)


def decode_generated_suffix(
    generated: dict,
    preprocessor: "EventPreprocessor",
    config: dict,
) -> list[dict]:
    batch = generated["generated_batch"]
    suffix_mask = generated["suffix_mask"].detach().cpu()
    event_type = batch["event_type"].detach().cpu()
    time_delta = batch["time_delta"].detach().cpu()
    num_features = batch["num_features"].detach().cpu()
    cat_features = batch["cat_features"].detach().cpu()
    entity_ids = batch.get("entity_id", [str(i) for i in range(event_type.shape[0])])
    sample_ids = generated.get("sample_ids", [0] * event_type.shape[0])

    event_vocab = _invert_vocab(preprocessor.vocab.get(preprocessor.event_type_col))
    cat_vocabs = {
        col: _invert_vocab(preprocessor.vocab.get(col))
        for col in preprocessor.categorical_cols
    }
    num_names = list(preprocessor.numerical_cols) + get_derived_time_feature_names(config)

    rows: list[dict] = []
    for i in range(event_type.shape[0]):
        suffix_positions = torch.where(suffix_mask[i])[0].tolist()
        for step, pos in enumerate(suffix_positions):
            event_id = int(event_type[i, pos].item())
            time_norm = float(time_delta[i, pos].item())
            row = {
                "entity_id": entity_ids[i],
                "sample_id": int(sample_ids[i]),
                "step": int(step),
                "position": int(pos),
                "event_type_id": event_id,
                "event_type_token": event_vocab.get(event_id, str(event_id)),
                "time_delta_normalized": time_norm,
                "time_delta": _inverse_time(time_norm, preprocessor),
            }
            for j, col in enumerate(num_names[: num_features.shape[-1]]):
                value = float(num_features[i, pos, j].item())
                row[f"{col}_normalized"] = value
                if j < len(preprocessor.numerical_cols):
                    row[col] = _inverse_num(value, col, preprocessor)
            for j, col in enumerate(preprocessor.categorical_cols[: cat_features.shape[-1]]):
                cat_id = int(cat_features[i, pos, j].item())
                row[f"{col}_id"] = cat_id
                row[f"{col}_token"] = cat_vocabs.get(col, {}).get(cat_id, str(cat_id))
            rows.append(row)
    return rows
