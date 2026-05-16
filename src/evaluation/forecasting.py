from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


def _to_numpy_int(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().long().numpy()


def _classification_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    pred_np = _to_numpy_int(pred)
    target_np = _to_numpy_int(target)
    return {
        f"{prefix}_accuracy": float(accuracy_score(target_np, pred_np)),
        f"{prefix}_macro_f1": float(
            f1_score(target_np, pred_np, average="macro", zero_division=0)
        ),
    }


def _event_profile_to_frequency(profile: torch.Tensor, forecast_stats: dict) -> torch.Tensor:
    global_freq = torch.as_tensor(
        forecast_stats["event_type_global_freq"],
        dtype=profile.dtype,
        device=profile.device,
    )
    global_freq = global_freq.clamp_min(1e-12)
    values = torch.exp(profile.clamp(-30.0, 30.0)) * global_freq
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _normalise_profile(values: torch.Tensor) -> torch.Tensor:
    clipped = values.nan_to_num(0.0).clamp_min(0.0)
    denom = clipped.sum(dim=-1, keepdim=True)
    softmax = torch.softmax(values.nan_to_num(0.0), dim=-1)
    return torch.where(denom > 0.0, clipped / denom.clamp_min(1e-12), softmax)


def _topk_recall(pred: torch.Tensor, target: torch.Tensor, top_k: int) -> float:
    if pred.numel() == 0 or target.numel() == 0:
        return float("nan")
    k = min(top_k, pred.shape[-1], target.shape[-1])
    pred_top = pred.topk(k, dim=-1).indices
    target_top = target.topk(k, dim=-1).indices
    recalls = []
    for pred_row, target_row in zip(pred_top, target_top):
        pred_set = set(pred_row.detach().cpu().tolist())
        target_set = set(target_row.detach().cpu().tolist())
        recalls.append(len(pred_set & target_set) / max(1, len(target_set)))
    return float(np.mean(recalls))


def compute_forecast_quality_metrics(
    outputs: dict,
    targets: dict,
    forecast_stats: dict,
    *,
    top_k: int = 5,
) -> dict[str, float]:
    """Compute aggregate future-behavior forecast metrics for one batch."""
    device = outputs["future_event_type_profile"].device
    target_event_profile = targets["future_event_type_profile"].to(device)
    pred_event_freq = _event_profile_to_frequency(
        outputs["future_event_type_profile"],
        forecast_stats,
    )
    target_event_freq = _event_profile_to_frequency(target_event_profile, forecast_stats)

    count_pred = outputs["future_count_bucket_logits"].argmax(dim=-1)
    count_target = targets["future_count_bucket"].to(device).long()
    gap_pred = outputs["future_gap_bucket_logits"].argmax(dim=-1)
    gap_target = targets["future_gap_bucket"].to(device).long()

    metrics = {
        **_classification_metrics(count_pred, count_target, "future_count"),
        **_classification_metrics(gap_pred, gap_target, "future_gap"),
        "event_profile_cosine": float(
            F.cosine_similarity(pred_event_freq, target_event_freq, dim=-1).mean().item()
        ),
        "event_profile_mae": float((pred_event_freq - target_event_freq).abs().mean().item()),
        "event_profile_top5_recall": _topk_recall(pred_event_freq, target_event_freq, top_k),
        "amount_stats_mae": float(
            (
                outputs["future_amount_stats"]
                - targets["future_amount_stats"].to(device)
            )
            .abs()
            .mean()
            .item()
        ),
    }

    cat_cosines: list[float] = []
    cat_maes: list[float] = []
    for pred, target in zip(
        outputs.get("future_cat_profiles") or [],
        targets.get("future_cat_profiles") or [],
    ):
        target = target.to(device)
        pred_profile = _normalise_profile(pred)
        target_profile = _normalise_profile(target)
        cat_cosines.append(
            float(F.cosine_similarity(pred_profile, target_profile, dim=-1).mean().item())
        )
        cat_maes.append(float((pred_profile - target_profile).abs().mean().item()))

    metrics["cat_profile_cosine_mean"] = float(np.mean(cat_cosines)) if cat_cosines else 0.0
    metrics["cat_profile_mae_mean"] = float(np.mean(cat_maes)) if cat_maes else 0.0
    return metrics


def _prefix_event_profile(batch: dict, forecast_stats: dict, device: torch.device) -> torch.Tensor:
    event_type = batch["event_type"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    vocab_size = int(forecast_stats["event_type_vocab_size"])
    global_freq = torch.as_tensor(
        forecast_stats["event_type_global_freq"],
        dtype=torch.float,
        device=device,
    ).clamp_min(1e-12)

    profiles = []
    for row, mask in zip(event_type, attention_mask):
        values = row[mask].clamp(0, vocab_size - 1)
        counts = torch.bincount(values, minlength=vocab_size).float() + 1e-6
        freq = counts / counts.sum().clamp_min(1e-12)
        profiles.append(torch.log(freq / global_freq))
    return torch.stack(profiles)


def compute_forecast_baseline_metrics(
    batch: dict,
    targets: dict,
    forecast_stats: dict,
    *,
    top_k: int = 5,
) -> dict[str, float]:
    """Compute simple train-stat and prefix-history baselines for one batch."""
    device = targets["future_event_type_profile"].device
    batch_size = int(targets["future_event_type_profile"].shape[0])
    event_vocab_size = int(forecast_stats["event_type_vocab_size"])
    count_buckets = len(forecast_stats.get("count_bucket_edges", [])) + 1
    gap_buckets = len(forecast_stats.get("gap_bucket_edges", [])) + 1
    cat_vocab_sizes = forecast_stats.get("cat_vocab_sizes", [])

    def bucket_logits(bucket: int, num_buckets: int) -> torch.Tensor:
        logits = torch.zeros(batch_size, num_buckets, device=device)
        logits[:, int(np.clip(bucket, 0, num_buckets - 1))] = 1.0
        return logits

    zero_event_profile = torch.zeros(batch_size, event_vocab_size, device=device)
    zero_amount = torch.zeros(batch_size, 4, device=device)
    global_outputs = {
        "future_event_type_profile": zero_event_profile,
        "future_count_bucket_logits": bucket_logits(
            int(forecast_stats.get("median_count_bucket", 0)),
            count_buckets,
        ),
        "future_amount_stats": zero_amount,
        "future_gap_bucket_logits": bucket_logits(
            int(forecast_stats.get("median_gap_bucket", 0)),
            gap_buckets,
        ),
        "future_cat_profiles": [
            torch.as_tensor(freq, dtype=torch.float, device=device).expand(batch_size, size)
            for freq, size in zip(forecast_stats.get("cat_global_freq", []), cat_vocab_sizes)
        ],
    }
    prefix_outputs = {
        **global_outputs,
        "future_event_type_profile": _prefix_event_profile(batch, forecast_stats, device),
    }

    global_metrics = compute_forecast_quality_metrics(
        global_outputs,
        targets,
        forecast_stats,
        top_k=top_k,
    )
    prefix_metrics = compute_forecast_quality_metrics(
        prefix_outputs,
        targets,
        forecast_stats,
        top_k=top_k,
    )
    return {
        **{f"baseline_global_{k}": v for k, v in global_metrics.items()},
        **{f"baseline_prefix_{k}": v for k, v in prefix_metrics.items()},
    }
