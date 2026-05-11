from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from src.corruption.categorical import _NUM_SPECIAL


def _safe_rate(numerator: torch.Tensor, mask: torch.Tensor) -> float:
    denom = int(mask.sum().item())
    if denom == 0:
        return float("nan")
    return float(numerator[mask].float().mean().item())


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return float("nan")
    return float(torch.abs(pred[mask] - target[mask]).mean().item())


def _bincount_distribution(ids: torch.Tensor, vocab_size: int) -> np.ndarray | None:
    flat = ids.detach().cpu().long().reshape(-1)
    flat = flat[(flat >= 0) & (flat < vocab_size)]
    if flat.numel() == 0:
        return None
    counts = torch.bincount(flat, minlength=vocab_size).float().numpy()
    total = float(counts.sum())
    return counts / total if total > 0 else None


def _js_divergence(p: np.ndarray | None, q: np.ndarray | None) -> float:
    if p is None or q is None:
        return float("nan")
    if p.shape != q.shape:
        raise ValueError("distribution shapes must match")
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        valid = a > 0
        return float(np.sum(a[valid] * np.log(a[valid] / np.clip(b[valid], 1e-12, None))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _transition_counts(tokens: torch.Tensor, mask: torch.Tensor) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    tokens_cpu = tokens.detach().cpu().long()
    mask_cpu = mask.detach().cpu().bool()
    for row_tokens, row_mask in zip(tokens_cpu, mask_cpu, strict=True):
        seq = row_tokens[row_mask].tolist()
        counts.update((int(a), int(b)) for a, b in zip(seq[:-1], seq[1:], strict=False))
    return counts


def _counter_js(left: Counter[tuple[int, int]], right: Counter[tuple[int, int]]) -> float:
    if not left or not right:
        return float("nan")
    keys = sorted(set(left) | set(right))
    left_total = float(sum(left.values()))
    right_total = float(sum(right.values()))
    p = np.array([left.get(key, 0) / left_total for key in keys], dtype=np.float64)
    q = np.array([right.get(key, 0) / right_total for key in keys], dtype=np.float64)
    return _js_divergence(p, q)


def _consecutive_duplicate_rate(tokens: torch.Tensor, mask: torch.Tensor) -> float:
    duplicate = 0
    total_pairs = 0
    tokens_cpu = tokens.detach().cpu().long()
    mask_cpu = mask.detach().cpu().bool()
    for row_tokens, row_mask in zip(tokens_cpu, mask_cpu, strict=True):
        seq = row_tokens[row_mask]
        if seq.numel() < 2:
            continue
        duplicate += int((seq[1:] == seq[:-1]).sum().item())
        total_pairs += int(seq.numel() - 1)
    if total_pairs == 0:
        return float("nan")
    return duplicate / total_pairs


def _topk_recall(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, k: int) -> float:
    if int(mask.sum().item()) == 0:
        return float("nan")
    k = min(int(k), int(logits.shape[-1]))
    topk = torch.topk(logits, k=k, dim=-1).indices
    hits = (topk == target.unsqueeze(-1)).any(dim=-1)
    return float(hits[mask].float().mean().item())


def compute_generation_metrics(generated: dict, vocab_info: dict | None = None) -> dict[str, float | int]:
    """Compute suffix-generation metrics against held-out future positions.

    target_mask marks generated suffix positions where the original sequence has a
    real future event. Metrics with a target compare generated values to that
    held-out future; token sanity metrics are computed over the whole suffix.
    """
    vocab_info = vocab_info or {}
    batch = generated["generated_batch"]
    targets = generated.get("targets", {})
    suffix_mask = generated["suffix_mask"].bool()
    target_mask = generated["target_mask"].bool() & suffix_mask
    final_outputs = generated.get("final_outputs", {}) or {}

    event_type = batch["event_type"]
    target_event = targets.get("event_type")
    event_vocab_size = int(vocab_info.get("event_type_vocab_size", max(int(event_type.max().item()) + 1, 1)))

    metrics: dict[str, float | int] = {
        "num_sequences": int(event_type.shape[0]),
        "suffix_positions": int(suffix_mask.sum().item()),
        "target_positions": int(target_mask.sum().item()),
        "invalid_event_token_rate": _safe_rate(event_type < _NUM_SPECIAL, suffix_mask),
        "duplicate_event_rate": _consecutive_duplicate_rate(event_type, suffix_mask),
    }

    if int(suffix_mask.sum().item()) > 0:
        suffix_ids = event_type[suffix_mask]
        metrics["event_type_unique_ratio"] = float(
            torch.unique(suffix_ids).numel() / max(1, suffix_ids.numel())
        )
    else:
        metrics["event_type_unique_ratio"] = float("nan")

    if target_event is not None and int(target_mask.sum().item()) > 0:
        metrics["event_type_accuracy"] = float(
            (event_type[target_mask] == target_event[target_mask]).float().mean().item()
        )
        if "event_type_logits" in final_outputs:
            metrics["event_type_top5_recall"] = _topk_recall(
                final_outputs["event_type_logits"],
                target_event,
                target_mask,
                k=5,
            )
        metrics["event_type_js_divergence"] = _js_divergence(
            _bincount_distribution(event_type[target_mask], event_vocab_size),
            _bincount_distribution(target_event[target_mask], event_vocab_size),
        )
        metrics["transition_js_divergence"] = _counter_js(
            _transition_counts(event_type, target_mask),
            _transition_counts(target_event, target_mask),
        )
    else:
        metrics["event_type_accuracy"] = float("nan")
        metrics["event_type_top5_recall"] = float("nan")
        metrics["event_type_js_divergence"] = float("nan")
        metrics["transition_js_divergence"] = float("nan")

    if "time_delta" in batch and "time_delta" in targets:
        metrics["time_delta_mae_normalized"] = _masked_mae(
            batch["time_delta"],
            targets["time_delta"],
            target_mask,
        )

    if "num_features" in batch and "num_features" in targets and batch["num_features"].shape[-1] > 0:
        num_mask = target_mask.unsqueeze(-1).expand_as(batch["num_features"])
        metrics["num_mae_normalized"] = _masked_mae(
            batch["num_features"],
            targets["num_features"],
            num_mask,
        )

    if "cat_features" in batch and batch["cat_features"].shape[-1] > 0:
        cat_mask_suffix = suffix_mask.unsqueeze(-1).expand_as(batch["cat_features"])
        metrics["invalid_cat_token_rate"] = _safe_rate(
            batch["cat_features"] < _NUM_SPECIAL,
            cat_mask_suffix,
        )
        if "cat_features" in targets and int(target_mask.sum().item()) > 0:
            cat_mask_target = target_mask.unsqueeze(-1).expand_as(batch["cat_features"])
            metrics["categorical_accuracy_mean"] = float(
                (batch["cat_features"][cat_mask_target] == targets["cat_features"][cat_mask_target])
                .float()
                .mean()
                .item()
            )
        else:
            metrics["categorical_accuracy_mean"] = float("nan")
    else:
        metrics["invalid_cat_token_rate"] = float("nan")
        metrics["categorical_accuracy_mean"] = float("nan")

    return metrics
