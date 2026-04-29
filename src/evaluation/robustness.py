from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.corruption.pipeline import CorruptionPipeline
from src.evaluation.classification import compute_classification_metrics

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from src.models.dme_encoder import DMEEncoder


def _scale_corruption_config(cfg: dict, scale: float) -> dict:
    """Deep-copy corruption sub-dict and multiply all probability fields by scale."""
    new_cfg = copy.deepcopy(cfg)
    _PROB_KEYS = [
        ("event_type", "selected_prob"),
        ("time_noise", "corruption_prob"),
        ("numerical_noise", "corruption_prob"),
        ("categorical_features", "mask_prob"),
        ("categorical_features", "random_replace_prob"),
        ("event_level_masking", "prob"),
    ]
    for section, key in _PROB_KEYS:
        sec = new_cfg.get(section, {})
        if isinstance(sec, dict) and key in sec:
            sec[key] = float(sec[key]) * scale
    return new_cfg


def evaluate_robustness(
    model: DMEEncoder,
    test_loader: DataLoader,
    corruption_pipeline: CorruptionPipeline,
    device: torch.device,
    corruption_levels: list[float] | None = None,
) -> dict:
    """Evaluate model classification performance at varying corruption intensities.

    For each level, all corruption probabilities in the pipeline config are scaled
    by that factor. Level 0.0 → clean evaluation; level 1.0 → original config.

    Args:
        model: fine-tuned DMEEncoder.
        test_loader: DataLoader in eval mode (produces clean batches with labels).
        corruption_pipeline: reference pipeline; its _cfg is used as the base config.
        device: torch device.
        corruption_levels: list of scale factors to evaluate.

    Returns:
        dict with keys:
            "per_level": {level: metrics_dict}
            "corruption_levels": list of levels
    """
    if corruption_levels is None:
        corruption_levels = [0.0, 0.1, 0.2, 0.3]

    num_classes: int = model.classifier.classifier[-1].out_features
    base_cfg = corruption_pipeline._cfg  # corruption sub-dict

    per_level: dict[float, dict] = {}

    for level in corruption_levels:
        scaled_cfg = _scale_corruption_config(base_cfg, level)
        scaled_pipeline = CorruptionPipeline(
            config=scaled_cfg,
            transition_matrix=corruption_pipeline._tm,
            vocab_sizes=corruption_pipeline._vocab_sizes,
            time_transform=corruption_pipeline._time_transform,
        )

        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        model.eval()
        with torch.no_grad():
            for clean_batch in test_loader:
                clean_batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in clean_batch.items()
                }
                corrupted_batch, _, _ = scaled_pipeline(clean_batch)

                outputs = model(corrupted_batch, mode="finetune")
                probs = torch.softmax(outputs["logits"], dim=-1)  # [B, num_classes]
                labels = clean_batch["label"].long()              # [B]

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        probs_np = np.concatenate(all_probs, axis=0)    # [N, num_classes]
        labels_np = np.concatenate(all_labels, axis=0)  # [N]

        per_level[level] = compute_classification_metrics(labels_np, probs_np, num_classes)

    return {
        "per_level": per_level,
        "corruption_levels": corruption_levels,
    }


def plot_robustness_curve(
    results: dict,
    output_dir: str | Path,
    prefix: str = "",
    metric_key: str = "roc_auc",
) -> None:
    """Save a robustness curve (metric vs corruption level) to output_dir.

    Falls back to roc_auc_ovr if metric_key is not present (multiclass case).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = results["corruption_levels"]
    per_level = results["per_level"]

    values = []
    for lvl in levels:
        m = per_level[lvl]
        val = m.get(metric_key, m.get("roc_auc_ovr", float("nan")))
        values.append(val)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(levels, values, marker="o", linewidth=2)
    ax.set_xlabel("Corruption Level")
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title("Robustness Curve")
    ax.set_xticks(levels)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}robustness_curve.png", dpi=150)
    plt.close(fig)
