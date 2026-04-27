from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

if TYPE_CHECKING:
    from src.data.preprocessing import EventPreprocessor


def compute_reconstruction_metrics(
    outputs: dict,
    targets: dict,
    masks: dict,
    preprocessor: EventPreprocessor,
) -> dict:
    """Compute pretraining reconstruction quality metrics.

    Args:
        outputs: model outputs from forward(batch, mode="pretrain").
                 Keys: event_type_logits [B,L,V], time_delta_pred [B,L,1],
                       num_pred [B,L,N] (optional).
        targets: pre-corruption values from CorruptionPipeline.
                 Keys: event_type [B,L], time_delta [B,L],
                       num_features [B,L,N] (optional).
        masks: boolean tensors from CorruptionPipeline.
               Keys: event_type [B,L], time_delta [B,L],
                     num_features [B,L] (optional).
        preprocessor: fitted EventPreprocessor used for inverse time transform.

    Returns:
        dict with float metrics (NaN when no masked positions in batch).
    """
    metrics: dict[str, float] = {}

    # ── event_type accuracy and macro F1 ─────────────────────────────────────
    et_mask = masks["event_type"].detach().cpu()  # [B, L] bool
    if et_mask.any():
        et_logits = outputs["event_type_logits"].detach().cpu()  # [B, L, V]
        et_targets = targets["event_type"].detach().cpu()        # [B, L]

        preds = et_logits[et_mask].argmax(dim=-1).numpy()   # [M]
        trues = et_targets[et_mask].numpy()                  # [M]

        metrics["event_type_accuracy"] = float(accuracy_score(trues, preds))
        metrics["event_type_macro_f1"] = float(
            f1_score(trues, preds, average="macro", zero_division=0)
        )
    else:
        metrics["event_type_accuracy"] = float("nan")
        metrics["event_type_macro_f1"] = float("nan")

    # ── time_delta MAE in scaled (log) space ─────────────────────────────────
    td_mask = masks["time_delta"].detach().cpu()  # [B, L] bool
    if td_mask.any():
        td_pred = outputs["time_delta_pred"].squeeze(-1).detach().cpu()  # [B, L]
        td_target = targets["time_delta"].detach().cpu()                  # [B, L]

        pred_scaled = td_pred[td_mask].numpy()   # [M]
        true_scaled = td_target[td_mask].numpy() # [M]

        metrics["time_delta_mae_normalized"] = float(np.abs(pred_scaled - true_scaled).mean())

        # ── time_delta MAE in original space (inverse transform) ──────────────
        td_params = preprocessor.scaler_params["time_delta"]
        scale = td_params["scale"] if td_params["scale"] != 0.0 else 1.0
        center = td_params["center"]

        pred_unscaled = pred_scaled * scale + center
        true_unscaled = true_scaled * scale + center

        time_transform = getattr(preprocessor, "time_transform", "log1p")
        if time_transform == "log1p":
            pred_orig = np.expm1(pred_unscaled)
            true_orig = np.expm1(true_unscaled)
        else:
            pred_orig = pred_unscaled
            true_orig = true_unscaled

        metrics["time_delta_mae_original"] = float(np.abs(pred_orig - true_orig).mean())
    else:
        metrics["time_delta_mae_normalized"] = float("nan")
        metrics["time_delta_mae_original"] = float("nan")

    # ── numerical MAE (optional) ──────────────────────────────────────────────
    if "num_pred" in outputs:
        num_mask_raw = masks.get("num_features")
        if num_mask_raw is not None:
            num_mask_cpu = num_mask_raw.detach().cpu()  # [B, L] bool
            if num_mask_cpu.any():
                num_pred = outputs["num_pred"].detach().cpu()       # [B, L, N]
                num_target = targets["num_features"].detach().cpu() # [B, L, N]

                # Broadcast [B, L] mask to [B, L, N]
                mask_exp = num_mask_cpu.unsqueeze(-1).expand_as(num_pred)  # [B, L, N]

                pred_np = num_pred[mask_exp].numpy()
                true_np = num_target[mask_exp].numpy()

                metrics["num_mae"] = float(np.abs(pred_np - true_np).mean())
            else:
                metrics["num_mae"] = float("nan")

    return metrics
