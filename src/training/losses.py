from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _masked_cross_entropy(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits[mask], targets[mask])


def _masked_huber(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return F.huber_loss(pred[mask], target[mask])


def compute_pretraining_loss(
    outputs: dict,
    targets: dict,
    masks: dict,
    config: dict,
) -> dict:
    loss_cfg = config.get("loss", {})
    lambda_type = loss_cfg.get("lambda_event_type", 1.0)
    lambda_time = loss_cfg.get("lambda_time", 1.0)
    lambda_num = loss_cfg.get("lambda_num", 0.5)
    lambda_cat = loss_cfg.get("lambda_cat", 0.5)
    lambda_exist = loss_cfg.get("lambda_exist", 0.1)

    device = outputs["event_type_logits"].device

    # L_type
    L_type = _masked_cross_entropy(
        outputs["event_type_logits"],
        targets["event_type"],
        masks["event_type"],
    )

    # L_time
    time_pred = outputs["time_delta_pred"].squeeze(-1)  # [B, L]
    L_time = _masked_huber(time_pred, targets["time_delta"], masks["time_delta"])

    # L_num
    num_pred = outputs.get("num_pred")
    if num_pred is not None:
        num_mask = masks["num_features"]  # [B, L]
        if num_mask.any():
            m = num_mask.unsqueeze(-1).expand_as(num_pred)
            L_num = F.huber_loss(num_pred[m], targets["num_features"][m])
        else:
            L_num = torch.zeros((), device=device)
    else:
        L_num = torch.zeros((), device=device)

    # L_cat
    cat_logits_list = outputs.get("cat_logits") or []
    if cat_logits_list:
        cat_targets = targets["cat_features"]   # [B, L, N_cat]
        cat_mask = masks["cat_features"]        # [B, L, N_cat]
        cat_losses = []
        for j, logits_j in enumerate(cat_logits_list):
            mask_j = cat_mask[:, :, j] if cat_mask.dim() == 3 else cat_mask
            if mask_j.any():
                cat_losses.append(
                    F.cross_entropy(logits_j[mask_j], cat_targets[:, :, j][mask_j])
                )
        L_cat = torch.stack(cat_losses).mean() if cat_losses else torch.zeros((), device=device)
    else:
        L_cat = torch.zeros((), device=device)

    # L_exist
    existence_pred = outputs["existence_logits"].squeeze(-1)  # [B, L]
    event_level_targets = masks["event_level"].float()
    attention_mask = masks["attention_mask"]  # [B, L]
    if attention_mask.any():
        L_exist = F.binary_cross_entropy_with_logits(
            existence_pred[attention_mask],
            event_level_targets[attention_mask],
        )
    else:
        L_exist = torch.zeros((), device=device)

    L_total = (
        lambda_type * L_type
        + lambda_time * L_time
        + lambda_num * L_num
        + lambda_cat * L_cat
        + lambda_exist * L_exist
    )

    return {
        "total": L_total,
        "event_type": L_type,
        "time_delta": L_time,
        "numerical": L_num,
        "categorical": L_cat,
        "existence": L_exist,
    }
