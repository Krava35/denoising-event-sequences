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


def compute_diffusion_pretraining_loss(
    outputs: dict,
    targets: dict,
    masks: dict,
    config: dict,
) -> dict:
    loss_cfg = config.get("loss", {})
    lambda_type = float(loss_cfg.get("lambda_event_type", 1.0))
    lambda_time = float(loss_cfg.get("lambda_time", 1.0))
    lambda_num = float(loss_cfg.get("lambda_num", 0.5))
    lambda_cat = float(loss_cfg.get("lambda_cat", 0.5))
    lambda_time_eps = float(loss_cfg.get("lambda_time_eps", 0.1))
    lambda_num_eps = float(loss_cfg.get("lambda_num_eps", 0.1))
    lambda_d3pm_event_prev = float(
        config.get("d3pm", {}).get("loss_weight_event_type_prev", 0.0)
    )

    device = outputs["event_type_logits"].device

    L_type = _masked_cross_entropy(
        outputs["event_type_logits"],
        targets["event_type"],
        masks["event_type"],
    )

    time_pred = outputs["time_delta_pred"].squeeze(-1)
    L_time = _masked_huber(time_pred, targets["time_delta"], masks["time_delta"])

    time_eps_pred = outputs["time_delta_eps_pred"].squeeze(-1)
    time_eps_mask = masks.get("time_delta_eps", masks["time_delta"])
    if time_eps_mask.any():
        L_time_eps = F.mse_loss(
            time_eps_pred[time_eps_mask],
            targets["time_delta_eps"][time_eps_mask],
        )
    else:
        L_time_eps = torch.zeros((), device=device)

    num_pred = outputs.get("num_pred")
    if num_pred is not None:
        num_mask = masks["num_features"]
        if num_mask.any():
            m = num_mask.unsqueeze(-1).expand_as(num_pred)
            L_num = F.huber_loss(num_pred[m], targets["num_features"][m])
        else:
            L_num = torch.zeros((), device=device)
    else:
        L_num = torch.zeros((), device=device)

    num_eps_pred = outputs.get("num_eps_pred")
    if num_eps_pred is not None:
        num_eps_mask = masks.get("num_features_eps", masks["num_features"])
        if num_eps_mask.any():
            m = num_eps_mask.unsqueeze(-1).expand_as(num_eps_pred)
            L_num_eps = F.mse_loss(num_eps_pred[m], targets["num_features_eps"][m])
        else:
            L_num_eps = torch.zeros((), device=device)
    else:
        L_num_eps = torch.zeros((), device=device)

    cat_logits_list = outputs.get("cat_logits") or []
    if cat_logits_list:
        cat_targets = targets["cat_features"]
        cat_mask = masks["cat_features"]
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

    has_d3pm_event_prev = (
        lambda_d3pm_event_prev > 0.0
        and "event_type_prev_logits" in outputs
        and "d3pm_event_type_prev" in targets
        and "d3pm_event_type_prev" in masks
    )
    if has_d3pm_event_prev:
        L_d3pm_event_prev = _masked_cross_entropy(
            outputs["event_type_prev_logits"],
            targets["d3pm_event_type_prev"],
            masks["d3pm_event_type_prev"],
        )
    else:
        L_d3pm_event_prev = torch.zeros((), device=device)

    L_total = (
        lambda_type * L_type
        + lambda_time * L_time
        + lambda_num * L_num
        + lambda_cat * L_cat
        + lambda_time_eps * L_time_eps
        + lambda_num_eps * L_num_eps
        + lambda_d3pm_event_prev * L_d3pm_event_prev
    )

    result = {
        "total": L_total,
        "event_type": L_type,
        "time_delta": L_time,
        "numerical": L_num,
        "categorical": L_cat,
        "time_delta_eps": L_time_eps,
        "numerical_eps": L_num_eps,
    }
    if has_d3pm_event_prev:
        result["d3pm_event_type_prev"] = L_d3pm_event_prev
    return result


def compute_forecast_loss(
    outputs: dict,
    targets: dict,
    config: dict,
) -> dict:
    f_cfg = config.get("forecasting", {})
    lambda_event = float(f_cfg.get("lambda_event_type_profile", 1.0))
    lambda_count = float(f_cfg.get("lambda_count", 0.5))
    lambda_amount = float(f_cfg.get("lambda_amount", 0.5))
    lambda_gap = float(f_cfg.get("lambda_gap", 0.3))
    lambda_cat = float(f_cfg.get("lambda_cat_profile", 0.5))

    device = outputs["future_event_type_profile"].device

    L_event = F.huber_loss(
        outputs["future_event_type_profile"],
        targets["future_event_type_profile"].to(device),
    )
    L_count = F.cross_entropy(
        outputs["future_count_bucket_logits"],
        targets["future_count_bucket"].to(device).long(),
    )
    L_amount = F.huber_loss(
        outputs["future_amount_stats"],
        targets["future_amount_stats"].to(device),
    )
    L_gap = F.cross_entropy(
        outputs["future_gap_bucket_logits"],
        targets["future_gap_bucket"].to(device).long(),
    )

    cat_outputs = outputs.get("future_cat_profiles") or []
    cat_targets = targets.get("future_cat_profiles") or []
    cat_losses = [
        F.huber_loss(pred, target.to(device))
        for pred, target in zip(cat_outputs, cat_targets)
    ]
    L_cat = torch.stack(cat_losses).mean() if cat_losses else torch.zeros((), device=device)

    L_total = (
        lambda_event * L_event
        + lambda_count * L_count
        + lambda_amount * L_amount
        + lambda_gap * L_gap
        + lambda_cat * L_cat
    )

    return {
        "total": L_total,
        "event_type_profile": L_event,
        "count_bucket": L_count,
        "amount_stats": L_amount,
        "gap_bucket": L_gap,
        "cat_profiles": L_cat,
    }
