from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from src.corruption.pipeline import CorruptionPipeline
from src.models.dme_encoder import DMEEncoder
from src.training.losses import compute_pretraining_loss
from src.training.optim import build_pretrain_optimizer, get_linear_warmup_scheduler
from src.utils.logging import MetricsLogger

# Maps loss_dict keys → config["loss"] lambda keys
_LOSS_KEY_TO_LAMBDA = {
    "event_type": "lambda_event_type",
    "time_delta": "lambda_time",
    "numerical": "lambda_num",
    "categorical": "lambda_cat",
    "existence": "lambda_exist",
}

_CAL_COMPONENTS = list(_LOSS_KEY_TO_LAMBDA.keys())


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _save_checkpoint(
    model: DMEEncoder,
    optimizer,
    epoch: int,
    val_loss: float,
    config: dict,
    path: str,
    vocab_info: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": config,
            "vocab_info": vocab_info,
        },
        path,
    )


def _apply_calibration(
    component_sums: dict[str, float],
    warmup_steps: int,
    config: dict,
    output_dir: str,
) -> dict:
    """Auto-apply λ-weights if max/min ratio > 5×. Returns (possibly updated) config copy."""
    import json as _json

    means = {k: component_sums[k] / max(1, warmup_steps) for k in _CAL_COMPONENTS}
    ref = means.get("event_type") or 1.0
    nonzero = [v for v in means.values() if v > 0]
    max_min_ratio = (max(nonzero) / min(nonzero)) if len(nonzero) >= 2 else 1.0

    loss_cfg = config.get("loss", {})
    recommended = {
        _LOSS_KEY_TO_LAMBDA[k]: ref / max(v, 1e-8)
        for k, v in means.items()
    }

    print("\n=== Loss Calibration ===")
    for k, v in means.items():
        lk = _LOSS_KEY_TO_LAMBDA[k]
        print(f"  {k:<16}: mean={v:.6f}  rec_λ={recommended[lk]:.4f}  "
              f"cur_λ={loss_cfg.get(lk, '—')}")
    print(f"  max/min ratio: {max_min_ratio:.1f}x")

    applied = max_min_ratio > 5.0
    if applied:
        print("  → Auto-applying calibrated lambdas (ratio > 5×)")
        config = {**config, "loss": {**loss_cfg, **recommended}}
    print("=======================\n")

    os.makedirs(output_dir, exist_ok=True)
    cal_path = os.path.join(output_dir, "calibrated_lambdas.json")
    with open(cal_path, "w") as f:
        _json.dump(
            {
                "component_means": means,
                "recommended_lambdas": recommended,
                "max_min_ratio": max_min_ratio,
                "auto_applied": applied,
            },
            f,
            indent=2,
        )
    print(f"  Calibration report saved → {cal_path}")

    return config


def evaluate_pretrain(
    model: DMEEncoder,
    val_loader: DataLoader,
    corruption_pipeline: CorruptionPipeline,
    config: dict,
    device: torch.device,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_loss_event_type = 0.0
    total_loss_time_delta = 0.0
    correct_types = 0
    total_type_positions = 0
    sum_mae = 0.0
    total_time_positions = 0
    n_batches = 0

    with torch.no_grad():
        for clean_batch in val_loader:
            clean_batch = _batch_to_device(clean_batch, device)
            corrupted_batch, targets, masks = corruption_pipeline(clean_batch)
            masks["attention_mask"] = corrupted_batch["attention_mask"]

            outputs = model(corrupted_batch, mode="pretrain")
            loss_dict = compute_pretraining_loss(outputs, targets, masks, config)

            total_loss += loss_dict["total"].item()
            total_loss_event_type += loss_dict["event_type"].item()
            total_loss_time_delta += loss_dict["time_delta"].item()

            type_mask = masks["event_type"]
            if type_mask.any():
                preds = outputs["event_type_logits"][type_mask].argmax(dim=-1)
                trues = targets["event_type"][type_mask]
                correct_types += (preds == trues).sum().item()
                total_type_positions += int(type_mask.sum().item())

            time_mask = masks["time_delta"]
            if time_mask.any():
                time_pred = outputs["time_delta_pred"].squeeze(-1)[time_mask]
                time_true = targets["time_delta"][time_mask]
                sum_mae += (time_pred - time_true).abs().sum().item()
                total_time_positions += int(time_mask.sum().item())

            n_batches += 1

    model.train()
    return {
        "loss_total": total_loss / max(1, n_batches),
        "loss_event_type": total_loss_event_type / max(1, n_batches),
        "loss_time_delta": total_loss_time_delta / max(1, n_batches),
        "event_type_accuracy": correct_types / max(1, total_type_positions),
        "time_delta_mae": sum_mae / max(1, total_time_positions),
    }


def pretrain(
    model: DMEEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    corruption_pipeline: CorruptionPipeline,
    config: dict,
    output_dir: str,
    device: torch.device,
    logger: MetricsLogger,
    vocab_info: Optional[dict] = None,
) -> str:
    training_cfg = config.get("training", {})
    num_epochs: int = training_cfg.get("num_epochs_pretrain", 30)
    grad_clip: float = training_cfg.get("gradient_clip_val", 1.0)
    log_every: int = training_cfg.get("log_every_n_steps", 50)
    mixed_precision: bool = training_cfg.get("mixed_precision", False)

    cal_cfg = config.get("loss_calibration", {})
    cal_enabled: bool = cal_cfg.get("enabled", True)
    cal_warmup_steps: int = cal_cfg.get("warmup_steps", 1000)

    # Optimizer + linear-warmup / cosine-decay scheduler (stepped per batch)
    optimizer, _ = build_pretrain_optimizer(model, config)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(training_cfg.get("warmup_ratio", 0.05) * total_steps)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # Mixed precision: fp16 on CUDA, autocast-only on MPS, nothing on CPU
    use_amp = mixed_precision and device.type in ("cuda", "mps")
    use_scaler = mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None

    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    model.train()

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(output_dir, "best_checkpoint.pt")
    global_step = 0

    cal_sums: dict[str, float] = defaultdict(float)
    cal_steps_done = 0
    cal_logged = False

    for epoch in range(num_epochs):
        for clean_batch in train_loader:
            clean_batch = _batch_to_device(clean_batch, device)
            corrupted_batch, targets, masks = corruption_pipeline(clean_batch)
            masks["attention_mask"] = corrupted_batch["attention_mask"]

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(corrupted_batch, mode="pretrain")
                loss_dict = compute_pretraining_loss(outputs, targets, masks, config)

            if use_scaler:
                scaler.scale(loss_dict["total"]).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict["total"].backward()
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Accumulate calibration statistics
            if cal_enabled and not cal_logged:
                for k in _CAL_COMPONENTS:
                    cal_sums[k] += loss_dict[k].item()
                cal_steps_done += 1
                if cal_steps_done >= cal_warmup_steps:
                    config = _apply_calibration(cal_sums, cal_steps_done, config, output_dir)
                    cal_logged = True

            if global_step % log_every == 0:
                logger.log_step(
                    global_step,
                    {f"train/loss_{k}": v.item() for k, v in loss_dict.items()},
                )

        # Epoch-level validation
        val_metrics = evaluate_pretrain(
            model, val_loader, corruption_pipeline, config, device
        )
        logger.log_epoch(epoch, {f"val/{k}": v for k, v in val_metrics.items()})
        model.train()

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            _save_checkpoint(
                model, optimizer, epoch, best_val_loss, config,
                best_checkpoint_path, vocab_info,
            )

    return best_checkpoint_path
