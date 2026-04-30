from __future__ import annotations

import os
from typing import Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from src.corruption.pipeline import CorruptionPipeline
from src.evaluation.forecasting import (
    compute_forecast_baseline_metrics,
    compute_forecast_quality_metrics,
)
from src.models.dme_encoder import DMEEncoder
from src.training.losses import compute_forecast_loss, compute_pretraining_loss
from src.training.optim import build_pretrain_optimizer, get_linear_warmup_scheduler
from src.utils.logging import MetricsLogger


def _batch_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _batch_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_batch_to_device(v, device) for v in value]
    return value


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


def _compute_hybrid_loss(
    model: DMEEncoder,
    clean_batch: dict,
    corruption_pipeline: CorruptionPipeline,
    config: dict,
) -> dict:
    forecast_targets = clean_batch["forecast_targets"]
    corrupted_batch, denoise_targets, masks = corruption_pipeline(clean_batch)
    masks["attention_mask"] = corrupted_batch["attention_mask"]

    denoise_outputs = model(corrupted_batch, mode="pretrain")
    denoise_loss = compute_pretraining_loss(denoise_outputs, denoise_targets, masks, config)

    forecast_outputs = model(clean_batch, mode="forecast")
    forecast_loss = compute_forecast_loss(forecast_outputs, forecast_targets, config)

    f_cfg = config.get("forecasting", {})
    forecast_weight = float(f_cfg.get("alpha_forecast", f_cfg.get("alpha_denoising", 0.2)))
    total = denoise_loss["total"] + forecast_weight * forecast_loss["total"]

    return {
        "total": total,
        "denoising_total": denoise_loss["total"],
        "forecast_total": forecast_loss["total"],
        **{f"denoising_{k}": v for k, v in denoise_loss.items() if k != "total"},
        **{f"forecast_{k}": v for k, v in forecast_loss.items() if k != "total"},
    }


def evaluate_forecast_pretrain(
    model: DMEEncoder,
    val_loader: DataLoader,
    corruption_pipeline: CorruptionPipeline,
    config: dict,
    device: torch.device,
) -> dict:
    model.eval()
    metric_sums: dict[str, float] = {}
    n_batches = 0

    with torch.no_grad():
        for clean_batch in val_loader:
            clean_batch = _batch_to_device(clean_batch, device)
            loss_dict = _compute_hybrid_loss(model, clean_batch, corruption_pipeline, config)
            batch_metrics = {
                "loss_total": loss_dict["total"].item(),
                "loss_denoising": loss_dict["denoising_total"].item(),
                "loss_forecast": loss_dict["forecast_total"].item(),
                "forecast_event_type_profile_loss": loss_dict[
                    "forecast_event_type_profile"
                ].item(),
                "forecast_count_bucket_loss": loss_dict["forecast_count_bucket"].item(),
                "forecast_amount_stats_loss": loss_dict["forecast_amount_stats"].item(),
                "forecast_gap_bucket_loss": loss_dict["forecast_gap_bucket"].item(),
                "forecast_cat_profile_loss": loss_dict["forecast_cat_profiles"].item(),
            }
            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value
            n_batches += 1

    model.train()
    return {key: value / max(1, n_batches) for key, value in metric_sums.items()}


def evaluate_forecast_quality(
    model: DMEEncoder,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    forecast_stats: dict,
) -> dict:
    model.eval()
    metric_sums: dict[str, float] = {}
    n_batches = 0
    top_k = int(config.get("forecasting", {}).get("scenario_top_k", 5))

    with torch.no_grad():
        for clean_batch in val_loader:
            clean_batch = _batch_to_device(clean_batch, device)
            outputs = model(clean_batch, mode="forecast")
            targets = clean_batch["forecast_targets"]
            batch_metrics = {
                **compute_forecast_quality_metrics(
                    outputs,
                    targets,
                    forecast_stats,
                    top_k=top_k,
                ),
                **compute_forecast_baseline_metrics(
                    clean_batch,
                    targets,
                    forecast_stats,
                    top_k=top_k,
                ),
            }
            for key, value in batch_metrics.items():
                if value == value:
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            n_batches += 1

    model.train()
    return {key: value / max(1, n_batches) for key, value in metric_sums.items()}


def forecast_pretrain(
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

    optimizer, _ = build_pretrain_optimizer(model, config)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(training_cfg.get("warmup_ratio", 0.05) * total_steps)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    use_amp = mixed_precision and device.type in ("cuda", "mps")
    use_scaler = mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None

    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    model.train()

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(output_dir, "best_forecast_checkpoint.pt")
    global_step = 0

    for epoch in range(num_epochs):
        for clean_batch in train_loader:
            clean_batch = _batch_to_device(clean_batch, device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss_dict = _compute_hybrid_loss(
                    model, clean_batch, corruption_pipeline, config
                )

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

            if global_step % log_every == 0:
                logger.log_step(
                    global_step,
                    {f"train/loss_{k}": v.item() for k, v in loss_dict.items()},
                )

        val_metrics = evaluate_forecast_pretrain(
            model, val_loader, corruption_pipeline, config, device
        )
        logger.log_epoch(epoch, {f"val/{k}": v for k, v in val_metrics.items()})
        model.train()

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            _save_checkpoint(
                model,
                optimizer,
                epoch,
                best_val_loss,
                config,
                best_checkpoint_path,
                vocab_info,
            )

    return best_checkpoint_path
