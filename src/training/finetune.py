from __future__ import annotations

import itertools
import os
from typing import Optional

import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
)

from src.models.dme_encoder import DMEEncoder
from src.training.optim import build_finetune_optimizer, get_linear_warmup_scheduler
from src.utils.logging import MetricsLogger


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _load_encoder_weights(model: DMEEncoder, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict: dict = ckpt["model_state_dict"]
    encoder_state = {
        k: v for k, v in state_dict.items()
        if k.startswith(("tokenizer.", "encoder.", "pooling."))
    }
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    # missing = classifier + denoising heads (expected); unexpected should be empty
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading encoder weights: {unexpected}")


def _save_finetune_checkpoint(
    model: DMEEncoder,
    optimizer,
    epoch: int,
    val_metrics: dict,
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
            "val_roc_auc": val_metrics.get("roc_auc", 0.0),
            "val_metrics": val_metrics,
            "config": config,
            "vocab_info": vocab_info,
        },
        path,
    )


def evaluate_finetune(
    model: DMEEncoder,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict:
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = _batch_to_device(batch, device)
            outputs = model(batch, mode="finetune")
            logits = outputs["logits"]  # [B, num_classes]
            probs = torch.softmax(logits, dim=-1)  # [B, num_classes]
            preds = logits.argmax(dim=-1)           # [B]
            labels = batch["label"].long()

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    model.train()

    probs_t = torch.cat(all_probs)      # [N, num_classes]
    preds_t = torch.cat(all_preds)      # [N]
    targets_t = torch.cat(all_targets)  # [N]

    targets_np = targets_t.numpy()
    preds_np = preds_t.numpy()

    balanced_acc = float(balanced_accuracy_score(targets_np, preds_np))

    if num_classes == 2:
        pos_probs = probs_t[:, 1]  # [N] probability of positive class
        metrics = {
            "accuracy": BinaryAccuracy()(preds_t, targets_t).item(),
            "macro_f1": MulticlassF1Score(num_classes=2, average="macro")(
                preds_t, targets_t
            ).item(),
            "weighted_f1": MulticlassF1Score(num_classes=2, average="weighted")(
                preds_t, targets_t
            ).item(),
            "roc_auc": BinaryAUROC()(pos_probs, targets_t).item(),
            "pr_auc": BinaryAveragePrecision()(pos_probs, targets_t).item(),
            "balanced_accuracy": balanced_acc,
        }
    else:
        metrics = {
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro")(
                preds_t, targets_t
            ).item(),
            "macro_f1": MulticlassF1Score(num_classes=num_classes, average="macro")(
                preds_t, targets_t
            ).item(),
            "weighted_f1": MulticlassF1Score(num_classes=num_classes, average="weighted")(
                preds_t, targets_t
            ).item(),
            "roc_auc": MulticlassAUROC(num_classes=num_classes, average="macro")(
                probs_t, targets_t
            ).item(),
            "macro_pr_auc": MulticlassAveragePrecision(num_classes=num_classes, average="macro")(
                probs_t, targets_t
            ).item(),
            "balanced_accuracy": balanced_acc,
        }

    return metrics


def finetune(
    model: DMEEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    output_dir: str,
    device: torch.device,
    logger: MetricsLogger,
    pretrained_checkpoint: Optional[str] = None,
    frozen_encoder: bool = False,
    label_fraction: float = 1.0,
    vocab_info: Optional[dict] = None,
) -> str:
    training_cfg = config.get("training", {})
    num_epochs: int = training_cfg.get("num_epochs_finetune", 20)
    grad_clip: float = training_cfg.get("gradient_clip_val", 1.0)
    log_every: int = training_cfg.get("log_every_n_steps", 50)
    patience: int = training_cfg.get("early_stopping_patience", 5)
    mixed_precision: bool = training_cfg.get("mixed_precision", False)

    # Determine num_classes from classifier head output dimension
    num_classes: int = model.classifier.classifier[-1].out_features

    if pretrained_checkpoint:
        _load_encoder_weights(model, pretrained_checkpoint)

    optimizer, _ = build_finetune_optimizer(model, config, frozen_encoder=frozen_encoder)

    # Compute total steps respecting label_fraction
    n_batches_per_epoch = max(1, round(len(train_loader) * label_fraction))
    total_steps = num_epochs * n_batches_per_epoch
    warmup_steps = int(training_cfg.get("warmup_ratio", 0.05) * total_steps)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    use_amp = mixed_precision and device.type in ("cuda", "mps")
    use_scaler = mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None

    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    model.train()

    best_roc_auc = -float("inf")
    patience_counter = 0
    best_checkpoint_path = os.path.join(output_dir, "best_finetune_checkpoint.pt")
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        for batch in itertools.islice(train_loader, n_batches_per_epoch):
            batch = _batch_to_device(batch, device)
            labels = batch["label"].long()

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch, mode="finetune")
                loss = F.cross_entropy(outputs["logits"], labels)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_every == 0:
                logger.log_step(global_step, {"train/loss": loss.item()})

        # Epoch validation
        val_metrics = evaluate_finetune(model, val_loader, num_classes, device)
        logger.log_epoch(epoch, {f"val/{k}": v for k, v in val_metrics.items()})

        # Early stopping on ROC-AUC (maximize)
        roc_auc = val_metrics["roc_auc"]
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            patience_counter = 0
            _save_finetune_checkpoint(
                model, optimizer, epoch, val_metrics, config,
                best_checkpoint_path, vocab_info,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_checkpoint_path
