from __future__ import annotations

import math

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.models.dme_encoder import DMEEncoder


def build_pretrain_optimizer(
    model: DMEEncoder,
    config: dict,
) -> tuple[AdamW, CosineAnnealingLR]:
    training_cfg = config.get("training", {})
    lr: float = training_cfg.get("lr", 3e-4)
    weight_decay: float = training_cfg.get("weight_decay", 0.01)
    num_epochs: int = training_cfg.get("num_epochs_pretrain", 30)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    return optimizer, scheduler


def build_finetune_optimizer(
    model: DMEEncoder,
    config: dict,
    frozen_encoder: bool = False,
) -> tuple[AdamW, CosineAnnealingLR]:
    training_cfg = config.get("training", {})
    lr: float = training_cfg.get("lr", 3e-4)
    lr_encoder: float = training_cfg.get("lr_encoder", lr * 0.1)
    weight_decay: float = training_cfg.get("weight_decay", 0.01)
    num_epochs: int = training_cfg.get("num_epochs_finetune", 20)

    if frozen_encoder:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True

        optimizer = AdamW(model.get_head_params(), lr=lr, weight_decay=weight_decay)
    else:
        param_groups = [
            {"params": model.get_encoder_params(), "lr": lr_encoder},
            {"params": model.get_head_params(), "lr": lr},
        ]
        optimizer = AdamW(param_groups, weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    return optimizer, scheduler


def get_linear_warmup_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, _lr_lambda)
