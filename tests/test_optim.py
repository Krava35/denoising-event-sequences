from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.data.splits import load_splits, save_splits
from src.models.dme_encoder import DMEEncoder
from src.training.optim import (
    build_finetune_optimizer,
    build_pretrain_optimizer,
    get_linear_warmup_scheduler,
)

_CONFIG = {
    "model": {
        "hidden_dim": 32,
        "num_layers": 1,
        "num_heads": 4,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "activation": "gelu",
        "event_type_emb_dim": 16,
        "cat_emb_dim": 8,
        "num_projection_dim": 16,
        "time_projection_dim": 16,
        "max_seq_len": 32,
    },
    "pooling": {"type": "mean"},
    "training": {
        "lr": 1e-3,
        "lr_encoder": 1e-4,
        "weight_decay": 0.01,
        "num_epochs_pretrain": 5,
        "num_epochs_finetune": 3,
    },
}

_VOCAB_INFO = {
    "event_type_vocab_size": 10,
    "cat_vocab_sizes": [6],
    "num_num_features": 2,
    "num_classes": 2,
}


def _make_model() -> DMEEncoder:
    torch.manual_seed(0)
    return DMEEncoder(_CONFIG, _VOCAB_INFO)


def test_build_pretrain_optimizer_types() -> None:
    model = _make_model()
    opt, sched = build_pretrain_optimizer(model, _CONFIG)
    assert isinstance(opt, AdamW)
    assert isinstance(sched, CosineAnnealingLR)


def test_build_pretrain_optimizer_lr() -> None:
    model = _make_model()
    opt, _ = build_pretrain_optimizer(model, _CONFIG)
    assert opt.defaults["lr"] == pytest.approx(1e-3)
    assert opt.defaults["weight_decay"] == pytest.approx(0.01)


def test_build_finetune_optimizer_two_param_groups() -> None:
    model = _make_model()
    opt, sched = build_finetune_optimizer(model, _CONFIG, frozen_encoder=False)
    assert isinstance(opt, AdamW)
    assert isinstance(sched, CosineAnnealingLR)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert opt.param_groups[1]["lr"] == pytest.approx(1e-3)


def test_build_finetune_optimizer_frozen_encoder() -> None:
    model = _make_model()
    build_finetune_optimizer(model, _CONFIG, frozen_encoder=True)
    for p in model.classifier.parameters():
        assert p.requires_grad
    for p in list(model.tokenizer.parameters()) + list(model.encoder.parameters()):
        assert not p.requires_grad


def test_get_linear_warmup_scheduler_warmup_phase() -> None:
    model = _make_model()
    opt, _ = build_pretrain_optimizer(model, _CONFIG)
    sched = get_linear_warmup_scheduler(opt, warmup_steps=10, total_steps=100)
    assert isinstance(sched, LambdaLR)
    lr_at_0 = sched.get_last_lr()[0]
    assert lr_at_0 == pytest.approx(0.0)


def test_get_linear_warmup_scheduler_cosine_decay() -> None:
    model = _make_model()
    opt, _ = build_pretrain_optimizer(model, _CONFIG)
    sched = get_linear_warmup_scheduler(opt, warmup_steps=5, total_steps=20)
    for _ in range(5):
        opt.step()
        sched.step()
    lr = sched.get_last_lr()[0]
    assert lr == pytest.approx(1e-3, rel=0.01)


def test_save_and_load_splits(tmp_path: Path) -> None:
    splits = {"train": ["a", "b", "c"], "val": ["d"], "test": ["e", "f"]}
    path = tmp_path / "splits.json"
    save_splits(splits, path)
    assert path.exists()
    loaded = load_splits(path)
    assert loaded.keys() == splits.keys()
    for k in splits:
        assert set(loaded[k]) == set(splits[k])
