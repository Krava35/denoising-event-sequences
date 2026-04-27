"""End-to-end smoke tests: data pipeline → corruption pipeline.

These tests verify the full path from raw DataFrame to a corrupted batch
that is ready for the pretraining loop, using small synthetic data so they
run quickly on CPU without any model weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.corruption.pipeline import CorruptionPipeline
from src.data.collate import collate_fn
from src.data.dataset import EventSequenceDataset
from src.data.preprocessing import EventPreprocessor
from src.data.splits import make_entity_splits
from src.models.dme_encoder import DMEEncoder
from src.training.losses import compute_pretraining_loss

# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_synthetic_df(n_entities: int = 30, events_per_entity: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    entity_ids = [f"e_{i:04d}" for i in range(n_entities)]
    targets = rng.integers(0, 2, size=n_entities)
    rows = []
    base_time = pd.Timestamp("2023-01-01")
    for i, eid in enumerate(entity_ids):
        t = base_time
        for _ in range(events_per_entity):
            t += pd.Timedelta(hours=int(rng.integers(1, 48)))
            rows.append({
                "entity_id": eid,
                "timestamp": t,
                "event_type": int(rng.integers(0, 10)),
                "amount": float(rng.exponential(100)),
                "cat_col": int(rng.integers(0, 5)),
                "target": int(targets[i]),
            })
    return pd.DataFrame(rows), entity_ids, targets.tolist()


_PIPELINE_CONFIG = {
    "data": {
        "event_type_col": "event_type",
        "timestamp_col": "timestamp",
        "numerical_cols": ["amount"],
        "categorical_cols": ["cat_col"],
        "group_col": "entity_id",
        "target_col": "target",
        "max_seq_len": 32,
        "min_seq_len": 2,
        "min_vocab_count": 1,
        "truncation_pretrain": "sliding_window",
        "truncation_eval": "last_events",
        "amount_transform": "robust_scaler",
        "time_transform": "log1p",
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    },
    "model": {
        "event_type_emb_dim": 16,
        "cat_emb_dim": 8,
        "num_projection_dim": 16,
        "time_projection_dim": 16,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "activation": "gelu",
        "max_seq_len": 32,
    },
    "pooling": {"type": "mean"},
    "loss": {
        "lambda_event_type": 1.0,
        "lambda_time": 1.0,
        "lambda_num": 0.5,
        "lambda_cat": 0.5,
        "lambda_exist": 0.1,
    },
    "training": {"batch_size": 16},
}

_CONFIG = {
    "data": {
        "event_type_col": "event_type",
        "timestamp_col": "timestamp",
        "numerical_cols": ["amount"],
        "categorical_cols": ["cat_col"],
        "group_col": "entity_id",
        "target_col": "target",
        "max_seq_len": 16,
        "min_vocab_count": 1,
        "truncation_pretrain": "sliding_window",
        "truncation_eval": "last_events",
    }
}

_CORRUPTION_CFG = {
    "event_level_masking": {"prob": 0.10},
    "event_type": {
        "selected_prob": 0.40,
        "mask_prob": 0.28,
        "transition_replace_prob": 0.00,
        "random_replace_prob": 0.10,
        "keep_predict_prob": 0.02,
        "use_transition_aware_replacement": False,
    },
    "categorical_features": {"mask_prob": 0.15, "random_replace_prob": 0.05},
    "time_noise": {
        "corruption_prob": 0.30,
        "min_std": 0.05,
        "max_std": 0.30,
        "sampling_level": "batch",
    },
    "numerical_noise": {
        "corruption_prob": 0.20,
        "min_std": 0.03,
        "max_std": 0.15,
        "sampling_level": "batch",
    },
}


# ── Smoke tests ───────────────────────────────────────────────────────────────

@pytest.mark.smoke
def test_preprocessing_smoke():
    """Smoke: EventPreprocessor fit/transform produces no NaN and correct vocab."""
    df, entity_ids, _ = _make_synthetic_df()
    train_ids = entity_ids[:20]

    prep = EventPreprocessor(_CONFIG)
    prep.fit(df, train_ids)

    transformed = prep.transform(df[df["entity_id"].isin(set(entity_ids))].copy())

    for col in ["event_type", "amount", "cat_col", "time_delta"]:
        assert not transformed[col].isna().any(), f"NaN in transformed column '{col}'"

    ev_vocab = prep.vocab[prep.event_type_col]
    assert "<PAD>" in ev_vocab and ev_vocab["<PAD>"] == 0
    assert "<MASK_TYPE>" in ev_vocab and ev_vocab["<MASK_TYPE>"] == 2


@pytest.mark.smoke
def test_dataset_and_dataloader_smoke():
    """Smoke: EventSequenceDataset + DataLoader yields correctly shaped batches."""
    df, entity_ids, _ = _make_synthetic_df()
    train_ids = entity_ids[:20]

    prep = EventPreprocessor(_CONFIG)
    prep.fit(df, train_ids)

    dataset = EventSequenceDataset(df, train_ids, prep, _CONFIG, mode="pretrain")
    assert len(dataset) == len(train_ids)

    sample = dataset[0]
    assert sample["event_type"].ndim == 1
    assert sample["attention_mask"].dtype == torch.bool
    assert sample["attention_mask"].all(), "single sample has no padding"
    assert not torch.isnan(sample["time_delta"]).any()
    assert not torch.isnan(sample["num_features"]).any()

    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch["event_type"].ndim == 2
    assert batch["num_features"].ndim == 3
    assert batch["attention_mask"].dtype == torch.bool
    assert not torch.isnan(batch["time_delta"]).any()
    assert not torch.isnan(batch["num_features"]).any()


@pytest.mark.smoke
def test_corruption_pipeline_smoke():
    """Smoke: CorruptionPipeline on a DataLoader batch returns valid structure."""
    df, entity_ids, _ = _make_synthetic_df()
    train_ids = entity_ids[:20]

    prep = EventPreprocessor(_CONFIG)
    prep.fit(df, train_ids)

    dataset = EventSequenceDataset(df, train_ids, prep, _CONFIG, mode="pretrain")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    vocab_sizes = {
        "event_type": len(prep.vocab[prep.event_type_col]),
        "cat_features": [len(prep.vocab[c]) for c in prep.categorical_cols],
    }
    pipe = CorruptionPipeline(_CORRUPTION_CFG, vocab_sizes=vocab_sizes)
    corrupted, targets, masks = pipe(batch)

    assert set(corrupted.keys()) == set(batch.keys())
    for key in ("event_type", "time_delta", "num_features", "cat_features"):
        assert corrupted[key].shape == batch[key].shape, f"shape mismatch for '{key}'"
        assert targets[key].shape == batch[key].shape, f"target shape mismatch for '{key}'"

    assert "event_level" in masks
    assert "event_type" in masks
    assert not torch.isnan(corrupted["time_delta"]).any()
    assert not torch.isnan(corrupted["num_features"]).any()

    # Padding must be intact after corruption
    pad = ~batch["attention_mask"]
    assert (corrupted["event_type"][pad] == 0).all(), "padding event_type must remain 0"


@pytest.mark.smoke
def test_splits_smoke():
    """Smoke: make_entity_splits yields disjoint, complete, stratified splits."""
    df, entity_ids, _ = _make_synthetic_df()

    splits = make_entity_splits(
        df,
        entity_col="entity_id",
        target_col="target",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_s, val_s, test_s = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train_s.isdisjoint(val_s), "train/val overlap"
    assert train_s.isdisjoint(test_s), "train/test overlap"
    assert val_s.isdisjoint(test_s), "val/test overlap"
    assert len(train_s) + len(val_s) + len(test_s) == len(entity_ids)


@pytest.mark.smoke
def test_full_pipeline():
    """End-to-end: preprocessing → dataset → collate → corruption → DMEEncoder → loss → backward."""
    torch.manual_seed(42)

    # 1. Синтетические данные: 100 entities, 20 events, 3 event types
    df, entity_ids, _ = _make_synthetic_df(n_entities=100, events_per_entity=20, seed=0)
    df["event_type"] = df["event_type"] % 3

    # 2. Splits + препроцессинг
    splits = make_entity_splits(
        df,
        entity_col="entity_id",
        target_col="target",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    prep = EventPreprocessor(_PIPELINE_CONFIG)
    prep.fit(df, splits["train"])

    # 3. Dataset + DataLoader
    dataset = EventSequenceDataset(df, splits["train"], prep, _PIPELINE_CONFIG, mode="pretrain")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)
    clean_batch = next(iter(loader))

    # 4. Corruption pipeline
    vocab_sizes = {
        "event_type": len(prep.vocab[prep.event_type_col]),
        "cat_features": [len(prep.vocab[c]) for c in prep.categorical_cols],
    }
    pipe = CorruptionPipeline(_CORRUPTION_CFG, vocab_sizes=vocab_sizes)
    corrupted_batch, targets, masks = pipe(clean_batch)
    masks["attention_mask"] = corrupted_batch["attention_mask"]

    B, L = corrupted_batch["event_type"].shape

    # 5. Модель
    event_type_vocab_size = len(prep.vocab[prep.event_type_col])
    cat_vocab_sizes = [len(prep.vocab[c]) for c in prep.categorical_cols]
    vocab_info = {
        "event_type_vocab_size": event_type_vocab_size,
        "cat_vocab_sizes": cat_vocab_sizes,
        "num_num_features": len(prep.numerical_cols),
        "num_classes": 2,
    }
    model = DMEEncoder(_PIPELINE_CONFIG, vocab_info)
    model.train()

    # 6. Forward
    outputs = model(corrupted_batch, mode="pretrain")

    # 7. Loss
    loss_dict = compute_pretraining_loss(outputs, targets, masks, _PIPELINE_CONFIG)

    # 8. Backward
    loss_dict["total"].backward()

    # 9. Assertions
    H = _PIPELINE_CONFIG["model"]["hidden_dim"]

    # Loss финитность и положительность
    assert torch.isfinite(loss_dict["total"]), "total loss is NaN/Inf"
    for component in ("event_type", "time_delta", "numerical", "categorical", "existence"):
        assert torch.isfinite(loss_dict[component]), f"loss['{component}'] is NaN/Inf"
    assert loss_dict["total"].item() > 0.0, "total loss должен быть > 0"

    # Shapes выходов модели
    assert outputs["event_type_logits"].shape == (B, L, event_type_vocab_size)
    assert outputs["time_delta_pred"].shape == (B, L, 1)
    assert outputs["existence_logits"].shape == (B, L, 1)
    assert outputs["hidden_states"].shape == (B, L, H)
    assert "num_pred" in outputs
    assert outputs["num_pred"].shape == (B, L, 1)
    assert "cat_logits" in outputs and len(outputs["cat_logits"]) == 1
    assert outputs["cat_logits"][0].shape == (B, L, cat_vocab_sizes[0])

    # В pretrain-режиме pooling и classifier не задействованы — их градиенты None это норма.
    # Проверяем, что хотя бы часть параметров получила градиенты (encoder + реконструкционные головы).
    grads_received = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads_received) > 0, "Ни один параметр не получил градиент"
    for p in grads_received:
        assert p.grad.isfinite().all(), f"NaN/Inf в градиенте параметра shape={p.shape}"
