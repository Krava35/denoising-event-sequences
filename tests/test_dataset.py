from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.collate import collate_fn
from src.data.dataset import EventSequenceDataset
from src.data.preprocessing import EventPreprocessor
from src.data.splits import make_entity_splits


@pytest.fixture(scope="module")
def synthetic_dataset_bundle() -> dict:
    rng = np.random.default_rng(42)
    n_entities = 200
    entity_ids = [f"e_{i:04d}" for i in range(n_entities)]

    rows: list[dict] = []
    base_time = pd.Timestamp("2023-01-01")
    for i, entity_id in enumerate(entity_ids):
        label = i % 2
        n_events = int(rng.integers(20, 51))
        t = base_time + pd.Timedelta(days=int(rng.integers(0, 30)))
        for _ in range(n_events):
            t += pd.Timedelta(minutes=int(rng.integers(1, 120)))
            rows.append(
                {
                    "entity_id": entity_id,
                    "timestamp": t,
                    "event_type": int(rng.integers(0, 5)),
                    "amount": float(rng.lognormal(mean=3.0, sigma=0.4)),
                    "num_feature": float(rng.normal(loc=0.0, scale=1.0)),
                    "cat_col": int(rng.integers(0, 4)),
                    "target": label,
                }
            )

    df = pd.DataFrame(rows)
    config = {
        "data": {
            "event_type_col": "event_type",
            "timestamp_col": "timestamp",
            "numerical_cols": ["amount", "num_feature"],
            "categorical_cols": ["cat_col"],
            "group_col": "entity_id",
            "target_col": "target",
            "max_seq_len": 50,
            "min_vocab_count": 1,
        }
    }

    splits = make_entity_splits(
        df,
        entity_col="entity_id",
        target_col="target",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        stratify=True,
    )
    train_ids = splits["train"]
    test_ids = splits["test"]

    preprocessor = EventPreprocessor(config)
    preprocessor.fit(df, train_ids)

    train_dataset = EventSequenceDataset(df, train_ids, preprocessor, config, mode="finetune")
    test_dataset = EventSequenceDataset(df, test_ids, preprocessor, config, mode="finetune")

    return {
        "config": config,
        "splits": splits,
        "preprocessor": preprocessor,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
    }


def test_clean_batch_only(synthetic_dataset_bundle: dict) -> None:
    dataset: EventSequenceDataset = synthetic_dataset_bundle["train_dataset"]
    preprocessor: EventPreprocessor = synthetic_dataset_bundle["preprocessor"]

    forbidden = {preprocessor.MASK_TYPE, preprocessor.MASK_CAT, preprocessor.MASK_EVENT}
    for i in range(len(dataset)):
        sample = dataset[i]
        tokens = set(sample["event_type"].tolist())
        assert tokens.isdisjoint(forbidden), f"Found mask token(s) in sample index {i}: {tokens}"


def test_shapes(synthetic_dataset_bundle: dict) -> None:
    dataset: EventSequenceDataset = synthetic_dataset_bundle["train_dataset"]
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    assert batch["event_type"].ndim == 2
    assert batch["time_delta"].ndim == 2
    assert batch["attention_mask"].ndim == 2
    assert batch["num_features"].ndim == 3
    assert batch["cat_features"].ndim == 3
    assert batch["label"].ndim == 1

    bsz, seq_len = batch["event_type"].shape
    assert batch["time_delta"].shape == (bsz, seq_len)
    assert batch["attention_mask"].shape == (bsz, seq_len)
    assert batch["num_features"].shape == (bsz, seq_len, 2)
    assert batch["cat_features"].shape == (bsz, seq_len, 1)
    assert batch["label"].shape == (bsz,)


def test_padding_mask(synthetic_dataset_bundle: dict) -> None:
    dataset: EventSequenceDataset = synthetic_dataset_bundle["train_dataset"]

    lengths = [dataset[i]["event_type"].shape[0] for i in range(len(dataset))]
    short_idx = int(np.argmin(lengths))
    long_idx = int(np.argmax(lengths))

    short_sample = dataset[short_idx]
    long_sample = dataset[long_idx]
    batch = collate_fn([long_sample, short_sample])

    long_len = long_sample["event_type"].shape[0]
    short_len = short_sample["event_type"].shape[0]
    assert long_len >= short_len

    assert batch["attention_mask"][0, :long_len].all()
    assert batch["attention_mask"][1, :short_len].all()
    assert not batch["attention_mask"][1, short_len:].any()

    assert (batch["event_type"][1, short_len:] == 0).all()
    assert (batch["time_delta"][1, short_len:] == 0).all()
    assert (batch["num_features"][1, short_len:] == 0).all()
    assert (batch["cat_features"][1, short_len:] == 0).all()


def test_no_nan(synthetic_dataset_bundle: dict) -> None:
    dataset: EventSequenceDataset = synthetic_dataset_bundle["train_dataset"]
    for i in range(len(dataset)):
        sample = dataset[i]
        assert not torch.isnan(sample["num_features"]).any(), f"NaN in num_features at sample {i}"
        assert not torch.isnan(sample["time_delta"]).any(), f"NaN in time_delta at sample {i}"

    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert not torch.isnan(batch["num_features"]).any()
    assert not torch.isnan(batch["time_delta"]).any()


def test_no_leakage(synthetic_dataset_bundle: dict) -> None:
    train_ids = synthetic_dataset_bundle["train_ids"]
    test_ids = synthetic_dataset_bundle["test_ids"]
    train_dataset: EventSequenceDataset = synthetic_dataset_bundle["train_dataset"]
    test_dataset: EventSequenceDataset = synthetic_dataset_bundle["test_dataset"]

    assert set(train_ids).isdisjoint(set(test_ids))
    assert len(set(train_ids) & set(test_ids)) == 0

    train_entities = {train_dataset[i]["entity_id"] for i in range(len(train_dataset))}
    test_entities = {test_dataset[i]["entity_id"] for i in range(len(test_dataset))}
    assert train_entities.isdisjoint(test_entities)


def test_collate(synthetic_dataset_bundle: dict) -> None:
    dataset: EventSequenceDataset = synthetic_dataset_bundle["train_dataset"]

    lengths = [dataset[i]["event_type"].shape[0] for i in range(len(dataset))]
    long_idx = int(np.argmax(lengths))
    long_sample = dataset[long_idx]

    short_sample = copy.deepcopy(long_sample)
    short_len = 7
    for key in ("event_type", "time_delta", "num_features", "cat_features", "attention_mask"):
        short_sample[key] = short_sample[key][:short_len]

    mid_sample = copy.deepcopy(long_sample)
    mid_len = 13
    for key in ("event_type", "time_delta", "num_features", "cat_features", "attention_mask"):
        mid_sample[key] = mid_sample[key][:mid_len]

    batch = collate_fn([long_sample, mid_sample, short_sample])
    max_len = long_sample["event_type"].shape[0]

    assert batch["event_type"].shape == (3, max_len)
    assert batch["time_delta"].shape == (3, max_len)
    assert batch["num_features"].shape == (3, max_len, 2)
    assert batch["cat_features"].shape == (3, max_len, 1)
    assert batch["attention_mask"].shape == (3, max_len)
    assert batch["label"].shape == (3,)
    assert len(batch["entity_id"]) == 3

    assert batch["attention_mask"][1, :mid_len].all()
    assert not batch["attention_mask"][1, mid_len:].any()
    assert batch["attention_mask"][2, :short_len].all()
    assert not batch["attention_mask"][2, short_len:].any()

    assert (batch["event_type"][1, mid_len:] == 0).all()
    assert (batch["time_delta"][1, mid_len:] == 0).all()
    assert (batch["num_features"][1, mid_len:] == 0).all()
    assert (batch["cat_features"][1, mid_len:] == 0).all()

    assert (batch["event_type"][2, short_len:] == 0).all()
    assert (batch["time_delta"][2, short_len:] == 0).all()
    assert (batch["num_features"][2, short_len:] == 0).all()
    assert (batch["cat_features"][2, short_len:] == 0).all()
