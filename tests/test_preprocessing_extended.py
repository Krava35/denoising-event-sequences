"""Тесты для EventPreprocessor, EventSequenceDataset и make_entity_splits."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import EventPreprocessor
from src.data.splits import make_entity_splits

# ── Helpers ───────────────────────────────────────────────────────────────────

_CONFIG = {
    "data": {
        "event_type_col": "event_type",
        "timestamp_col": "timestamp",
        "numerical_cols": ["amount"],
        "categorical_cols": [],
        "group_col": "entity_id",
        "min_vocab_count": 1,
        "amount_cols": ["amount"],
        "robust_scale_cols": ["amount"],
    }
}


def _make_df(n_entities: int = 10, events_per_entity: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for eid in range(n_entities):
        for i in range(events_per_entity):
            rows.append(
                {
                    "entity_id": eid,
                    "event_type": rng.choice(["A", "B", "C"]),
                    "timestamp": float(i),
                    "amount": float(rng.normal(100, 20)),
                    "target": eid % 2,
                }
            )
    return pd.DataFrame(rows)


def _fitted_preprocessor(df: pd.DataFrame) -> EventPreprocessor:
    pp = EventPreprocessor(_CONFIG)
    train_ids = df["entity_id"].unique().tolist()
    pp.fit(df, train_ids)
    return pp


# ── Preprocessor tests ────────────────────────────────────────────────────────


def test_preprocessor_oov_maps_to_unk() -> None:
    """OOV event type должен маппироваться в UNK (индекс 1)."""
    df = _make_df()
    pp = _fitted_preprocessor(df)
    oov_df = df.copy()
    oov_df["event_type"] = "NEVER_SEEN_TOKEN"
    result = pp.transform(oov_df)
    assert (result["event_type"] == EventPreprocessor.UNK).all()


def test_preprocessor_save_load_roundtrip(tmp_path) -> None:
    """Сохранение и загрузка сохраняют vocab и scaler_params без потерь."""
    df = _make_df()
    pp = _fitted_preprocessor(df)
    path = tmp_path / "preprocessor.json"
    pp.save(path)

    pp2 = EventPreprocessor(_CONFIG)
    pp2.load(path)

    assert pp2.vocab == pp.vocab
    assert pp2.scaler_params == pp.scaler_params
    assert pp2._fitted


def test_preprocessor_nan_imputation() -> None:
    """NaN в числовой колонке заменяется центром, итого scaled≈0."""
    df = _make_df()
    pp = _fitted_preprocessor(df)

    nan_df = df.copy()
    nan_df.loc[nan_df.index[:5], "amount"] = float("nan")
    result = pp.transform(nan_df)
    # Позиции с NaN после imputation → center → scaled=0
    assert result["amount"].iloc[:5].abs().max() < 1e-6
    assert result.notna().all().all()


def test_preprocessor_no_nan_after_transform() -> None:
    """transform() не производит NaN ни в одной колонке."""
    df = _make_df()
    pp = _fitted_preprocessor(df)
    result = pp.transform(df)
    nan_cols = [c for c in result.columns if result[c].isna().any()]
    assert not nan_cols, f"NaN в колонках после transform: {nan_cols}"


def test_preprocessor_robust_scale_cols_independent() -> None:
    """robust_scale_cols может отличаться от amount_cols."""
    cfg = {
        "data": {
            "event_type_col": "event_type",
            "timestamp_col": "timestamp",
            "numerical_cols": ["amount", "other"],
            "categorical_cols": [],
            "group_col": "entity_id",
            "min_vocab_count": 1,
            "amount_cols": ["amount"],       # только amount — amount transform
            "robust_scale_cols": ["other"],  # только other — robust scaler
        }
    }
    rng = np.random.default_rng(1)
    rows = [
        {
            "entity_id": i // 5,
            "event_type": "A",
            "timestamp": float(i % 5),
            "amount": float(rng.normal(0, 1)),
            "other": float(rng.normal(0, 1)),
            "target": 0,
        }
        for i in range(50)
    ]
    df = pd.DataFrame(rows)
    pp = EventPreprocessor(cfg)
    pp.fit(df, df["entity_id"].unique().tolist())
    # "other" is in robust_scale_cols → robust scaler
    assert pp.scaler_params["other"]["type"] == "robust"
    # "amount" is in amount_cols but NOT in robust_scale_cols → standard scaler
    assert pp.scaler_params["amount"]["type"] == "standard"


# ── Dataset tests ─────────────────────────────────────────────────────────────


def test_dataset_invalid_mode_raises() -> None:
    """Неверный mode → ValueError при создании датасета."""
    from src.data.dataset import EventSequenceDataset

    df = _make_df()
    pp = _fitted_preprocessor(df)
    entity_ids = df["entity_id"].unique().tolist()

    with pytest.raises(ValueError, match="mode must be pretrain|finetune|eval"):
        EventSequenceDataset(df, entity_ids, pp, _CONFIG, mode="invalid_mode")


def test_pretrain_window_within_max_seq_len() -> None:
    """Sliding window никогда не превышает max_seq_len."""
    from src.data.dataset import EventSequenceDataset

    cfg = dict(_CONFIG)
    cfg = {**_CONFIG, "data": {**_CONFIG["data"], "max_seq_len": 10}}
    df = _make_df(n_entities=5, events_per_entity=30)  # 30 > 10
    pp = EventPreprocessor(cfg)
    pp.fit(df, df["entity_id"].unique().tolist())
    entity_ids = df["entity_id"].unique().tolist()

    ds = EventSequenceDataset(df, entity_ids, pp, cfg, mode="pretrain")
    for _ in range(20):
        sample = ds[0]
        L = sample["event_type"].shape[0]
        assert L <= 10, f"Длина {L} превышает max_seq_len=10"


# ── Splits tests ──────────────────────────────────────────────────────────────


def test_splits_ratio_sum_validation() -> None:
    """train+val+test != 1.0 → ValueError."""
    df = _make_df()
    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        make_entity_splits(
            df,
            entity_col="entity_id",
            target_col="target",
            train_ratio=0.5,
            val_ratio=0.3,
            test_ratio=0.3,
        )


def test_splits_no_overlap() -> None:
    """train/val/test не пересекаются и покрывают все сущности."""
    df = _make_df(n_entities=30)
    splits = make_entity_splits(df, entity_col="entity_id", target_col="target")
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])
    all_entities = set(df["entity_id"].unique())

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == all_entities
