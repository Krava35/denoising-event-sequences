import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Special token indices — fixed, never reassigned
PAD = 0
UNK = 1
MASK_TYPE = 2
MASK_CAT = 3
MASK_EVENT = 4

_SPECIAL_TOKENS: dict[str, int] = {
    "<PAD>": PAD,
    "<UNK>": UNK,
    "<MASK_TYPE>": MASK_TYPE,
    "<MASK_CAT>": MASK_CAT,
    "<MASK_EVENT>": MASK_EVENT,
}

_ROBUST_KEYWORDS = frozenset({"amount", "amnt", "sum", "total"})


def _is_amount_col(col: str) -> bool:
    lower = col.lower()
    return any(kw in lower for kw in _ROBUST_KEYWORDS)


def _to_seconds(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.values.astype(float)
    return pd.to_datetime(series).astype(np.int64) / 1_000_000_000.0


class EventPreprocessor:
    PAD = PAD
    UNK = UNK
    MASK_TYPE = MASK_TYPE
    MASK_CAT = MASK_CAT
    MASK_EVENT = MASK_EVENT

    def __init__(self, config: dict) -> None:
        data_cfg = config.get("data", {})
        self.event_type_col: str = data_cfg.get("event_type_col", "event_type")
        self.timestamp_col: str = data_cfg.get("timestamp_col", "timestamp")
        self.numerical_cols: list[str] = list(data_cfg.get("numerical_cols") or [])
        self.categorical_cols: list[str] = list(data_cfg.get("categorical_cols") or [])
        self.entity_col: str = data_cfg.get("group_col", "entity_id")
        self._amount_cols: set[str] = set(data_cfg.get("amount_cols") or [])
        self.min_count: int = int(data_cfg.get("min_vocab_count", 5))

        self.vocab: dict[str, dict[str, int]] = {}
        self.scaler_params: dict[str, dict] = {}
        self._fitted: bool = False

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, train_entity_ids: list) -> None:
        train_df = df[df[self.entity_col].isin(set(train_entity_ids))]

        self._build_vocab(train_df, self.event_type_col)
        for col in self.categorical_cols:
            self._build_vocab(train_df, col)

        time_delta = self.compute_time_delta(train_df, self.entity_col, self.timestamp_col)
        self._fit_scaler("time_delta", np.log1p(time_delta.values.astype(float)), robust=True)

        for col in self.numerical_cols:
            robust = col in self._amount_cols or _is_amount_col(col)
            self._fit_scaler(col, train_df[col].values.astype(float), robust=robust)

        self._fitted = True
        logger.info(
            "Fitted on %d train entities | event_type vocab=%d | categorical=%s",
            len(train_entity_ids),
            len(self.vocab.get(self.event_type_col, {})),
            {c: len(self.vocab.get(c, {})) for c in self.categorical_cols},
        )

    def _build_vocab(self, df: pd.DataFrame, col: str) -> None:
        counts = df[col].astype(str).value_counts()
        vocab: dict[str, int] = dict(_SPECIAL_TOKENS)
        for token in counts[counts >= self.min_count].index:
            if token not in vocab:
                vocab[token] = len(vocab)
        self.vocab[col] = vocab

    def _fit_scaler(self, col: str, values: np.ndarray, robust: bool) -> None:
        clean = values[np.isfinite(values)]
        if len(clean) == 0:
            self.scaler_params[col] = {
                "type": "robust" if robust else "standard",
                "center": 0.0,
                "scale": 1.0,
            }
            return

        if robust:
            center = float(np.median(clean))
            q75, q25 = np.percentile(clean, [75, 25])
            scale = float(q75 - q25) or 1.0
        else:
            center = float(np.mean(clean))
            scale = float(np.std(clean)) or 1.0

        self.scaler_params[col] = {
            "type": "robust" if robust else "standard",
            "center": center,
            "scale": scale,
        }

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        result = df.copy()

        ev_vocab = self.vocab[self.event_type_col]
        result[self.event_type_col] = (
            result[self.event_type_col].astype(str).map(lambda x: ev_vocab.get(x, UNK))
        )

        for col in self.categorical_cols:
            col_vocab = self.vocab[col]
            result[col] = result[col].astype(str).map(lambda x, v=col_vocab: v.get(x, UNK))

        time_delta = self.compute_time_delta(df, self.entity_col, self.timestamp_col)
        result["time_delta"] = self._scale(
            np.log1p(time_delta.values.astype(float)), "time_delta"
        )

        for col in self.numerical_cols:
            values = result[col].values.astype(float)
            center = self.scaler_params[col]["center"]
            values = np.where(np.isfinite(values), values, center)
            result[col] = self._scale(values, col)

        nan_cols = [c for c in result.columns if result[c].isna().any()]
        assert not nan_cols, f"NaN after transform in columns: {nan_cols}"

        return result

    def _scale(self, values: np.ndarray, col: str) -> np.ndarray:
        p = self.scaler_params[col]
        scale = p["scale"] if p["scale"] != 0.0 else 1.0
        return (values - p["center"]) / scale

    # ── Time delta ────────────────────────────────────────────────────────────

    def compute_time_delta(
        self, df: pd.DataFrame, entity_col: str, timestamp_col: str
    ) -> pd.Series:
        working = pd.DataFrame(
            {
                "_entity": df[entity_col].values,
                "_ts": _to_seconds(df[timestamp_col]),
                "_pos": np.arange(len(df)),
            }
        )
        working = working.sort_values(["_entity", "_ts"], kind="stable")
        working["_delta"] = (
            working.groupby("_entity", sort=False)["_ts"]
            .diff()
            .fillna(0.0)
            .clip(lower=0.0)
        )
        out = np.empty(len(df), dtype=float)
        out[working["_pos"].values] = working["_delta"].values
        return pd.Series(out, index=df.index, dtype=float)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "vocab": self.vocab,
            "scaler_params": self.scaler_params,
            "meta": {
                "event_type_col": self.event_type_col,
                "timestamp_col": self.timestamp_col,
                "numerical_cols": self.numerical_cols,
                "categorical_cols": self.categorical_cols,
                "entity_col": self.entity_col,
                "min_count": self.min_count,
            },
        }
        with path.open("w") as f:
            json.dump(artifact, f, indent=2)
        logger.info("Saved preprocessor to %s", path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with path.open() as f:
            artifact = json.load(f)
        self.vocab = artifact["vocab"]
        self.scaler_params = artifact["scaler_params"]
        self._fitted = True
        logger.info("Loaded preprocessor from %s", path)


# ── __main__ tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)
    n_entities = 200
    events_per_entity = 5
    n_event_types = 15

    entity_ids = [f"e_{i:04d}" for i in range(n_entities)]
    targets = rng.integers(0, 2, size=n_entities)

    rows = []
    base_time = pd.Timestamp("2023-01-01")
    for i, eid in enumerate(entity_ids):
        t = base_time + pd.Timedelta(days=int(rng.integers(0, 30)))
        for _ in range(events_per_entity):
            t += pd.Timedelta(hours=int(rng.integers(1, 48)))
            rows.append(
                {
                    "entity_id": eid,
                    "timestamp": t,
                    # event type 99 added as rare: appears only in first 2 entities
                    "event_type": 99 if (i < 2) else int(rng.integers(0, n_event_types)),
                    "amount": float(rng.exponential(100)),
                    "num_feature": float(rng.normal(0, 1)),
                    "cat_col": int(rng.integers(0, 8)),
                    "target": int(targets[i]),
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
            "min_vocab_count": 5,
        }
    }

    train_ids = entity_ids[:140]
    val_ids = entity_ids[140:170]
    test_ids = entity_ids[170:]

    prep = EventPreprocessor(config)
    prep.fit(df, train_ids)

    # Special token indices must be fixed
    for col_vocab in prep.vocab.values():
        assert col_vocab["<PAD>"] == PAD, "PAD index wrong"
        assert col_vocab["<UNK>"] == UNK, "UNK index wrong"
        assert col_vocab["<MASK_TYPE>"] == MASK_TYPE, "MASK_TYPE index wrong"
        assert col_vocab["<MASK_CAT>"] == MASK_CAT, "MASK_CAT index wrong"
        assert col_vocab["<MASK_EVENT>"] == MASK_EVENT, "MASK_EVENT index wrong"

    # Rare type 99 (only in entities 0-1, both in train, but count < 5 in 140-entity train)
    # entities 0-1 contribute 10 rows with type 99; min_count=5, so it depends on count.
    # With 2 entities × 5 events = 10 occurrences in full data, 10 in train (entities 0-1 are in train).
    # count=10 >= 5, so type 99 IS in vocab. Let's verify it's there.
    ev_vocab = prep.vocab[prep.event_type_col]
    assert "99" in ev_vocab, "type 99 with 10 occurrences should be in vocab"

    # Scaler types
    assert prep.scaler_params["time_delta"]["type"] == "robust"
    assert prep.scaler_params["amount"]["type"] == "robust"
    assert prep.scaler_params["num_feature"]["type"] == "standard"

    # Transform all splits
    def get_split_df(ids: list) -> pd.DataFrame:
        return df[df["entity_id"].isin(set(ids))].copy()

    train_t = prep.transform(get_split_df(train_ids))
    val_t = prep.transform(get_split_df(val_ids))
    test_t = prep.transform(get_split_df(test_ids))

    # No NaN in transformed cols
    for name, tdf in [("train", train_t), ("val", val_t), ("test", test_t)]:
        for col in ["event_type", "amount", "num_feature", "cat_col", "time_delta"]:
            assert not tdf[col].isna().any(), f"NaN in {name}/{col}"

    # OOV event type → UNK
    oov_df = get_split_df(test_ids).copy()
    oov_df["event_type"] = 9999
    oov_t = prep.transform(oov_df)
    assert (oov_t["event_type"] == UNK).all(), "OOV must map to UNK"

    # Vocab built only from train: event types seen only in val/test must map to UNK
    val_only_df = get_split_df(val_ids).copy()
    val_only_df["event_type"] = 9999  # definitely not in train vocab
    val_only_t = prep.transform(val_only_df)
    assert (val_only_t["event_type"] == UNK).all(), "val-only type must map to UNK"

    # time_delta: first event of each entity → delta=0 → log1p(0)=0 → scaled value
    # After scaling: (0 - center) / scale. Just verify finite and no NaN.
    assert np.isfinite(train_t["time_delta"].values).all(), "time_delta must be finite"

    # NaN in numerical col → imputed with center → scaled to 0
    nan_df = get_split_df(test_ids).copy()
    nan_df["amount"] = np.nan
    nan_t = prep.transform(nan_df)
    assert not nan_t["amount"].isna().any(), "NaN numerical should be imputed"
    expected_val = 0.0  # (center - center) / scale = 0
    assert np.allclose(nan_t["amount"].values, expected_val), "imputed value must be 0 after scaling"

    # Save / load round-trip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    prep.save(tmp_path)
    prep2 = EventPreprocessor(config)
    prep2.load(tmp_path)

    assert prep2._fitted
    assert prep2.vocab == prep.vocab
    assert prep2.scaler_params == prep.scaler_params

    test_t2 = prep2.transform(get_split_df(test_ids))
    pd.testing.assert_frame_equal(test_t[["event_type", "amount", "num_feature", "cat_col", "time_delta"]],
                                  test_t2[["event_type", "amount", "num_feature", "cat_col", "time_delta"]])

    print("All assertions passed.")
