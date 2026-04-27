import json
import logging
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
        self._robust_scale_cols: set[str] = set(
            data_cfg.get("robust_scale_cols") or self._amount_cols
        )
        self.min_count: int = int(data_cfg.get("min_vocab_count", 5))
        # Transform configs from EDA findings:
        #   amount_transform: robust_scaler | sign_log1p | log1p
        #   time_transform:   log1p (default, backward-compat) | none
        self.amount_transform: str = data_cfg.get("amount_transform", "robust_scaler")
        self.time_transform: str = data_cfg.get("time_transform", "log1p")

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
        td_values = self._transform_time(
            pd.Series(time_delta.values.astype(float)), self.time_transform
        ).values
        self._fit_scaler("time_delta", td_values, robust=True)

        for col in self.numerical_cols:
            is_amount = col in self._amount_cols
            use_robust = col in self._robust_scale_cols
            raw = pd.Series(train_df[col].values.astype(float))
            values = (
                self._transform_amount(raw, self.amount_transform).values
                if is_amount
                else raw.values
            )
            self._fit_scaler(col, values, robust=use_robust)

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
        td_values = self._transform_time(
            pd.Series(time_delta.values.astype(float)), self.time_transform
        ).values
        result["time_delta"] = self._scale(td_values, "time_delta")

        for col in self.numerical_cols:
            is_amount = col in self._amount_cols
            raw = pd.Series(result[col].values.astype(float))
            center = self.scaler_params[col]["center"]
            if is_amount:
                values = self._transform_amount(raw, self.amount_transform).values
            else:
                values = raw.values
            values = np.where(np.isfinite(values), values, center)
            result[col] = self._scale(values, col)

        nan_cols = [c for c in result.columns if result[c].isna().any()]
        assert not nan_cols, f"NaN after transform in columns: {nan_cols}"

        return result

    @staticmethod
    def _transform_amount(series: pd.Series, transform: str) -> pd.Series:
        """
        Apply amount-specific pre-transform before RobustScaler.

        sign_log1p: sign(x) * log1p(|x|) — for datasets where amounts are
                    predominantly negative (e.g. gender: 80.9% expenses).
        log1p:      log1p(x) — for non-negative amounts with heavy tail.
        robust_scaler (default): identity — RobustScaler applied directly.
        """
        if transform == "sign_log1p":
            return np.sign(series) * np.log1p(np.abs(series))
        if transform == "log1p":
            return np.log1p(series)
        return series

    @staticmethod
    def _transform_time(series: pd.Series, transform: str) -> pd.Series:
        """
        Apply time_delta pre-transform before RobustScaler.

        log1p (default): for heavy-tail time_delta distributions (gender).
        none: for integer-day datasets where log1p adds little (rosbank, age_group).
        """
        if transform == "log1p":
            return np.log1p(series)
        return series

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
