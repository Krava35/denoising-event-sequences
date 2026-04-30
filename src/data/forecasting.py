from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from src.data.preprocessing import EventPreprocessor

DERIVED_TIME_FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "time_since_first",
    "relative_age",
]


def get_derived_time_feature_names(config: dict) -> list[str]:
    data_cfg = config.get("data", {})
    if not data_cfg.get("use_time_features", False):
        return []
    mode = data_cfg.get("time_feature_mode", "derived_numeric")
    if mode != "derived_numeric":
        raise ValueError(f"Unsupported time_feature_mode: {mode}")
    return list(DERIVED_TIME_FEATURE_NAMES)


def get_num_feature_dim(preprocessor: EventPreprocessor, config: dict) -> int:
    return len(preprocessor.numerical_cols) + len(get_derived_time_feature_names(config))


def _timestamps_to_seconds(timestamps: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(timestamps):
        return timestamps.to_numpy(dtype=float)

    dt = pd.to_datetime(timestamps, errors="coerce")
    seconds = dt.astype("int64").to_numpy(dtype=float) / 1_000_000_000.0
    valid = dt.notna().to_numpy()
    return np.where(valid & np.isfinite(seconds), seconds, 0.0)


def _timestamps_to_datetime(timestamps: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(timestamps):
        seconds = timestamps.to_numpy(dtype=float)
        return pd.Series(pd.to_datetime(seconds, unit="s", errors="coerce"))
    return pd.Series(pd.to_datetime(timestamps, errors="coerce"))


def compute_derived_time_features(timestamps: pd.Series) -> np.ndarray:
    """Return finite per-event calendar and sequence-age features."""
    n = len(timestamps)
    if n == 0:
        return np.empty((0, len(DERIVED_TIME_FEATURE_NAMES)), dtype=np.float32)

    dt = _timestamps_to_datetime(timestamps.reset_index(drop=True))
    hour = dt.dt.hour.fillna(0).to_numpy(dtype=float)
    day_of_week = dt.dt.dayofweek.fillna(0).to_numpy(dtype=float)
    month = dt.dt.month.fillna(1).to_numpy(dtype=float)

    seconds = _timestamps_to_seconds(timestamps.reset_index(drop=True))
    elapsed = np.maximum(seconds - seconds[0], 0.0)
    span = float(np.max(elapsed)) if len(elapsed) else 0.0
    denom = span if span > 0.0 else 1.0

    hour_angle = 2.0 * np.pi * hour / 24.0
    dow_angle = 2.0 * np.pi * day_of_week / 7.0
    month_angle = 2.0 * np.pi * (month - 1.0) / 12.0

    features = np.column_stack(
        [
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(dow_angle),
            np.cos(dow_angle),
            np.sin(month_angle),
            np.cos(month_angle),
            (day_of_week >= 5.0).astype(float),
            np.log1p(elapsed),
            elapsed / denom,
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _valid_cut_bounds(n_events: int, config: dict) -> tuple[int, int] | None:
    f_cfg = config.get("forecasting", {})
    min_ratio = float(f_cfg.get("cut_min_ratio", 0.50))
    max_ratio = float(f_cfg.get("cut_max_ratio", 0.80))
    min_future = int(f_cfg.get("min_future_events", 2))

    if n_events < min_future + 1:
        return None

    cut_low = max(1, int(np.floor(n_events * min_ratio)))
    cut_high = min(n_events - min_future, int(np.floor(n_events * max_ratio)))
    if cut_high < cut_low:
        cut_low = max(1, n_events - min_future)
        cut_high = n_events - min_future
    if cut_high < cut_low:
        return None
    return cut_low, cut_high


def has_valid_forecast_cut(n_events: int, config: dict) -> bool:
    return _valid_cut_bounds(n_events, config) is not None


def sample_forecast_cut(n_events: int, config: dict) -> int:
    bounds = _valid_cut_bounds(n_events, config)
    if bounds is None:
        raise ValueError(f"Need at least one prefix event and future events, got n={n_events}")
    cut_low, cut_high = bounds
    return int(np.random.randint(cut_low, cut_high + 1))


def _candidate_cuts(n_events: int, config: dict) -> list[int]:
    bounds = _valid_cut_bounds(n_events, config)
    if bounds is None:
        return []
    cut_low, cut_high = bounds
    return sorted({cut_low, (cut_low + cut_high) // 2, cut_high})


def _quantile_edges(values: list[float], num_buckets: int) -> list[float]:
    if num_buckets <= 1:
        return []
    if not values:
        return [0.0] * (num_buckets - 1)
    qs = np.linspace(0.0, 1.0, num_buckets + 1)[1:-1]
    edges = np.quantile(np.asarray(values, dtype=float), qs)
    return [float(x) for x in np.nan_to_num(edges, nan=0.0, posinf=0.0, neginf=0.0)]


def _bucketize(value: float, edges: list[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=float), value, side="right"))


def _format_bucket_value(value: float, *, integer: bool = False) -> str:
    if not np.isfinite(value):
        return "inf"
    if integer:
        return str(int(round(value)))
    return f"{value:.6g}"


def _bucket_bounds(edges: list[float], bucket: int) -> tuple[float, float]:
    clean_edges = [
        float(x)
        for x in np.nan_to_num(np.asarray(edges, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ]
    bucket = int(np.clip(bucket, 0, len(clean_edges)))
    lower = float("-inf") if bucket == 0 else clean_edges[bucket - 1]
    upper = float("inf") if bucket == len(clean_edges) else clean_edges[bucket]
    return lower, upper


def _make_bucket_labels(
    edges: list[float],
    *,
    name: str,
    integer: bool = False,
    non_negative: bool = False,
) -> list[str]:
    clean_edges = [
        float(x)
        for x in np.nan_to_num(np.asarray(edges, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ]
    if non_negative:
        clean_edges = [max(0.0, x) for x in clean_edges]
    labels: list[str] = []
    for bucket in range(len(clean_edges) + 1):
        lower, upper = _bucket_bounds(clean_edges, bucket)
        if bucket == 0:
            labels.append(f"{name} <= {_format_bucket_value(upper, integer=integer)}")
        elif bucket == len(clean_edges):
            labels.append(f"{name} > {_format_bucket_value(lower, integer=integer)}")
        else:
            labels.append(
                f"{_format_bucket_value(lower, integer=integer)} < {name} <= "
                f"{_format_bucket_value(upper, integer=integer)}"
            )
    return labels


def get_count_bucket_labels(forecast_stats: dict) -> list[str]:
    return _make_bucket_labels(
        forecast_stats.get("count_bucket_edges", []),
        name="future_count",
        integer=True,
    )


def get_gap_bucket_labels(forecast_stats: dict) -> list[str]:
    return _make_bucket_labels(
        forecast_stats.get("gap_bucket_edges", []),
        name="gap",
        non_negative=True,
    )


def _prepare_transformed_subset(
    df_events: pd.DataFrame,
    entity_ids: list,
    preprocessor: EventPreprocessor,
) -> pd.DataFrame:
    entity_set = set(entity_ids)
    df_subset = df_events[df_events[preprocessor.entity_col].isin(entity_set)].copy()
    transformed = preprocessor.transform(df_subset)
    transformed["_ec"] = df_subset[preprocessor.entity_col].values
    transformed["_tc"] = df_subset[preprocessor.timestamp_col].values
    return transformed.sort_values(["_ec", "_tc"], kind="stable")


def build_forecast_stats(
    df_events: pd.DataFrame,
    entity_ids: list,
    preprocessor: EventPreprocessor,
    config: dict,
) -> dict:
    """Build train-only statistics used by forecast targets."""
    transformed = _prepare_transformed_subset(df_events, entity_ids, preprocessor)

    event_vocab_size = len(preprocessor.vocab.get(preprocessor.event_type_col, {}))
    cat_vocab_sizes = [
        len(preprocessor.vocab.get(col, {})) for col in preprocessor.categorical_cols
    ]
    f_cfg = config.get("forecasting", {})
    count_num_buckets = int(f_cfg.get("count_num_buckets", 6))
    gap_num_buckets = int(f_cfg.get("gap_num_buckets", 6))

    event_values = transformed[preprocessor.event_type_col].to_numpy(dtype=np.int64)
    event_counts = np.bincount(
        np.clip(event_values, 0, max(event_vocab_size - 1, 0)),
        minlength=event_vocab_size,
    ).astype(float)
    eps = 1e-6
    event_freq = (event_counts + eps) / max(float(event_counts.sum() + eps * event_vocab_size), eps)

    cat_global_freq: list[list[float]] = []
    for col, vocab_size in zip(preprocessor.categorical_cols, cat_vocab_sizes):
        values = transformed[col].to_numpy(dtype=np.int64)
        counts = np.bincount(np.clip(values, 0, vocab_size - 1), minlength=vocab_size).astype(float)
        freq = (counts + eps) / max(float(counts.sum() + eps * vocab_size), eps)
        cat_global_freq.append(freq.astype(float).tolist())

    amount_cols = list(config.get("data", {}).get("amount_cols") or [])
    amount_col = next((col for col in amount_cols if col in preprocessor.numerical_cols), None)
    if amount_col is None and preprocessor.numerical_cols:
        amount_col = preprocessor.numerical_cols[0]
    amount_feature_index = (
        preprocessor.numerical_cols.index(amount_col) if amount_col is not None else None
    )
    if amount_col is not None and amount_col in transformed.columns:
        amount_values = transformed[amount_col].to_numpy(dtype=float)
        clean_amount = amount_values[np.isfinite(amount_values)]
        if len(clean_amount):
            amount_clip = np.quantile(clean_amount, [0.01, 0.99]).astype(float).tolist()
        else:
            amount_clip = [0.0, 0.0]
    else:
        amount_clip = [0.0, 0.0]

    future_counts: list[float] = []
    first_future_gaps: list[float] = []
    for _, ent in transformed.groupby("_ec", sort=False):
        n_events = len(ent)
        for cut in _candidate_cuts(n_events, config):
            future_counts.append(float(n_events - cut))
            first_future_gaps.append(float(ent["time_delta"].iloc[cut]))

    count_edges = _quantile_edges(future_counts, count_num_buckets)
    gap_edges = _quantile_edges(first_future_gaps, gap_num_buckets)
    median_count = float(np.median(future_counts)) if future_counts else 0.0
    median_gap = float(np.median(first_future_gaps)) if first_future_gaps else 0.0

    return {
        "event_type_col": preprocessor.event_type_col,
        "categorical_cols": list(preprocessor.categorical_cols),
        "event_type_vocab_size": event_vocab_size,
        "cat_vocab_sizes": cat_vocab_sizes,
        "event_type_global_freq": event_freq.astype(float).tolist(),
        "cat_global_freq": cat_global_freq,
        "count_bucket_edges": count_edges,
        "gap_bucket_edges": gap_edges,
        "count_bucket_labels": _make_bucket_labels(
            count_edges,
            name="future_count",
            integer=True,
        ),
        "gap_bucket_labels": _make_bucket_labels(gap_edges, name="gap", non_negative=True),
        "median_count_bucket": _bucketize(median_count, count_edges),
        "median_gap_bucket": _bucketize(median_gap, gap_edges),
        "amount_clip_quantiles": amount_clip,
        "amount_col": amount_col,
        "amount_feature_index": amount_feature_index,
    }


def save_forecast_stats(stats: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(stats, f, indent=2)


def load_forecast_stats(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def make_forecast_targets(sample: dict, cut: int, forecast_stats: dict) -> dict:
    future_event_type = sample["event_type"][cut:]
    future_time_delta = sample["time_delta"][cut:]
    future_num = sample["num"][cut:]
    future_cat = sample["cat"][cut:]

    n_future = max(int(len(future_event_type)), 1)
    event_vocab_size = int(forecast_stats["event_type_vocab_size"])
    event_counts = np.bincount(
        np.clip(future_event_type.astype(np.int64), 0, event_vocab_size - 1),
        minlength=event_vocab_size,
    ).astype(float)
    eps = 1e-6
    future_freq = (event_counts + eps) / (float(event_counts.sum()) + eps * event_vocab_size)
    global_freq = np.asarray(forecast_stats["event_type_global_freq"], dtype=float)
    global_freq = np.clip(global_freq, eps, None)
    event_type_profile = np.log(future_freq / global_freq).astype(np.float32)

    amount_feature_index = forecast_stats.get("amount_feature_index")
    if amount_feature_index is not None and future_num.shape[1] > int(amount_feature_index):
        values = future_num[:, int(amount_feature_index)].astype(float)
        lo, hi = forecast_stats.get("amount_clip_quantiles", [0.0, 0.0])
        values = np.clip(values, float(lo), float(hi))
        amount_stats = np.asarray(
            [
                float(np.mean(values)),
                float(np.sum(values)),
                float(np.std(values)),
                float(np.mean(values > 0.0)),
            ],
            dtype=np.float32,
        )
    else:
        amount_stats = np.zeros(4, dtype=np.float32)

    cat_profiles: list[np.ndarray] = []
    for j, vocab_size in enumerate(forecast_stats.get("cat_vocab_sizes", [])):
        if future_cat.shape[1] <= j:
            cat_profiles.append(np.zeros(int(vocab_size), dtype=np.float32))
            continue
        counts = np.bincount(
            np.clip(future_cat[:, j].astype(np.int64), 0, int(vocab_size) - 1),
            minlength=int(vocab_size),
        ).astype(float)
        cat_profiles.append((counts / max(float(n_future), 1.0)).astype(np.float32))

    first_gap = float(future_time_delta[0]) if len(future_time_delta) else 0.0

    return {
        "future_event_type_profile": event_type_profile,
        "future_count_bucket": _bucketize(
            float(len(future_event_type)),
            forecast_stats["count_bucket_edges"],
        ),
        "future_amount_stats": np.nan_to_num(amount_stats, nan=0.0, posinf=0.0, neginf=0.0),
        "future_gap_bucket": _bucketize(first_gap, forecast_stats["gap_bucket_edges"]),
        "future_cat_profiles": cat_profiles,
    }


def _as_numpy(value, *, index: int = 0) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().float().numpy()
    else:
        arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim >= 2:
        return np.asarray(arr[index], dtype=float)
    return np.asarray(arr, dtype=float)


def _normalise_nonnegative(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    clipped = np.clip(values, 0.0, None)
    total = float(clipped.sum())
    if total > 0.0:
        return clipped / total

    shifted = values - float(np.max(values)) if len(values) else values
    exp_values = np.exp(np.clip(shifted, -50.0, 50.0))
    exp_total = float(exp_values.sum())
    if exp_total > 0.0:
        return exp_values / exp_total
    return np.full_like(values, 1.0 / max(len(values), 1), dtype=float)


def _profile_to_frequency(profile: np.ndarray, forecast_stats: dict) -> np.ndarray:
    global_freq = np.asarray(forecast_stats["event_type_global_freq"], dtype=float)
    size = min(len(profile), len(global_freq))
    if size == 0:
        return np.empty(0, dtype=float)
    profile = np.nan_to_num(profile[:size], nan=0.0, posinf=0.0, neginf=0.0)
    global_freq = np.clip(global_freq[:size], 1e-12, None)
    frequency = np.exp(np.clip(profile, -30.0, 30.0)) * global_freq
    return _normalise_nonnegative(frequency)


def _invert_vocab(mapping: dict | None) -> dict[int, str]:
    if not isinstance(mapping, dict):
        return {}
    inverted: dict[int, str] = {}
    for key, value in mapping.items():
        if isinstance(value, (int, np.integer)):
            inverted[int(value)] = str(key)
        elif isinstance(key, (int, np.integer)):
            inverted[int(key)] = str(value)
        elif isinstance(key, str) and key.isdigit():
            inverted[int(key)] = str(value)
    return inverted


def _lookup_vocab_mapping(
    vocab: dict,
    *,
    column: str | None,
    size: int,
    used_columns: set[str] | None = None,
) -> dict[int, str]:
    used_columns = used_columns if used_columns is not None else set()
    if column and column in vocab and isinstance(vocab[column], dict):
        used_columns.add(column)
        return _invert_vocab(vocab[column])

    preferred = ["event_type", "event_type_col", "mcc", "merchant_category"]
    for key in preferred:
        if key in vocab and key not in used_columns and isinstance(vocab[key], dict):
            candidate = _invert_vocab(vocab[key])
            if candidate:
                used_columns.add(key)
                return candidate

    for key, value in vocab.items():
        if key in used_columns or not isinstance(value, dict):
            continue
        candidate = _invert_vocab(value)
        if candidate and (not size or max(candidate) < size or len(candidate) == size):
            used_columns.add(key)
            return candidate

    return {i: str(i) for i in range(size)}


def _valid_profile_ids(id_to_token: dict[int, str], size: int) -> list[int]:
    valid = [
        idx
        for idx, token in sorted(id_to_token.items())
        if 0 <= idx < size and not str(token).startswith("<")
    ]
    if valid:
        return valid
    return list(range(size))


def _top_profile_entries(
    probabilities: np.ndarray,
    id_to_token: dict[int, str],
    *,
    top_k: int,
) -> list[dict]:
    size = len(probabilities)
    valid_ids = _valid_profile_ids(id_to_token, size)
    ranked = sorted(valid_ids, key=lambda idx: float(probabilities[idx]), reverse=True)
    entries: list[dict] = []
    for idx in ranked[: max(1, top_k)]:
        entries.append(
            {
                "id": int(idx),
                "token": id_to_token.get(idx, str(idx)),
                "probability": float(np.clip(probabilities[idx], 0.0, 1.0)),
            }
        )
    return entries


def _bucket_label(stats: dict, key: str, bucket: int) -> str:
    if key == "count":
        labels = stats.get("count_bucket_labels") or get_count_bucket_labels(stats)
    elif key == "gap":
        labels = stats.get("gap_bucket_labels") or get_gap_bucket_labels(stats)
    else:
        raise ValueError(f"Unknown bucket label key: {key}")
    bucket = int(np.clip(bucket, 0, len(labels) - 1))
    return labels[bucket]


def _expected_amount_stats(
    values: np.ndarray,
    forecast_stats: dict,
    *,
    count_bucket: int,
) -> dict[str, float]:
    amount = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    if len(amount) < 4:
        amount = np.pad(amount, (0, 4 - len(amount)), constant_values=0.0)

    lo, hi = forecast_stats.get("amount_clip_quantiles", [0.0, 0.0])
    lo = float(lo)
    hi = float(hi)
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        lo, hi = min(lo, 0.0), max(hi, 0.0)

    count_lower, count_upper = _bucket_bounds(
        forecast_stats.get("count_bucket_edges", []),
        count_bucket,
    )
    count_cap = count_upper
    if not np.isfinite(count_cap):
        count_cap = max(count_lower if np.isfinite(count_lower) else 0.0, 1.0) + 1.0
    count_cap = max(1.0, float(count_cap))

    sum_lo = min(lo * count_cap, hi * count_cap)
    sum_hi = max(lo * count_cap, hi * count_cap)
    std_cap = max(abs(hi - lo), 0.0)

    return {
        "mean": float(np.clip(amount[0], lo, hi)),
        "sum": float(np.clip(amount[1], sum_lo, sum_hi)),
        "std": float(np.clip(amount[2], 0.0, std_cap if std_cap > 0.0 else None)),
        "positive_share": float(np.clip(amount[3], 0.0, 1.0)),
    }


def scenario_from_outputs(
    outputs: dict,
    forecast_stats: dict,
    vocab: dict,
    top_k: int = 5,
) -> dict:
    """Convert raw forecast head outputs into a constrained aggregate scenario.

    The helper consumes one model output or the first row of a batched output.
    Returned event/category ids are constrained to the train vocab, bucket names
    come from forecast_stats, and amount/gap summaries are clipped to finite
    train-derived ranges.
    """
    event_profile = _as_numpy(outputs["future_event_type_profile"])
    event_prob = _profile_to_frequency(event_profile, forecast_stats)
    event_vocab = _lookup_vocab_mapping(
        vocab,
        column=forecast_stats.get("event_type_col"),
        size=len(event_prob),
    )

    count_logits = _as_numpy(outputs["future_count_bucket_logits"])
    count_bucket = int(np.argmax(count_logits)) if len(count_logits) else 0
    count_labels = forecast_stats.get("count_bucket_labels") or get_count_bucket_labels(
        forecast_stats
    )
    count_bucket = int(np.clip(count_bucket, 0, len(count_labels) - 1))

    gap_logits = _as_numpy(outputs["future_gap_bucket_logits"])
    gap_bucket = int(np.argmax(gap_logits)) if len(gap_logits) else 0
    gap_labels = forecast_stats.get("gap_bucket_labels") or get_gap_bucket_labels(
        forecast_stats
    )
    gap_bucket = int(np.clip(gap_bucket, 0, len(gap_labels) - 1))

    cat_profiles: dict[str, list[dict]] = {}
    used_columns: set[str] = {forecast_stats.get("event_type_col", "")}
    cat_columns = forecast_stats.get("categorical_cols") or []
    cat_outputs = outputs.get("future_cat_profiles") or []
    for j, pred in enumerate(cat_outputs):
        values = _as_numpy(pred)
        probabilities = _normalise_nonnegative(values)
        col = cat_columns[j] if j < len(cat_columns) else f"cat_{j}"
        cat_vocab = _lookup_vocab_mapping(
            vocab,
            column=col,
            size=len(probabilities),
            used_columns=used_columns,
        )
        cat_profiles[col] = _top_profile_entries(
            probabilities,
            cat_vocab,
            top_k=top_k,
        )

    return {
        "future_count_bucket": _bucket_label(forecast_stats, "count", count_bucket),
        "dominant_event_types": _top_profile_entries(event_prob, event_vocab, top_k=top_k),
        "expected_amount_stats": _expected_amount_stats(
            _as_numpy(outputs["future_amount_stats"]),
            forecast_stats,
            count_bucket=count_bucket,
        ),
        "gap_bucket": _bucket_label(forecast_stats, "gap", gap_bucket),
        "categorical_profiles": cat_profiles,
    }


def scenario_from_targets(
    targets: dict,
    forecast_stats: dict,
    vocab: dict,
    *,
    index: int = 0,
    top_k: int = 5,
) -> dict:
    """Summarize forecast targets with the same constrained schema as predictions."""
    count_values = _as_numpy(targets["future_count_bucket"])
    gap_values = _as_numpy(targets["future_gap_bucket"])
    count_bucket = int(count_values[index] if len(count_values) > index else count_values[0])
    gap_bucket = int(gap_values[index] if len(gap_values) > index else gap_values[0])
    count_size = len(forecast_stats.get("count_bucket_edges", [])) + 1
    gap_size = len(forecast_stats.get("gap_bucket_edges", [])) + 1

    count_logits = np.full(count_size, -1.0, dtype=float)
    count_logits[int(np.clip(count_bucket, 0, count_size - 1))] = 1.0
    gap_logits = np.full(gap_size, -1.0, dtype=float)
    gap_logits[int(np.clip(gap_bucket, 0, gap_size - 1))] = 1.0

    cat_profiles = []
    for profile in targets.get("future_cat_profiles", []):
        cat_profiles.append(_as_numpy(profile, index=index))

    outputs = {
        "future_event_type_profile": _as_numpy(
            targets["future_event_type_profile"],
            index=index,
        ),
        "future_count_bucket_logits": count_logits,
        "future_amount_stats": _as_numpy(targets["future_amount_stats"], index=index),
        "future_gap_bucket_logits": gap_logits,
        "future_cat_profiles": cat_profiles,
    }
    return scenario_from_outputs(outputs, forecast_stats, vocab, top_k=top_k)
