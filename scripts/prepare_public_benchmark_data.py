#!/usr/bin/env python3
"""Prepare public benchmark datasets used by the Kaggle notebooks.

Supported datasets:
  * gender    — PTLS gender benchmark files under data/raw/gender
  * age_group — PTLS age-group benchmark files under data/raw/age-group

The public raw files contain duplicate raw timestamps per client.  DME expects a
strict event order, so this script creates a deterministic `event_timestamp`
column by preserving the raw timestamp and adding a within-client duplicate
ordinal.  Labels are merged before splitting, and only labelled entities are
kept.  The official hidden/public test files are intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import EventPreprocessor
from src.data.splits import make_entity_splits, save_splits
from src.utils.config import load_experiment_config, save_config

logger = logging.getLogger(__name__)

DATASET_CONFIG_PATHS = {
    "gender": "configs/datasets/gender.yaml",
    "age_group": "configs/datasets/age_group.yaml",
}

DEFAULT_RAW_SUBDIRS = {
    "gender": "gender",
    "age_group": "age-group",
}


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _resolve_dataset_dir(raw_root: Path, dataset_name: str) -> Path:
    if dataset_name == "gender" and (raw_root / "transactions.csv.gz").exists():
        return raw_root
    if dataset_name == "age_group" and (raw_root / "transactions_train.csv.gz").exists():
        return raw_root
    return raw_root / DEFAULT_RAW_SUBDIRS[dataset_name]


def parse_gender_tr_datetime(series: pd.Series) -> pd.Series:
    """Parse gender benchmark timestamps of form `day HH:MM:SS`."""
    parts = series.astype(str).str.strip().str.split(" ", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("gender tr_datetime must have format '<day> HH:MM:SS'")

    days = pd.to_numeric(parts[0], errors="coerce")
    day_delta = pd.to_timedelta(days, unit="D")
    time_delta = pd.to_timedelta(parts[1], errors="coerce")
    parsed = pd.Timestamp("2010-01-01") + day_delta + time_delta

    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise ValueError(f"Failed to parse {bad} gender tr_datetime value(s)")
    return parsed


def _canonicalize_datetime_timestamp(
    df: pd.DataFrame,
    entity_col: str,
    parsed_timestamp_col: str,
    output_col: str = "event_timestamp",
) -> tuple[pd.DataFrame, int]:
    """Create unique datetime timestamps while preserving raw order within ties."""
    result = df.copy()
    result["_raw_order"] = np.arange(len(result), dtype=np.int64)
    raw_dupes = int(result.duplicated([entity_col, parsed_timestamp_col]).sum())
    result["_tie_ordinal"] = result.groupby(
        [entity_col, parsed_timestamp_col], sort=False
    ).cumcount()
    result[output_col] = (
        result[parsed_timestamp_col]
        + pd.to_timedelta(result["_tie_ordinal"].astype(np.int64), unit="us")
    )
    result = result.sort_values([entity_col, output_col, "_raw_order"], kind="stable")
    result = result.drop(columns=["_raw_order", "_tie_ordinal"]).reset_index(drop=True)
    return result, raw_dupes


def _canonicalize_day_timestamp(
    df: pd.DataFrame,
    entity_col: str,
    day_col: str,
    output_col: str = "event_timestamp",
) -> tuple[pd.DataFrame, int]:
    """Create unique numeric timestamps from integer-day public benchmark time."""
    result = df.copy()
    result["_raw_order"] = np.arange(len(result), dtype=np.int64)
    result[day_col] = pd.to_numeric(result[day_col], errors="raise").astype(np.int64)
    raw_dupes = int(result.duplicated([entity_col, day_col]).sum())
    result["_tie_ordinal"] = result.groupby([entity_col, day_col], sort=False).cumcount()
    result[output_col] = (
        result[day_col].astype(np.float64) * 86_400.0
        + result["_tie_ordinal"].astype(np.float64)
    )
    result = result.sort_values([entity_col, output_col, "_raw_order"], kind="stable")
    result = result.drop(columns=["_raw_order", "_tie_ordinal"]).reset_index(drop=True)
    return result, raw_dupes


def make_gender_events(
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
    max_entities: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Merge and canonicalize the gender benchmark raw tables."""
    required_tx = {"customer_id", "tr_datetime", "mcc_code", "tr_type", "amount", "term_id"}
    required_labels = {"customer_id", "gender"}
    missing_tx = required_tx - set(transactions.columns)
    missing_labels = required_labels - set(labels.columns)
    if missing_tx:
        raise ValueError(f"gender transactions missing columns: {sorted(missing_tx)}")
    if missing_labels:
        raise ValueError(f"gender labels missing columns: {sorted(missing_labels)}")

    tx = transactions.copy()
    tx["parsed_tr_datetime"] = parse_gender_tr_datetime(tx["tr_datetime"])
    labels = labels[["customer_id", "gender"]].drop_duplicates("customer_id")
    df = tx.merge(labels, on="customer_id", how="inner", validate="many_to_one")

    if max_entities is not None:
        keep_entities = sorted(df["customer_id"].drop_duplicates().tolist())[:max_entities]
        df = df[df["customer_id"].isin(keep_entities)].copy()

    df, raw_dupes = _canonicalize_datetime_timestamp(
        df, entity_col="customer_id", parsed_timestamp_col="parsed_tr_datetime"
    )
    df["mcc_code"] = df["mcc_code"].astype(str)
    df["tr_type"] = df["tr_type"].astype(str)
    df["term_id"] = df["term_id"].fillna("__MISSING__").astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).astype(float)
    df["gender"] = pd.to_numeric(df["gender"], errors="raise").astype(int)

    out_cols = [
        "customer_id",
        "event_timestamp",
        "tr_datetime",
        "mcc_code",
        "amount",
        "tr_type",
        "term_id",
        "gender",
    ]
    df = df[out_cols]
    report = {
        "raw_duplicate_entity_timestamp_pairs": raw_dupes,
        "canonical_duplicate_entity_timestamp_pairs": int(
            df.duplicated(["customer_id", "event_timestamp"]).sum()
        ),
        "official_test_ignored": True,
    }
    return df, report


def make_age_group_events(
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
    max_entities: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Merge and canonicalize the age-group benchmark raw tables."""
    required_tx = {"client_id", "trans_date", "small_group", "amount_rur"}
    required_labels = {"client_id", "bins"}
    missing_tx = required_tx - set(transactions.columns)
    missing_labels = required_labels - set(labels.columns)
    if missing_tx:
        raise ValueError(f"age_group transactions missing columns: {sorted(missing_tx)}")
    if missing_labels:
        raise ValueError(f"age_group labels missing columns: {sorted(missing_labels)}")

    labels = labels[["client_id", "bins"]].drop_duplicates("client_id")
    df = transactions.copy().merge(labels, on="client_id", how="inner", validate="many_to_one")

    if max_entities is not None:
        keep_entities = sorted(df["client_id"].drop_duplicates().tolist())[:max_entities]
        df = df[df["client_id"].isin(keep_entities)].copy()

    df, raw_dupes = _canonicalize_day_timestamp(
        df, entity_col="client_id", day_col="trans_date"
    )
    df["small_group"] = df["small_group"].astype(str)
    df["amount_rur"] = pd.to_numeric(df["amount_rur"], errors="coerce").fillna(0.0).astype(float)
    df["bins"] = pd.to_numeric(df["bins"], errors="raise").astype(int)

    out_cols = ["client_id", "event_timestamp", "trans_date", "small_group", "amount_rur", "bins"]
    df = df[out_cols]
    report = {
        "raw_duplicate_entity_timestamp_pairs": raw_dupes,
        "canonical_duplicate_entity_timestamp_pairs": int(
            df.duplicated(["client_id", "event_timestamp"]).sum()
        ),
        "official_test_ignored": True,
    }
    return df, report


def build_public_benchmark_config(
    dataset_name: str,
    base_config_path: str | Path = "configs/base.yaml",
    max_seq_len: int = 512,
) -> dict:
    """Load repo config and patch public benchmark timestamp/task settings."""
    if dataset_name not in DATASET_CONFIG_PATHS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    config = load_experiment_config(
        base_path=str(base_config_path),
        dataset_path=DATASET_CONFIG_PATHS[dataset_name],
    )
    config["data"]["timestamp_col"] = "event_timestamp"
    config["data"]["max_seq_len"] = int(max_seq_len)
    config["model"]["max_seq_len"] = int(max_seq_len)
    if dataset_name == "gender":
        # `term_id` is very high-cardinality in the public gender benchmark.
        # Reconstructing it creates [B, L, |term_id_vocab|] logits and easily OOMs
        # on 40GB GPUs. Keep it in canonical_events.parquet for external baselines,
        # but exclude it from DME categorical reconstruction.
        config["data"]["categorical_cols"] = ["tr_type"]
    config["training"]["num_classes"] = 2 if dataset_name == "gender" else 4
    config["training"]["task"] = "binary" if dataset_name == "gender" else "multiclass"
    config["training"]["selection_metric"] = "roc_auc" if dataset_name == "gender" else "macro_f1"
    return config


def load_public_benchmark_raw(
    dataset_name: str,
    raw_root: str | Path = "data/raw",
    max_entities: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    raw_root = Path(raw_root)
    dataset_dir = _resolve_dataset_dir(raw_root, dataset_name)
    if dataset_name == "gender":
        tx_path = dataset_dir / "transactions.csv.gz"
        labels_path = dataset_dir / "gender_train.csv"
        if not tx_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Expected gender raw files under {dataset_dir}: "
                "transactions.csv.gz and gender_train.csv"
            )
        transactions = pd.read_csv(tx_path)
        labels = pd.read_csv(labels_path)
        df, report = make_gender_events(transactions, labels, max_entities=max_entities)
    elif dataset_name == "age_group":
        tx_path = dataset_dir / "transactions_train.csv.gz"
        labels_path = dataset_dir / "train_target.csv"
        if not tx_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Expected age_group raw files under {dataset_dir}: "
                "transactions_train.csv.gz and train_target.csv"
            )
        transactions = pd.read_csv(tx_path)
        labels = pd.read_csv(labels_path)
        df, report = make_age_group_events(transactions, labels, max_entities=max_entities)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    report.update(
        {
            "dataset": dataset_name,
            "raw_root": str(raw_root),
            "raw_dataset_dir": str(dataset_dir),
        }
    )
    return df, report


def _entity_report(df: pd.DataFrame, config: dict) -> dict:
    entity_col = config["data"]["group_col"]
    target_col = config["data"]["target_col"]
    event_type_col = config["data"]["event_type_col"]
    seq_lens = df.groupby(entity_col).size()
    labels = df.groupby(entity_col)[target_col].first()
    return {
        "rows": int(len(df)),
        "entities": int(df[entity_col].nunique()),
        "unique_event_types": int(df[event_type_col].nunique()),
        "class_counts": _jsonify(labels.value_counts().sort_index().to_dict()),
        "sequence_length": {
            "min": int(seq_lens.min()),
            "p25": float(seq_lens.quantile(0.25)),
            "p50": float(seq_lens.quantile(0.50)),
            "mean": float(seq_lens.mean()),
            "p75": float(seq_lens.quantile(0.75)),
            "p95": float(seq_lens.quantile(0.95)),
            "max": int(seq_lens.max()),
        },
    }


def prepare_public_benchmark_dataset(
    dataset_name: str,
    raw_root: str | Path = "data/raw",
    output_dir: str | Path | None = None,
    base_config_path: str | Path = "configs/base.yaml",
    max_seq_len: int = 512,
    max_entities: int | None = None,
    dry_run: bool = False,
) -> dict:
    config = build_public_benchmark_config(dataset_name, base_config_path, max_seq_len)
    df, report = load_public_benchmark_raw(dataset_name, raw_root, max_entities=max_entities)

    entity_col = config["data"]["group_col"]
    timestamp_col = config["data"]["timestamp_col"]
    target_col = config["data"]["target_col"]
    seed = int((config.get("seed") or {}).get("global_seed", 42))

    duplicate_after = int(df.duplicated([entity_col, timestamp_col]).sum())
    if duplicate_after:
        raise ValueError(
            f"Canonicalized dataset still has {duplicate_after} duplicate "
            f"({entity_col}, {timestamp_col}) pairs"
        )

    df = df.sort_values([entity_col, timestamp_col], kind="stable").reset_index(drop=True)
    report.update(_entity_report(df, config))
    logger.info(
        "%s canonicalized: %d rows, %d entities",
        dataset_name,
        report["rows"],
        report["entities"],
    )

    if dry_run:
        logger.info("--dry-run: no output written")
        return report

    out_dir = Path(output_dir) if output_dir is not None else Path("data/processed") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_path = out_dir / "canonical_events.parquet"
    df.to_parquet(canonical_path, index=False)

    splits = make_entity_splits(
        df,
        entity_col=entity_col,
        target_col=target_col,
        train_ratio=float(config["data"].get("train_ratio", 0.70)),
        val_ratio=float(config["data"].get("val_ratio", 0.15)),
        test_ratio=float(config["data"].get("test_ratio", 0.15)),
        seed=seed,
        stratify=True,
    )
    save_splits(splits, out_dir / "splits.json")

    # Training/evaluation scripts pass events.parquet back through EventSequenceDataset,
    # and EventSequenceDataset applies the fitted preprocessor internally. Keep
    # events.parquet as canonical raw events to avoid double-transforming vocab ids
    # and continuous features.
    events_path = out_dir / "events.parquet"
    df.to_parquet(events_path, index=False)

    preprocessor = EventPreprocessor(config)
    preprocessor.fit(df, splits["train"])
    transformed = preprocessor.transform(df)
    transformed_path = out_dir / "transformed_events.parquet"
    transformed.to_parquet(transformed_path, index=False)

    with (out_dir / "preprocessor.pkl").open("wb") as f:
        pickle.dump(preprocessor, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_config(config, str(out_dir / "prepared_config.yaml"))

    report["output_dir"] = str(out_dir)
    report["events_are_raw_pretransform"] = True
    report["transformed_events_path"] = str(transformed_path)
    report["split_sizes"] = {name: len(ids) for name, ids in splits.items()}
    report["event_type_vocab_size"] = len(preprocessor.vocab.get(preprocessor.event_type_col, {}))
    report["categorical_vocab_sizes"] = {
        col: len(preprocessor.vocab.get(col, {})) for col in preprocessor.categorical_cols
    }

    with (out_dir / "data_report.json").open("w") as f:
        json.dump(_jsonify(report), f, indent=2, ensure_ascii=False)

    logger.info("Saved prepared artifacts to %s", out_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare public gender/age_group benchmark data for DME experiments."
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG_PATHS), required=True)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )

    report = prepare_public_benchmark_dataset(
        dataset_name=args.dataset,
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        base_config_path=args.base_config,
        max_seq_len=args.max_seq_len,
        max_entities=args.max_entities,
        dry_run=args.dry_run,
    )
    print(json.dumps(_jsonify(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
