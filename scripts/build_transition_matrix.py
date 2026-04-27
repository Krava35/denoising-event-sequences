#!/usr/bin/env python3
"""Build frozen transition matrix artifact from processed train events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corruption.transition_matrix import TransitionMatrix


def _load_yaml(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def _resolve_paths(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Path]:
    processed_dir = Path(args.processed_dir or "data/processed")
    tm_cfg = ((config.get("corruption") or {}).get("transition_matrix") or {})

    if args.events_path is not None:
        events_path = Path(args.events_path)
    else:
        events_path = processed_dir / "events.parquet"

    if args.splits_path is not None:
        splits_path = Path(args.splits_path)
    else:
        splits_path = processed_dir / "splits.json"

    if args.output_npy is not None:
        output_npy = Path(args.output_npy)
    else:
        output_npy = Path(tm_cfg.get("artifact_path", processed_dir / "transition_matrix.npy"))

    if args.output_meta is not None:
        output_meta = Path(args.output_meta)
    else:
        output_meta = Path(
            tm_cfg.get("metadata_path", processed_dir / "transition_matrix_meta.json")
        )

    return {
        "events_path": events_path,
        "splits_path": splits_path,
        "output_npy": output_npy,
        "output_meta": output_meta,
    }


def _resolve_params(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.get("data") or {}
    tm_cfg = ((config.get("corruption") or {}).get("transition_matrix") or {})

    return {
        "event_type_col": args.event_type_col or data_cfg.get("event_type_col", "event_type"),
        "entity_col": args.entity_col or data_cfg.get("group_col", "entity_id"),
        "timestamp_col": args.timestamp_col or data_cfg.get("timestamp_col", "timestamp"),
        "smoothing_alpha": (
            args.smoothing_alpha
            if args.smoothing_alpha is not None
            else float(tm_cfg.get("smoothing_alpha", 0.1))
        ),
        "min_count": (
            args.min_count
            if args.min_count is not None
            else int(tm_cfg.get("min_transition_count", 5))
        ),
        "fallback": args.fallback or str(tm_cfg.get("fallback", "frequency_aware")),
    }


def _load_train_sequences(
    events_path: Path,
    splits_path: Path,
    event_type_col: str,
    entity_col: str,
    timestamp_col: str,
) -> list[list[int]]:
    if not events_path.exists():
        raise FileNotFoundError(f"events.parquet not found: {events_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_path}")

    events = pd.read_parquet(events_path)
    with splits_path.open() as f:
        splits = json.load(f)

    if "train" not in splits:
        raise KeyError("splits.json must contain a 'train' key")
    if event_type_col not in events.columns:
        raise KeyError(f"Column '{event_type_col}' was not found in events.parquet")
    if entity_col not in events.columns:
        raise KeyError(f"Column '{entity_col}' was not found in events.parquet")

    train_ids = set(splits["train"])
    train_events = events[events[entity_col].isin(train_ids)].copy()
    if train_events.empty:
        raise ValueError("No train events found after filtering by splits['train']")

    sort_cols = [entity_col]
    if timestamp_col in train_events.columns:
        sort_cols.append(timestamp_col)
    train_events = train_events.sort_values(sort_cols, kind="stable")

    if train_events[event_type_col].isna().any():
        raise ValueError(f"Column '{event_type_col}' contains NaN in train events")

    event_ids = train_events[event_type_col].to_numpy(dtype="int64")
    if event_ids.min() < 0:
        raise ValueError("event_type ids must be non-negative")

    grouped = train_events.groupby(entity_col, sort=False)[event_type_col]
    return [series.astype("int64").tolist() for _, series in grouped]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--events-path", default=None)
    parser.add_argument("--splits-path", default=None)
    parser.add_argument("--output-npy", default=None)
    parser.add_argument("--output-meta", default=None)
    parser.add_argument("--event-type-col", default=None)
    parser.add_argument("--entity-col", default=None)
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument("--smoothing-alpha", type=float, default=None)
    parser.add_argument("--min-count", type=int, default=None)
    parser.add_argument("--fallback", choices=["frequency_aware", "uniform"], default=None)
    args = parser.parse_args()

    config = _load_yaml(args.config)
    paths = _resolve_paths(args, config)
    params = _resolve_params(args, config)

    train_sequences = _load_train_sequences(
        events_path=paths["events_path"],
        splits_path=paths["splits_path"],
        event_type_col=params["event_type_col"],
        entity_col=params["entity_col"],
        timestamp_col=params["timestamp_col"],
    )

    if not train_sequences:
        raise ValueError("No train sequences were built from train events")

    vocab_size = 1 + max(max(seq) for seq in train_sequences if seq)
    matrix = TransitionMatrix(
        vocab_size=vocab_size,
        smoothing_alpha=float(params["smoothing_alpha"]),
        min_count=int(params["min_count"]),
        fallback=str(params["fallback"]),
    )
    matrix.fit(train_sequences)
    matrix.save(str(paths["output_npy"]), str(paths["output_meta"]))

    print("Transition matrix built successfully")
    print(f"  Events source : {paths['events_path']}")
    print(f"  Splits source : {paths['splits_path']}")
    print(f"  Train sequences: {len(train_sequences)}")
    print(f"  Vocab size    : {matrix.vocab_size}")
    print(f"  Output .npy   : {paths['output_npy']}")
    print(f"  Output .json  : {paths['output_meta']}")
    print(
        "  Coverage      : "
        f"{matrix.covered_types}/{matrix.vocab_size} "
        f"({matrix.coverage * 100:.2f}%)"
    )
    print(f"  Fallback rows : {matrix.fallback_type_count}")


if __name__ == "__main__":
    main()
