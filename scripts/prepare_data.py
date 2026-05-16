#!/usr/bin/env python3
"""Offline data preparation pipeline for DME-Encoder.

Steps:
  1.  Load raw CSV / Parquet
  2.  Validate required columns, no duplicate (entity, timestamp) pairs
  3.  Sort by (entity_id, timestamp)
  4.  Compute raw time_delta (seconds)
  5.  make_entity_splits — stratified by target, strict entity-level
  6.  Save splits.json
  7.  Fit EventPreprocessor on train split only
  8.  Save events.parquet (raw pre-transform events for EventSequenceDataset)
  9.  Save transformed_events.parquet (diagnostic cache with scaled/vocab features)
  10. Save preprocessor.pkl
  11. Print statistics

Use --dry-run to validate without writing any output.

Example:
  python scripts/prepare_data.py \\
      --config base.yaml \\
      --input data/raw/rosbank.parquet \\
      --output-dir data/processed/rosbank
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import EventPreprocessor
from src.data.splits import make_entity_splits, save_splits
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_data(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    elif p.suffix == ".tsv":
        df = pd.read_csv(p, sep="\t")
    else:
        raise ValueError(f"Unsupported format '{p.suffix}'. Use .parquet, .csv, or .tsv.")
    logger.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), p.name)
    return df


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame, config: dict) -> None:
    data_cfg = config["data"]
    entity_col: str = data_cfg["group_col"]
    timestamp_col: str = data_cfg["timestamp_col"]
    event_type_col: str = data_cfg["event_type_col"]
    target_col: str = data_cfg["target_col"]
    num_cols: list[str] = list(data_cfg.get("numerical_cols") or [])
    cat_cols: list[str] = list(data_cfg.get("categorical_cols") or [])

    required = {entity_col, timestamp_col, event_type_col, target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feature_cols = set(num_cols) | set(cat_cols)
    missing_features = feature_cols - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns declared in config: {sorted(missing_features)}")

    # No null values in key columns
    for col in [entity_col, timestamp_col, event_type_col]:
        n_null = int(df[col].isna().sum())
        if n_null:
            raise ValueError(f"Column '{col}' contains {n_null} null value(s)")

    # No duplicate (entity, timestamp) pairs
    n_dup = int(df.duplicated(subset=[entity_col, timestamp_col]).sum())
    if n_dup:
        raise ValueError(
            f"Found {n_dup} duplicate ({entity_col}, {timestamp_col}) pair(s). "
            "Resolve before proceeding."
        )

    # Warn on entities below min_seq_len
    min_seq_len = int(data_cfg.get("min_seq_len", 2))
    seq_lens = df.groupby(entity_col).size()
    n_short = int((seq_lens < min_seq_len).sum())
    if n_short:
        logger.warning(
            "%d entities have fewer than %d events (min_seq_len=%d) — consider filtering them",
            n_short, min_seq_len, min_seq_len,
        )

    n_entities = df[entity_col].nunique()
    logger.info(
        "Validation passed: %d events, %d entities, %d unique event types",
        len(df),
        n_entities,
        df[event_type_col].nunique(),
    )


# ── Statistics ─────────────────────────────────────────────────────────────────

def _print_stats(
    df: pd.DataFrame,
    splits: dict[str, list],
    preprocessor: EventPreprocessor,
    config: dict,
) -> None:
    data_cfg = config["data"]
    entity_col = data_cfg["group_col"]
    target_col = data_cfg["target_col"]
    event_type_col = data_cfg["event_type_col"]

    total_entities = sum(len(v) for v in splits.values())
    entity_targets = df.groupby(entity_col)[target_col].first()
    seq_lens = df.groupby(entity_col).size()
    vocab_size = len(preprocessor.vocab.get(event_type_col, {}))

    print()
    print("═" * 56)
    print("  Prepare-data Statistics")
    print("═" * 56)
    print(f"  Total events      : {len(df):>10,}")
    print(f"  Total entities    : {total_entities:>10,}")
    print()

    print("  Split sizes:")
    for name, ids in splits.items():
        pct = 100.0 * len(ids) / total_entities
        print(f"    {name:5s}  {len(ids):>6,} entities  ({pct:5.1f}%)")
    print()

    print(f"  Event type vocab  : {vocab_size:>6,} tokens  "
          f"(5 special + {vocab_size - 5} learned)")
    print()

    print("  Class balance (entity level):")
    class_counts = entity_targets.value_counts().sort_index()
    for cls, cnt in class_counts.items():
        pct = 100.0 * cnt / len(entity_targets)
        print(f"    class {cls}  {cnt:>6,}  ({pct:5.1f}%)")
    print()

    print("  Sequence lengths:")
    print(
        f"    min={seq_lens.min()},  "
        f"p50={seq_lens.median():.0f},  "
        f"mean={seq_lens.mean():.1f},  "
        f"max={seq_lens.max()}"
    )
    print("═" * 56)
    print()


def _print_dry_run_stats(df: pd.DataFrame, config: dict) -> None:
    data_cfg = config["data"]
    entity_col = data_cfg["group_col"]
    target_col = data_cfg["target_col"]
    event_type_col = data_cfg["event_type_col"]

    seq_lens = df.groupby(entity_col).size()
    entity_targets = df.groupby(entity_col)[target_col].first()
    class_counts = entity_targets.value_counts().sort_index()

    print()
    print("── Dry-run Statistics ────────────────────────────────")
    print(f"  Events       : {len(df):,}")
    print(f"  Entities     : {df[entity_col].nunique():,}")
    print(f"  Event types  : {df[event_type_col].nunique():,}")
    print(f"  Seq lengths  : min={seq_lens.min()}, "
          f"p50={seq_lens.median():.0f}, max={seq_lens.max()}")
    print(f"  Class counts : {dict(class_counts)}")
    print("──────────────────────────────────────────────────────")
    print()


# ── Synthetic data generator ──────────────────────────────────────────────────

def _make_synthetic_df(config: dict) -> pd.DataFrame:
    """Generate a minimal synthetic DataFrame for validation / dry-run."""
    import numpy as np
    rng = np.random.default_rng(42)
    data_cfg = config.get("data", {})
    entity_col = data_cfg.get("group_col", "entity_id")
    timestamp_col = data_cfg.get("timestamp_col", "timestamp")
    event_type_col = data_cfg.get("event_type_col", "event_type")
    target_col = data_cfg.get("target_col", "target")
    num_cols = list(data_cfg.get("numerical_cols") or [])
    cat_cols = list(data_cfg.get("categorical_cols") or [])

    n_entities = 30
    events_per = 20
    entity_ids = [f"synth_{i:04d}" for i in range(n_entities)]
    targets = rng.integers(0, 2, size=n_entities)
    base_time = pd.Timestamp("2023-01-01")
    rows = []
    for i, eid in enumerate(entity_ids):
        t = base_time
        for _ in range(events_per):
            t += pd.Timedelta(hours=int(rng.integers(1, 48)))
            row: dict = {
                entity_col: eid,
                timestamp_col: t,
                event_type_col: int(rng.integers(0, 10)),
                target_col: int(targets[i]),
            }
            for c in num_cols:
                row[c] = float(rng.normal(0, 1))
            for c in cat_cols:
                row[c] = int(rng.integers(0, 5))
            rows.append(row)
    df = pd.DataFrame(rows)
    logger.info("Generated synthetic DataFrame: %d rows, %d entities", len(df), n_entities)
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, split, and preprocess event sequence data for DME-Encoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to full YAML config (e.g. base.yaml or configs/pretrain.yaml)",
    )
    parser.add_argument(
        "--input", required=True, metavar="PATH",
        help="Path to raw data file (.csv, .tsv, or .parquet)",
    )
    parser.add_argument(
        "--output-dir", default="data/processed", metavar="DIR",
        help="Directory to write outputs (default: data/processed)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate data only — no files written",
    )
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    config = load_config(args.config)
    data_cfg = config["data"]
    seed = int((config.get("seed") or {}).get("global_seed", 42))

    entity_col: str = data_cfg["group_col"]
    timestamp_col: str = data_cfg["timestamp_col"]
    target_col: str = data_cfg["target_col"]

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if args.input == "synthetic":
        df = _make_synthetic_df(config)
    else:
        df = _load_data(args.input)

    # ── 2. Validate ───────────────────────────────────────────────────────────
    _validate(df, config)

    if args.dry_run:
        logger.info("--dry-run: validation passed — no output written")
        _print_dry_run_stats(df, config)
        return

    # ── 3. Sort ───────────────────────────────────────────────────────────────
    df = df.sort_values([entity_col, timestamp_col], kind="stable").reset_index(drop=True)
    logger.info("Sorted by (%s, %s)", entity_col, timestamp_col)

    # ── 4. Compute raw time_delta (seconds) ───────────────────────────────────
    # Unfitted preprocessor used only for the utility method; no scaler involved.
    _helper = EventPreprocessor(config)
    df["time_delta"] = _helper.compute_time_delta(df, entity_col, timestamp_col)
    logger.info("Computed time_delta: %.2f s median", float(df["time_delta"].median()))

    # ── 5. Split ──────────────────────────────────────────────────────────────
    splits = make_entity_splits(
        df,
        entity_col=entity_col,
        target_col=target_col,
        train_ratio=float(data_cfg.get("train_ratio", 0.70)),
        val_ratio=float(data_cfg.get("val_ratio", 0.15)),
        test_ratio=float(data_cfg.get("test_ratio", 0.15)),
        seed=seed,
        stratify=True,
    )

    # ── 6. Save splits ────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_path = output_dir / "splits.json"
    save_splits(splits, splits_path)

    # ── 7. Fit preprocessor on train split only ───────────────────────────────
    preprocessor = EventPreprocessor(config)
    preprocessor.fit(df, splits["train"])

    # ── 8. Save raw events ────────────────────────────────────────────────────
    # EventSequenceDataset applies the fitted preprocessor internally. Therefore
    # events.parquet must stay raw/pre-transform; otherwise training double-encodes
    # event_type/categorical ids and double-scales continuous features.
    events_path = output_dir / "events.parquet"
    df.to_parquet(events_path, index=False)
    size_mb = events_path.stat().st_size / 1e6
    logger.info("Saved raw events.parquet → %s  (%.1f MB)", events_path, size_mb)

    # ── 9. Save transformed diagnostic cache ──────────────────────────────────
    # time_delta column is recomputed internally and replaced with log1p-scaled version.
    transformed = preprocessor.transform(df)
    transformed_path = output_dir / "transformed_events.parquet"
    transformed.to_parquet(transformed_path, index=False)
    transformed_size_mb = transformed_path.stat().st_size / 1e6
    logger.info(
        "Saved transformed_events.parquet → %s  (%.1f MB)",
        transformed_path,
        transformed_size_mb,
    )

    # ── 10. Save preprocessor ─────────────────────────────────────────────────
    prep_path = output_dir / "preprocessor.pkl"
    with prep_path.open("wb") as f:
        pickle.dump(preprocessor, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved preprocessor.pkl → %s", prep_path)

    # ── 11. Statistics ────────────────────────────────────────────────────────
    _print_stats(df, splits, preprocessor, config)


if __name__ == "__main__":
    main()
