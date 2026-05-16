"""Conditional event suffix generation with a diffusion-pretrained DME-Encoder.

Expected --data-dir layout:
  events.parquet
  splits.json
  preprocessor.pkl
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collate import collate_fn
from src.data.dataset import EventSequenceDataset
from src.data.forecasting import get_num_feature_dim
from src.data.splits import load_splits
from src.generation import (
    build_conditional_generation_batch,
    compute_generation_metrics,
    decode_generated_suffix,
    generate_suffix,
)
from src.models.dme_encoder import DMEEncoder
from src.utils.config import load_experiment_config, save_config
from src.utils.seed import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

_COUNT_KEYS = {"num_sequences", "suffix_positions", "target_positions"}
_SUFFIX_WEIGHTED_KEYS = {
    "invalid_event_token_rate",
    "duplicate_event_rate",
    "event_type_unique_ratio",
    "invalid_cat_token_rate",
}
_TARGET_WEIGHTED_KEYS = {
    "event_type_accuracy",
    "event_type_top5_recall",
    "time_delta_mae_normalized",
    "num_mae_normalized",
    "categorical_accuracy_mean",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate fixed-K event suffixes")
    p.add_argument("--config", default="configs/base.yaml", help="Base config path")
    p.add_argument("--dataset", default=None, help="Dataset config override")
    p.add_argument("--ablation", default=None, help="Ablation config override")
    p.add_argument("--checkpoint", required=True, help="Diffusion checkpoint path")
    p.add_argument("--data-dir", required=True, help="Directory with processed data artifacts")
    p.add_argument("--output-dir", default="outputs/generation", help="Output directory")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="Entity split")
    p.add_argument("--num-entities", type=int, default=64, help="Max entities to generate for")
    p.add_argument("--num-samples", type=int, default=1, help="Generated suffixes per entity")
    p.add_argument("--suffix-len", type=int, default=None, help="Override generation.suffix_len")
    p.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size")
    return p.parse_args()


def _load_preprocessor(data_dir: Path):
    pkl = data_dir / "preprocessor.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"preprocessor.pkl not found in {data_dir}")
    with pkl.open("rb") as f:
        return pickle.load(f)


def _build_vocab_info(preprocessor, config: dict) -> dict:
    return {
        "event_type_vocab_size": len(preprocessor.vocab.get(preprocessor.event_type_col, {})),
        "cat_vocab_sizes": [
            len(preprocessor.vocab.get(col, {})) for col in preprocessor.categorical_cols
        ],
        "num_num_features": get_num_feature_dim(preprocessor, config),
        "num_classes": int(config.get("training", {}).get("num_classes", 2)),
    }


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    return value


def _load_model(config: dict, vocab_info: dict, checkpoint_path: Path, device: torch.device) -> DMEEncoder:
    if config.get("pretraining", {}).get("objective") != "diffusion":
        raise ValueError("Event generation requires pretraining.objective='diffusion'")

    model = DMEEncoder(config, vocab_info)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _weighted_mean(rows: list[dict], key: str, weight_key: str) -> float | None:
    total = 0.0
    denom = 0.0
    for row in rows:
        value = row.get(key)
        weight = float(row.get(weight_key, 0))
        if _finite(value) and weight > 0:
            total += float(value) * weight
            denom += weight
    return total / denom if denom > 0 else None


def _aggregate_metrics(rows: list[dict]) -> dict:
    summary: dict[str, float | int | None] = {}
    if not rows:
        return summary

    for key in _COUNT_KEYS:
        summary[key] = int(sum(int(row.get(key, 0)) for row in rows))

    all_keys = sorted(set().union(*(row.keys() for row in rows)))
    for key in all_keys:
        if key in _COUNT_KEYS:
            continue
        if key in _SUFFIX_WEIGHTED_KEYS:
            summary[key] = _weighted_mean(rows, key, "suffix_positions")
        elif key in _TARGET_WEIGHTED_KEYS:
            summary[key] = _weighted_mean(rows, key, "target_positions")
        else:
            values = [float(row[key]) for row in rows if _finite(row.get(key))]
            summary[key] = float(np.mean(values)) if values else None
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, args.dataset, args.ablation)
    config.setdefault("generation", {})["enabled"] = True

    seed = int(config.get("seed", {}).get("global_seed", 42))
    set_seed(seed)
    device = get_device()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(output_dir / "generation_config.yaml"))

    preprocessor = _load_preprocessor(data_dir)
    splits = load_splits(data_dir / "splits.json")
    if args.split not in splits or not splits[args.split]:
        raise ValueError(f"Split '{args.split}' is empty or missing in splits.json")

    entity_ids = splits[args.split][: max(1, int(args.num_entities))]
    df_events = pd.read_parquet(data_dir / "events.parquet")
    vocab_info = _build_vocab_info(preprocessor, config)
    model = _load_model(config, vocab_info, Path(args.checkpoint), device)

    batch_size = int(args.batch_size or config.get("training", {}).get("batch_size", 128))
    dataset = EventSequenceDataset(df_events, entity_ids, preprocessor, config, mode="eval")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    decoded_rows: list[dict] = []
    metric_rows: list[dict] = []
    for batch in loader:
        batch = _move_to_device(batch, device)
        generation_seed = build_conditional_generation_batch(
            batch,
            config,
            suffix_len=args.suffix_len,
            num_samples=args.num_samples,
        )
        generated = generate_suffix(model, generation_seed, config, vocab_info)
        metric_rows.append(compute_generation_metrics(generated, vocab_info))
        decoded_rows.extend(decode_generated_suffix(generated, preprocessor, config))

    metrics = _aggregate_metrics(metric_rows)
    generated_df = pd.DataFrame(decoded_rows)
    parquet_path = output_dir / "generated_events.parquet"
    jsonl_path = output_dir / "generated_events.jsonl"
    if not generated_df.empty:
        generated_df.to_parquet(parquet_path, index=False)
        generated_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    else:
        jsonl_path.write_text("", encoding="utf-8")

    with (output_dir / "generation_metrics.json").open("w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    with (output_dir / "generation_examples.json").open("w") as f:
        json.dump(_json_safe(decoded_rows[:50]), f, indent=2, ensure_ascii=False)

    logger.info("Generated %d suffix events", len(decoded_rows))
    logger.info("Metrics saved to %s", output_dir / "generation_metrics.json")


if __name__ == "__main__":
    main()
