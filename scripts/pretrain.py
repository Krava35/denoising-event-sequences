"""DME-Encoder denoising pretraining.

Expected --data-dir layout:
  events.parquet            — raw (pre-transform) event data fed to EventSequenceDataset
  splits.json               — {"train": [...], "val": [...], "test": [...]}
  preprocessor.pkl          — pickled EventPreprocessor fitted on train split
  transition_matrix.npy     — optional, loaded when use_transition_aware_replacement=true
  transition_matrix_meta.json
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corruption.pipeline import CorruptionPipeline
from src.corruption.transition_matrix import TransitionMatrix
from src.data.collate import collate_fn
from src.data.dataset import EventSequenceDataset
from src.data.splits import load_splits
from src.models.dme_encoder import DMEEncoder
from src.training.pretrain import pretrain
from src.utils.config import load_experiment_config, save_config
from src.utils.logging import MetricsLogger
from src.utils.seed import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DME-Encoder pretraining")
    p.add_argument("--config", default="configs/base.yaml", help="Base config path")
    p.add_argument("--dataset", default=None, help="Dataset config override")
    p.add_argument("--ablation", default=None, help="Ablation config override")
    p.add_argument("--data-dir", required=True, help="Directory with events.parquet, splits.json, preprocessor.pkl")
    p.add_argument("--output-dir", default="outputs", help="Root output directory")
    p.add_argument("--resume", default=None, help="Checkpoint path to restore model weights from")
    return p.parse_args()


# ── Shared helpers ────────────────────────────────────────────────────────────

def load_preprocessor(data_dir: Path):
    pkl = data_dir / "preprocessor.pkl"
    if pkl.exists():
        with pkl.open("rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"preprocessor.pkl not found in {data_dir}")


def build_vocab_info(preprocessor, config: dict) -> dict:
    return {
        "event_type_vocab_size": len(preprocessor.vocab.get(preprocessor.event_type_col, {})),
        "cat_vocab_sizes": [
            len(preprocessor.vocab.get(col, {})) for col in preprocessor.categorical_cols
        ],
        "num_num_features": len(preprocessor.numerical_cols),
        "num_classes": int(config.get("training", {}).get("num_classes", 2)),
    }


def load_transition_matrix(data_dir: Path, config: dict):
    tm_cfg = config.get("corruption", {}).get("transition_matrix", {})
    npy = Path(tm_cfg.get("artifact_path", str(data_dir / "transition_matrix.npy")))
    meta = Path(tm_cfg.get("metadata_path", str(data_dir / "transition_matrix_meta.json")))
    if not npy.exists():
        npy = data_dir / "transition_matrix.npy"
        meta = data_dir / "transition_matrix_meta.json"
    if npy.exists() and meta.exists():
        tm = TransitionMatrix.load(str(npy), str(meta))
        logger.info("Loaded transition matrix from %s", npy)
        return tm
    return None


def build_corruption_pipeline(config: dict, vocab_info: dict, transition_matrix) -> CorruptionPipeline:
    return CorruptionPipeline(
        config=config.get("corruption", {}),
        transition_matrix=transition_matrix,
        vocab_sizes={
            "event_type": vocab_info["event_type_vocab_size"],
            "cat_features": vocab_info["cat_vocab_sizes"],
        },
        time_transform=config.get("data", {}).get("time_transform", "log1p"),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    config = load_experiment_config(
        base_path=args.config,
        dataset_path=args.dataset,
        ablation_path=args.ablation,
    )

    seed = config.get("seed", {}).get("global_seed", 42)
    set_seed(seed)
    device = get_device()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(checkpoints_dir / "pretrain_config.yaml"))

    # ── Load data artifacts ───────────────────────────────────────────────────
    preprocessor = load_preprocessor(data_dir)
    splits = load_splits(data_dir / "splits.json")
    df_events = pd.read_parquet(data_dir / "events.parquet")
    logger.info(
        "Data loaded | train=%d val=%d test=%d entities | rows=%d",
        len(splits["train"]), len(splits["val"]), len(splits.get("test", [])),
        len(df_events),
    )

    vocab_info = build_vocab_info(preprocessor, config)
    logger.info("Vocab info: %s", {k: v for k, v in vocab_info.items() if k != "cat_vocab_sizes"})

    # ── Datasets and loaders ──────────────────────────────────────────────────
    batch_size = int(config.get("training", {}).get("batch_size", 128))

    train_loader = DataLoader(
        EventSequenceDataset(df_events, splits["train"], preprocessor, config, mode="pretrain"),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        EventSequenceDataset(df_events, splits["val"], preprocessor, config, mode="pretrain"),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    logger.info("DataLoaders: %d train batches | %d val batches", len(train_loader), len(val_loader))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DMEEncoder(config, vocab_info)
    param_counts = model.count_parameters()
    logger.info("Model parameters: total=%d", param_counts["total"])

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Restored model weights from %s (epoch=%d)", args.resume, ckpt.get("epoch", -1))

    # ── Corruption pipeline ───────────────────────────────────────────────────
    transition_matrix = load_transition_matrix(data_dir, config)
    corruption_pipeline = build_corruption_pipeline(config, vocab_info, transition_matrix)

    # ── Train ─────────────────────────────────────────────────────────────────
    exp_name = Path(args.config).stem
    if args.dataset:
        exp_name = f"{exp_name}_{Path(args.dataset).stem}"
    ml = MetricsLogger(str(output_dir / "logs"), f"{exp_name}_pretrain")
    ml.log_config(config)

    best_ckpt = pretrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        corruption_pipeline=corruption_pipeline,
        config=config,
        output_dir=str(checkpoints_dir),
        device=device,
        logger=ml,
        vocab_info=vocab_info,
    )
    logger.info("Best pretrain checkpoint: %s", best_ckpt)

    # ── Save final metrics summary ────────────────────────────────────────────
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = {"best_checkpoint": str(best_ckpt), "vocab_info": vocab_info}
    summary_path = metrics_dir / f"{exp_name}_pretrain_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Pretrain summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
