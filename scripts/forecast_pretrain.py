"""Hybrid denoising + forecast pretraining for DME-Encoder."""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
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
from src.data.forecasting import (
    build_forecast_stats,
    get_num_feature_dim,
    load_forecast_stats,
    save_forecast_stats,
    scenario_from_outputs,
    scenario_from_targets,
)
from src.data.splits import load_splits
from src.models.dme_encoder import DMEEncoder
from src.training.forecast_pretrain import evaluate_forecast_quality, forecast_pretrain
from src.utils.config import load_experiment_config, save_config
from src.utils.logging import MetricsLogger
from src.utils.seed import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DME-Encoder forecast pretraining")
    p.add_argument("--config", default="configs/base.yaml", help="Base config path")
    p.add_argument("--dataset", default=None, help="Dataset config override")
    p.add_argument("--ablation", default=None, help="Ablation config override")
    p.add_argument("--data-dir", required=True, help="Directory with events.parquet, splits.json, preprocessor.pkl")
    p.add_argument("--output-dir", default="outputs", help="Root output directory")
    p.add_argument("--resume", default=None, help="Checkpoint path to restore model weights from")
    p.add_argument("--forecast-stats", default=None, help="Optional existing forecast_stats.json")
    return p.parse_args()


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
        "num_num_features": get_num_feature_dim(preprocessor, config),
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


def _batch_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _batch_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_batch_to_device(v, device) for v in value]
    return value


def _slice_outputs(outputs: dict, index: int) -> dict:
    sliced = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            sliced[key] = value[index : index + 1]
        elif isinstance(value, list):
            sliced[key] = [item[index : index + 1] for item in value]
        else:
            sliced[key] = value
    return sliced


def save_scenario_examples(
    model: DMEEncoder,
    val_loader: DataLoader,
    forecast_stats: dict,
    vocab: dict,
    output_path: Path,
    device: torch.device,
    *,
    limit: int = 20,
    top_k: int = 5,
) -> None:
    model.eval()
    examples: list[dict] = []
    with torch.no_grad():
        for batch in val_loader:
            batch_on_device = _batch_to_device(batch, device)
            outputs = model(batch_on_device, mode="forecast")
            for i, entity_id in enumerate(batch["entity_id"]):
                examples.append(
                    {
                        "entity_id": entity_id,
                        "predicted_scenario": scenario_from_outputs(
                            _slice_outputs(outputs, i),
                            forecast_stats,
                            vocab,
                            top_k=top_k,
                        ),
                        "true_future_summary": scenario_from_targets(
                            batch_on_device["forecast_targets"],
                            forecast_stats,
                            vocab,
                            index=i,
                            top_k=top_k,
                        ),
                    }
                )
                if len(examples) >= limit:
                    break
            if len(examples) >= limit:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    logger.info("Scenario examples saved to %s", output_path)


def main() -> None:
    args = parse_args()

    config = load_experiment_config(
        base_path=args.config,
        dataset_path=args.dataset,
        ablation_path=args.ablation,
    )
    config = {
        **config,
        "forecasting": {**config.get("forecasting", {}), "enabled": True},
    }

    seed = config.get("seed", {}).get("global_seed", 42)
    set_seed(seed)
    device = get_device()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(checkpoints_dir / "forecast_pretrain_config.yaml"))

    preprocessor = load_preprocessor(data_dir)
    splits = load_splits(data_dir / "splits.json")
    df_events = pd.read_parquet(data_dir / "events.parquet")
    logger.info(
        "Data loaded | train=%d val=%d test=%d entities | rows=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits.get("test", [])),
        len(df_events),
    )

    if args.forecast_stats:
        forecast_stats = load_forecast_stats(args.forecast_stats)
        stats_path = Path(args.forecast_stats)
    else:
        forecast_stats = build_forecast_stats(df_events, splits["train"], preprocessor, config)
        stats_path = checkpoints_dir / "forecast_stats.json"
        save_forecast_stats(forecast_stats, stats_path)
    logger.info("Forecast stats: %s", stats_path)

    vocab_info = build_vocab_info(preprocessor, config)
    batch_size = int(config.get("training", {}).get("batch_size", 128))

    train_loader = DataLoader(
        EventSequenceDataset(
            df_events,
            splits["train"],
            preprocessor,
            config,
            mode="forecast",
            forecast_stats=forecast_stats,
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        EventSequenceDataset(
            df_events,
            splits["val"],
            preprocessor,
            config,
            mode="forecast",
            forecast_stats=forecast_stats,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    logger.info("DataLoaders: %d train batches | %d val batches", len(train_loader), len(val_loader))

    model = DMEEncoder(config, vocab_info)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Restored model weights from %s (epoch=%d)", args.resume, ckpt.get("epoch", -1))

    transition_matrix = load_transition_matrix(data_dir, config)
    corruption_pipeline = build_corruption_pipeline(config, vocab_info, transition_matrix)

    exp_name = Path(args.config).stem
    if args.dataset:
        exp_name = f"{exp_name}_{Path(args.dataset).stem}"
    ml = MetricsLogger(str(output_dir / "logs"), f"{exp_name}_forecast_pretrain")
    ml.log_config(config)

    best_ckpt = forecast_pretrain(
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
    logger.info("Best forecast pretrain checkpoint: %s", best_ckpt)

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_jsonl_path = metrics_dir / "forecast_pretrain_metrics.jsonl"
    if ml.metrics_path.exists():
        shutil.copyfile(ml.metrics_path, metrics_jsonl_path)
        logger.info("Forecast pretrain metrics copied to %s", metrics_jsonl_path)

    best_state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    forecast_eval_metrics = evaluate_forecast_quality(
        model=model,
        val_loader=val_loader,
        config=config,
        device=device,
        forecast_stats=forecast_stats,
    )
    forecast_eval_path = metrics_dir / "forecast_eval_metrics.json"
    with forecast_eval_path.open("w") as f:
        json.dump(forecast_eval_metrics, f, indent=2)
    logger.info("Forecast eval metrics saved to %s", forecast_eval_path)

    scenario_examples_path = metrics_dir / "scenario_examples.json"
    save_scenario_examples(
        model=model,
        val_loader=val_loader,
        forecast_stats=forecast_stats,
        vocab=preprocessor.vocab,
        output_path=scenario_examples_path,
        device=device,
        limit=20,
        top_k=int(config.get("forecasting", {}).get("scenario_top_k", 5)),
    )

    summary = {
        "best_checkpoint": str(best_ckpt),
        "forecast_stats": str(stats_path),
        "forecast_pretrain_metrics": str(metrics_jsonl_path),
        "forecast_eval_metrics": str(forecast_eval_path),
        "scenario_examples": str(scenario_examples_path),
        "vocab_info": vocab_info,
    }
    summary_path = metrics_dir / f"{exp_name}_forecast_pretrain_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Forecast pretrain summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
