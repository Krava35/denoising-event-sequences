"""DME-Encoder evaluation script."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corruption.pipeline import CorruptionPipeline
from src.data.collate import collate_fn
from src.data.dataset import EventSequenceDataset
from src.data.preprocessing import EventPreprocessor
from src.evaluation import (
    compute_classification_metrics,
    compute_reconstruction_metrics,
    evaluate_robustness,
    plot_results,
    plot_robustness_curve,
)
from src.models.dme_encoder import DMEEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DME-Encoder evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to fine-tuned checkpoint (.pt)")
    p.add_argument(
        "--data", required=True,
        help="Directory containing test.parquet and preprocessor.json",
    )
    p.add_argument(
        "--config", default="configs/base.yaml",
        help="Base config path (merged over checkpoint config when provided)",
    )
    p.add_argument("--output-dir", default="outputs", help="Root output directory")
    p.add_argument("--classification", action="store_true", help="Run classification evaluation")
    p.add_argument("--reconstruction", action="store_true", help="Run reconstruction evaluation")
    p.add_argument("--robustness", action="store_true", help="Run robustness evaluation")
    return p.parse_args()


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def main() -> None:
    args = parse_args()

    if not any([args.classification, args.reconstruction, args.robustness]):
        logger.warning(
            "No evaluation mode selected. Use --classification, --reconstruction, "
            "or --robustness."
        )
        return

    # ── Output directories ────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_stem = Path(args.checkpoint).stem

    # ── Load checkpoint ───────────────────────────────────────────────────────
    logger.info("Loading checkpoint from %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config: dict = ckpt["config"]
    vocab_info: dict = ckpt["vocab_info"]
    logger.info(
        "Checkpoint: epoch=%d | val_metrics=%s",
        ckpt.get("epoch", -1),
        ckpt.get("val_metrics", {}),
    )

    # ── Build model ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = DMEEncoder(config, vocab_info)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    num_classes: int = model.classifier.classifier[-1].out_features
    logger.info("Model loaded | num_classes=%d", num_classes)

    # ── Load preprocessor ─────────────────────────────────────────────────────
    data_dir = Path(args.data)
    preprocessor = EventPreprocessor(config)
    preprocessor.load(str(data_dir / "preprocessor.json"))
    logger.info("Preprocessor loaded from %s", data_dir / "preprocessor.json")

    # ── Load test data ────────────────────────────────────────────────────────
    test_parquet = data_dir / "test.parquet"
    df_test = pd.read_parquet(test_parquet)
    entity_col = preprocessor.entity_col
    test_entity_ids = df_test[entity_col].unique().tolist()
    logger.info("Test entities: %d | rows: %d", len(test_entity_ids), len(df_test))

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    batch_size = int(config.get("training", {}).get("batch_size", 64))

    eval_dataset = EventSequenceDataset(
        df_test, test_entity_ids, preprocessor, config, mode="eval"
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    pretrain_dataset = EventSequenceDataset(
        df_test, test_entity_ids, preprocessor, config, mode="pretrain"
    )
    pretrain_loader = torch.utils.data.DataLoader(
        pretrain_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # ── Build CorruptionPipeline ──────────────────────────────────────────────
    corruption_cfg = config.get("corruption", {})
    vocab_sizes = {
        "event_type": vocab_info["event_type_vocab_size"],
        "cat_features": vocab_info.get("cat_vocab_sizes", []),
    }

    transition_matrix = None
    tm_cfg = corruption_cfg.get("transition_matrix", {})
    tm_path_npy = Path(tm_cfg.get("artifact_path", "data/processed/transition_matrix.npy"))
    tm_path_meta = Path(
        tm_cfg.get("metadata_path", "data/processed/transition_matrix_meta.json")
    )
    if tm_path_npy.exists() and tm_path_meta.exists():
        from src.corruption.transition_matrix import TransitionMatrix
        transition_matrix = TransitionMatrix.load(str(tm_path_npy), str(tm_path_meta))
        logger.info("Loaded transition matrix from %s", tm_path_npy)

    corruption_pipeline = CorruptionPipeline(
        config=corruption_cfg,
        transition_matrix=transition_matrix,
        vocab_sizes=vocab_sizes,
        time_transform=config.get("data", {}).get("time_transform", "log1p"),
    )

    # ── Classification evaluation ─────────────────────────────────────────────
    if args.classification:
        logger.info("Running classification evaluation...")
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                batch = _batch_to_device(batch, device)
                outputs = model(batch, mode="finetune")
                probs = torch.softmax(outputs["logits"], dim=-1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch["label"].cpu().numpy())

        probs_np = np.concatenate(all_probs, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        cls_metrics = compute_classification_metrics(labels_np, probs_np, num_classes)
        logger.info("Classification metrics: %s", cls_metrics)

        metrics_path = metrics_dir / f"{checkpoint_stem}_classification_metrics.json"
        with metrics_path.open("w") as f:
            json.dump(cls_metrics, f, indent=2)
        logger.info("Saved classification metrics to %s", metrics_path)

        plot_results(labels_np, probs_np, str(figures_dir), prefix=f"{checkpoint_stem}_")
        logger.info("Saved classification plots to %s", figures_dir)

    # ── Reconstruction evaluation ─────────────────────────────────────────────
    if args.reconstruction:
        logger.info("Running reconstruction evaluation...")
        accum: dict[str, float] = {}
        accum_counts: dict[str, int] = {}

        model.eval()
        with torch.no_grad():
            for clean_batch in pretrain_loader:
                clean_batch = _batch_to_device(clean_batch, device)
                corrupted_batch, targets, masks = corruption_pipeline(clean_batch)
                outputs = model(corrupted_batch, mode="pretrain")

                batch_metrics = compute_reconstruction_metrics(
                    outputs, targets, masks, preprocessor
                )
                for key, val in batch_metrics.items():
                    if not np.isnan(val):
                        accum[key] = accum.get(key, 0.0) + val
                        accum_counts[key] = accum_counts.get(key, 0) + 1

        recon_metrics: dict[str, float] = {
            k: accum[k] / accum_counts[k] for k in accum
        }
        # Fill any metric that had no non-NaN batches
        for k in [
            "event_type_accuracy", "event_type_macro_f1",
            "time_delta_mae_normalized", "time_delta_mae_original",
        ]:
            recon_metrics.setdefault(k, float("nan"))

        logger.info("Reconstruction metrics: %s", recon_metrics)

        metrics_path = metrics_dir / f"{checkpoint_stem}_reconstruction_metrics.json"
        with metrics_path.open("w") as f:
            json.dump(recon_metrics, f, indent=2)
        logger.info("Saved reconstruction metrics to %s", metrics_path)

    # ── Robustness evaluation ─────────────────────────────────────────────────
    if args.robustness:
        logger.info("Running robustness evaluation...")
        robustness_results = evaluate_robustness(
            model=model,
            test_loader=eval_loader,
            corruption_pipeline=corruption_pipeline,
            device=device,
            corruption_levels=[0.0, 0.1, 0.2, 0.3],
        )
        logger.info(
            "Robustness per level: %s",
            {
                lvl: m.get("roc_auc", m.get("roc_auc_ovr"))
                for lvl, m in robustness_results["per_level"].items()
            },
        )

        serializable = {
            "corruption_levels": robustness_results["corruption_levels"],
            "per_level": {
                str(lvl): metrics
                for lvl, metrics in robustness_results["per_level"].items()
            },
        }
        metrics_path = metrics_dir / f"{checkpoint_stem}_robustness_metrics.json"
        with metrics_path.open("w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Saved robustness metrics to %s", metrics_path)

        plot_robustness_curve(robustness_results, str(figures_dir), prefix=f"{checkpoint_stem}_")
        logger.info("Saved robustness curve to %s", figures_dir)


if __name__ == "__main__":
    main()
