"""Low-label fine-tuning experiments.

For each label_fraction × label_sampling_seed combination, fine-tunes
the pretrained encoder on a stratified subset of train entities and
evaluates on the full test set.

Produces outputs/metrics/low_label.csv with individual runs and
aggregate mean ± std per fraction.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collate import collate_fn
from src.data.dataset import EventSequenceDataset
from src.data.forecasting import get_num_feature_dim
from src.data.splits import load_splits
from src.evaluation.classification import compute_classification_metrics
from src.models.dme_encoder import DMEEncoder
from src.training.finetune import finetune
from src.utils.config import load_experiment_config, save_config
from src.utils.logging import MetricsLogger
from src.utils.seed import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DME-Encoder low-label experiments")
    p.add_argument("--config", default="configs/base.yaml", help="Base config path")
    p.add_argument("--dataset", default=None, help="Dataset config override")
    p.add_argument("--pretrained-checkpoint", required=True, help="Pretrained encoder checkpoint")
    p.add_argument("--data-dir", required=True, help="Processed data directory")
    p.add_argument("--output-dir", default="outputs", help="Root output directory")
    return p.parse_args()


def _load_preprocessor(data_dir: Path):
    pkl = data_dir / "preprocessor.pkl"
    if pkl.exists():
        with pkl.open("rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"preprocessor.pkl not found in {data_dir}")


def _build_vocab_info(preprocessor, config: dict) -> dict:
    return {
        "event_type_vocab_size": len(preprocessor.vocab.get(preprocessor.event_type_col, {})),
        "cat_vocab_sizes": [
            len(preprocessor.vocab.get(col, {})) for col in preprocessor.categorical_cols
        ],
        "num_num_features": get_num_feature_dim(preprocessor, config),
        "num_classes": int(config.get("training", {}).get("num_classes", 2)),
    }


def _subsample_train_ids(train_ids: list, fraction: float, seed: int) -> list:
    """Return a stratified (by entity) random subset of train_ids."""
    if fraction >= 1.0:
        return train_ids
    n = max(1, round(len(train_ids) * fraction))
    rng = random.Random(seed)
    return rng.sample(train_ids, n)


def _evaluate_on_test(model, test_loader, num_classes: int, device) -> dict:
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            probs = torch.softmax(model(batch, mode="finetune")["logits"], dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
    probs_np = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    return compute_classification_metrics(labels_np, probs_np, num_classes)


def main() -> None:
    args = parse_args()

    config = load_experiment_config(
        base_path=args.config,
        dataset_path=args.dataset,
    )

    base_seed = config.get("seed", {}).get("global_seed", 42)
    set_seed(base_seed)
    device = get_device()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shared data artifacts ────────────────────────────────────────────
    preprocessor = _load_preprocessor(data_dir)
    splits = load_splits(data_dir / "splits.json")
    df_events = pd.read_parquet(data_dir / "events.parquet")
    logger.info("Data loaded | %d rows | train=%d val=%d test=%d",
                len(df_events), len(splits["train"]), len(splits["val"]),
                len(splits.get("test", [])))

    vocab_info = _build_vocab_info(preprocessor, config)
    num_classes = vocab_info["num_classes"]
    batch_size = int(config.get("training", {}).get("batch_size", 128))
    save_config(config, str(output_dir / "low_label_config.yaml"))

    # ── Read protocol from config ─────────────────────────────────────────────
    proto = config.get("low_label_protocol", {})
    fractions: list[float] = proto.get("label_fractions", [0.01, 0.05, 0.10, 0.25, 0.50, 1.00])
    seeds: list[int] = proto.get("label_sampling_seeds", [42, 43, 44])
    logger.info("Protocol: fractions=%s | seeds=%s", fractions, seeds)

    # Shared val / test loaders (always use full val/test)
    val_loader = DataLoader(
        EventSequenceDataset(df_events, splits["val"], preprocessor, config, mode="eval"),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        EventSequenceDataset(df_events, splits.get("test", []), preprocessor, config, mode="eval"),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    ) if splits.get("test") else None

    all_rows: list[dict] = []

    for fraction in fractions:
        for seed in seeds:
            set_seed(seed)
            run_id = f"frac{fraction:.2f}_seed{seed}"
            logger.info("=== %s ===", run_id)

            # Subsample train entities
            train_ids = _subsample_train_ids(splits["train"], fraction, seed)
            logger.info("  Train entities: %d / %d (fraction=%.2f, seed=%d)",
                        len(train_ids), len(splits["train"]), fraction, seed)

            train_loader = DataLoader(
                EventSequenceDataset(df_events, train_ids, preprocessor, config, mode="finetune"),
                batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
            )

            # Fresh model for each run
            model = DMEEncoder(config, vocab_info)
            run_dir = output_dir / "low_label" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            ml = MetricsLogger(str(run_dir / "logs"), run_id)

            best_ckpt = finetune(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                output_dir=str(run_dir / "checkpoints"),
                device=device,
                logger=ml,
                pretrained_checkpoint=args.pretrained_checkpoint,
                frozen_encoder=False,
                vocab_info=vocab_info,
            )

            # Evaluate on test set
            test_metrics: dict = {}
            if test_loader is not None:
                ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(device)
                test_metrics = _evaluate_on_test(model, test_loader, num_classes, device)
                logger.info("  Test metrics: %s", test_metrics)

            row = {"fraction": fraction, "seed": seed, **test_metrics}
            all_rows.append(row)

            with (run_dir / "test_metrics.json").open("w") as f:
                json.dump(row, f, indent=2)

    # ── Aggregate mean ± std per fraction ────────────────────────────────────
    metric_keys = [k for k in (all_rows[0].keys() if all_rows else []) if k not in ("fraction", "seed")]
    agg_rows: list[dict] = []

    for fraction in fractions:
        runs = [r for r in all_rows if r["fraction"] == fraction]
        agg: dict = {"fraction": fraction, "n_seeds": len(runs)}
        for mk in metric_keys:
            vals = [r[mk] for r in runs if isinstance(r.get(mk), float)]
            if vals:
                agg[f"{mk}_mean"] = float(np.mean(vals))
                agg[f"{mk}_std"] = float(np.std(vals))
        agg_rows.append(agg)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    if all_rows:
        run_csv = metrics_dir / "low_label.csv"
        fieldnames_run = ["fraction", "seed"] + metric_keys
        with run_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_run, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info("Per-run results saved to %s", run_csv)

    if agg_rows:
        agg_csv = metrics_dir / "low_label_agg.csv"
        agg_fields = list(agg_rows[0].keys())
        with agg_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=agg_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(agg_rows)
        logger.info("Aggregate results saved to %s", agg_csv)

        # Print summary table
        logger.info("\n=== Low-Label Summary (mean ± std) ===")
        primary = "roc_auc_mean" if "roc_auc_mean" in agg_rows[0] else (
            "roc_auc_ovr_mean" if "roc_auc_ovr_mean" in agg_rows[0] else None
        )
        f1_mean = "macro_f1_mean" if "macro_f1_mean" in agg_rows[0] else None
        for row in agg_rows:
            auc_str = (f"{row[primary]:.4f}±{row.get(primary.replace('_mean','_std'),0):.4f}"
                       if primary else "—")
            f1_str = (f"{row[f1_mean]:.4f}±{row.get(f1_mean.replace('_mean','_std'),0):.4f}"
                      if f1_mean else "—")
            logger.info("  fraction=%.2f  roc_auc=%s  macro_f1=%s",
                        row["fraction"], auc_str, f1_str)


if __name__ == "__main__":
    main()
