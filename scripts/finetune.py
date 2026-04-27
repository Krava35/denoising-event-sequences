"""DME-Encoder fine-tuning on downstream classification task.

Expected --data-dir layout: same as pretrain.py (events.parquet, splits.json,
preprocessor.pkl, optional transition_matrix files).
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

from src.data.collate import collate_fn
from src.data.dataset import EventSequenceDataset
from src.data.splits import load_splits
from src.evaluation.classification import compute_classification_metrics, plot_results
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
    p = argparse.ArgumentParser(description="DME-Encoder fine-tuning")
    p.add_argument("--config", default="configs/base.yaml", help="Base config path")
    p.add_argument("--dataset", default=None, help="Dataset config override")
    p.add_argument("--ablation", default=None, help="Ablation config override")
    p.add_argument("--pretrained-checkpoint", default=None, help="Pretrained encoder checkpoint")
    p.add_argument("--data-dir", required=True, help="Directory with events.parquet, splits.json, preprocessor.pkl")
    p.add_argument("--output-dir", default="outputs", help="Root output directory")
    p.add_argument("--frozen-encoder", action="store_true", help="Freeze encoder during fine-tuning")
    p.add_argument("--label-fraction", type=float, default=1.0, help="Fraction of training data to use")
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
        "num_num_features": len(preprocessor.numerical_cols),
        "num_classes": int(config.get("training", {}).get("num_classes", 2)),
    }



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
    save_config(config, str(checkpoints_dir / "finetune_config.yaml"))

    # ── Load data artifacts ───────────────────────────────────────────────────
    preprocessor = _load_preprocessor(data_dir)
    splits = load_splits(data_dir / "splits.json")
    df_events = pd.read_parquet(data_dir / "events.parquet")
    logger.info(
        "Data loaded | train=%d val=%d test=%d entities",
        len(splits["train"]), len(splits["val"]), len(splits.get("test", [])),
    )

    vocab_info = _build_vocab_info(preprocessor, config)
    num_classes: int = vocab_info["num_classes"]
    batch_size = int(config.get("training", {}).get("batch_size", 128))

    # ── Datasets and loaders ──────────────────────────────────────────────────
    train_loader = DataLoader(
        EventSequenceDataset(df_events, splits["train"], preprocessor, config, mode="finetune"),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        EventSequenceDataset(df_events, splits["val"], preprocessor, config, mode="eval"),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        EventSequenceDataset(df_events, splits.get("test", []), preprocessor, config, mode="eval"),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    ) if splits.get("test") else None

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DMEEncoder(config, vocab_info)

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    exp_name = Path(args.config).stem
    if args.dataset:
        exp_name = f"{exp_name}_{Path(args.dataset).stem}"
    ml = MetricsLogger(str(output_dir / "logs"), f"{exp_name}_finetune")
    ml.log_config(config)

    best_ckpt = finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=str(checkpoints_dir),
        device=device,
        logger=ml,
        pretrained_checkpoint=args.pretrained_checkpoint,
        frozen_encoder=args.frozen_encoder,
        label_fraction=args.label_fraction,
        vocab_info=vocab_info,
    )
    logger.info("Best finetune checkpoint: %s", best_ckpt)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if test_loader is not None:
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        import numpy as np
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
        test_metrics = compute_classification_metrics(labels_np, probs_np, num_classes)
        logger.info("Test metrics: %s", test_metrics)

        plot_results(labels_np, probs_np, str(figures_dir), prefix=f"{exp_name}_")

        summary = {"best_checkpoint": str(best_ckpt), "test_metrics": test_metrics}
        with (metrics_dir / f"{exp_name}_finetune_metrics.json").open("w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Finetune metrics saved to %s/metrics/", output_dir)
    else:
        with (metrics_dir / f"{exp_name}_finetune_metrics.json").open("w") as f:
            json.dump({"best_checkpoint": str(best_ckpt)}, f, indent=2)


if __name__ == "__main__":
    main()
