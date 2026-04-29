"""Run DME-Encoder ablation sweep (A0–A7).

A0 is a supervised baseline (no pretraining). A1–A7 perform denoising
pretraining followed by fine-tuning, each adding one more component.

Produces outputs/metrics/ablations.csv with one row per ablation.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corruption.pipeline import CorruptionPipeline
from src.corruption.transition_matrix import TransitionMatrix
from src.data.collate import collate_fn
from src.data.dataset import EventSequenceDataset
from src.data.splits import load_splits
from src.evaluation.classification import compute_classification_metrics
from src.models.dme_encoder import DMEEncoder
from src.training.finetune import finetune
from src.training.pretrain import pretrain
from src.utils.config import load_experiment_config, save_config
from src.utils.logging import MetricsLogger
from src.utils.seed import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Ordered ablation chain; A0 is supervised (no pretraining).
_ALL_ABLATIONS = [
    "configs/ablations/A0_supervised.yaml",
    "configs/ablations/A1_simple_masking.yaml",
    "configs/ablations/A2_mixed_low_rate.yaml",
    "configs/ablations/A2b_mixed_high_rate.yaml",
    "configs/ablations/A3_transition_aware.yaml",
    "configs/ablations/A4_event_level_masking.yaml",
    "configs/ablations/A5_gated_pooling.yaml",
    "configs/ablations/A6_hybrid_backbone.yaml",
    "configs/ablations/A7_full_dme.yaml",
]

# A0 is the supervised baseline: fine-tune from random init, no pretraining.
_NO_PRETRAIN_IDS = {"A0"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DME-Encoder ablation sweep")
    p.add_argument("--config", default="configs/base.yaml", help="Base config path")
    p.add_argument("--dataset", default=None, help="Dataset config (fixed for entire sweep)")
    p.add_argument("--config-dir", default="configs/ablations", help="Directory containing ablation YAMLs")
    p.add_argument("--data-dir", required=True, help="Processed data directory")
    p.add_argument("--output-dir", default="outputs", help="Root output directory")
    p.add_argument(
        "--ablation-ids", default="all",
        help="Comma-separated ablation IDs (e.g. 'A0,A1,A2') or 'all'",
    )
    return p.parse_args()


def _resolve_ablation_paths(config_dir: str, ablation_ids: str) -> list[str]:
    if ablation_ids.strip().lower() == "all":
        return _ALL_ABLATIONS
    ids = [x.strip() for x in ablation_ids.split(",") if x.strip()]
    resolved = []
    for aid in ids:
        # Match by prefix: "A3" matches "A3_transition_aware.yaml"
        candidates = list(Path(config_dir).glob(f"{aid}_*.yaml"))
        if not candidates:
            # Try exact match
            candidates = list(Path(config_dir).glob(f"{aid}.yaml"))
        if candidates:
            resolved.append(str(sorted(candidates)[0]))
        else:
            logger.warning("No ablation config found for ID '%s' in %s", aid, config_dir)
    return resolved


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


def _load_transition_matrix(data_dir: Path, config: dict):
    tm_cfg = config.get("corruption", {}).get("transition_matrix", {})
    npy = Path(tm_cfg.get("artifact_path", str(data_dir / "transition_matrix.npy")))
    meta = Path(tm_cfg.get("metadata_path", str(data_dir / "transition_matrix_meta.json")))
    if not npy.exists():
        npy = data_dir / "transition_matrix.npy"
        meta = data_dir / "transition_matrix_meta.json"
    if npy.exists() and meta.exists():
        return TransitionMatrix.load(str(npy), str(meta))
    return None


def _build_loaders(df_events, splits, preprocessor, config, batch_size, mode_train):
    train_loader = DataLoader(
        EventSequenceDataset(df_events, splits["train"], preprocessor, config, mode=mode_train),
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
    return train_loader, val_loader, test_loader


def _evaluate_on_test(model, test_loader, num_classes, device) -> dict:
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


def _run_one_ablation(
    ablation_cfg_path: str,
    base_config_path: str,
    dataset_path: str | None,
    data_dir: Path,
    output_dir: Path,
    df_events: pd.DataFrame,
    splits: dict,
    preprocessor,
    transition_matrix,
    device: torch.device,
) -> dict:
    """Run pretrain (if applicable) + finetune + test evaluation for one ablation."""
    ablation_id = Path(ablation_cfg_path).stem  # e.g. "A3_transition_aware"
    short_id = ablation_id.split("_")[0]         # e.g. "A3"

    config = load_experiment_config(
        base_path=base_config_path,
        dataset_path=dataset_path,
        ablation_path=ablation_cfg_path,
    )
    seed = config.get("seed", {}).get("global_seed", 42)
    set_seed(seed)

    vocab_info = _build_vocab_info(preprocessor, config)
    num_classes = vocab_info["num_classes"]
    batch_size = int(config.get("training", {}).get("batch_size", 128))

    abl_out = output_dir / ablation_id
    checkpoints_dir = abl_out / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(checkpoints_dir / "config.yaml"))

    ml_label = f"{ablation_id}"
    ml = MetricsLogger(str(abl_out / "logs"), ml_label)

    pretrained_ckpt = None

    # ── Pretraining (A1–A7 only) ───────────────────────────────────────────
    if short_id not in _NO_PRETRAIN_IDS:
        corruption_pipeline = CorruptionPipeline(
            config=config.get("corruption", {}),
            transition_matrix=transition_matrix,
            vocab_sizes={
                "event_type": vocab_info["event_type_vocab_size"],
                "cat_features": vocab_info["cat_vocab_sizes"],
            },
            time_transform=config.get("data", {}).get("time_transform", "log1p"),
        )
        model = DMEEncoder(config, vocab_info)
        pretrain_train, pretrain_val, _ = _build_loaders(
            df_events, splits, preprocessor, config, batch_size, mode_train="pretrain"
        )
        pretrained_ckpt = pretrain(
            model=model,
            train_loader=pretrain_train,
            val_loader=pretrain_val,
            corruption_pipeline=corruption_pipeline,
            config=config,
            output_dir=str(checkpoints_dir),
            device=device,
            logger=ml,
            vocab_info=vocab_info,
        )
        logger.info("[%s] Pretrain done → %s", ablation_id, pretrained_ckpt)
    else:
        logger.info("[%s] Supervised baseline — skipping pretraining", ablation_id)

    # ── Fine-tuning ────────────────────────────────────────────────────────
    model = DMEEncoder(config, vocab_info)
    ft_train, ft_val, ft_test = _build_loaders(
        df_events, splits, preprocessor, config, batch_size, mode_train="finetune"
    )
    best_ckpt = finetune(
        model=model,
        train_loader=ft_train,
        val_loader=ft_val,
        config=config,
        output_dir=str(checkpoints_dir),
        device=device,
        logger=ml,
        pretrained_checkpoint=pretrained_ckpt,
        frozen_encoder=False,
        vocab_info=vocab_info,
    )
    logger.info("[%s] Finetune done → %s", ablation_id, best_ckpt)

    # ── Test evaluation ────────────────────────────────────────────────────
    test_metrics: dict = {}
    if ft_test is not None:
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        test_metrics = _evaluate_on_test(model, ft_test, num_classes, device)
        logger.info("[%s] Test metrics: %s", ablation_id, test_metrics)

    result = {"ablation_id": ablation_id, **test_metrics}
    with (abl_out / "test_metrics.json").open("w") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # Load shared data artifacts once
    preprocessor = _load_preprocessor(data_dir)
    splits = load_splits(data_dir / "splits.json")
    df_events = pd.read_parquet(data_dir / "events.parquet")
    logger.info("Data loaded | %d rows | %d train / %d val / %d test entities",
                len(df_events), len(splits["train"]), len(splits["val"]),
                len(splits.get("test", [])))

    # Load a reference config just for transition matrix path
    ref_config = load_experiment_config(base_path=args.config, dataset_path=args.dataset)
    transition_matrix = _load_transition_matrix(data_dir, ref_config)

    ablation_paths = _resolve_ablation_paths(args.config_dir, args.ablation_ids)
    logger.info("Running %d ablations: %s", len(ablation_paths),
                [Path(p).stem for p in ablation_paths])

    all_results: list[dict] = []

    for abl_path in ablation_paths:
        if not Path(abl_path).exists():
            logger.warning("Ablation config not found, skipping: %s", abl_path)
            continue
        try:
            result = _run_one_ablation(
                ablation_cfg_path=abl_path,
                base_config_path=args.config,
                dataset_path=args.dataset,
                data_dir=data_dir,
                output_dir=output_dir,
                df_events=df_events,
                splits=splits,
                preprocessor=preprocessor,
                transition_matrix=transition_matrix,
                device=device,
            )
            all_results.append(result)
        except Exception as exc:
            logger.error("Ablation %s failed: %s", Path(abl_path).stem, exc, exc_info=True)
            all_results.append({"ablation_id": Path(abl_path).stem, "error": str(exc)})

    # ── Save ablations.csv ────────────────────────────────────────────────────
    if all_results:
        fieldnames = list(all_results[0].keys())
        for row in all_results[1:]:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)

        csv_path = metrics_dir / "ablations.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        logger.info("Ablation results saved to %s", csv_path)

        # Pretty-print summary
        logger.info("\n=== Ablation Summary ===")
        for row in all_results:
            auc = row.get("roc_auc", row.get("roc_auc_ovr", "—"))
            f1 = row.get("macro_f1", "—")
            logger.info("  %-35s roc_auc=%.4f  macro_f1=%.4f",
                        row["ablation_id"],
                        auc if isinstance(auc, float) else float("nan"),
                        f1 if isinstance(f1, float) else float("nan"))


if __name__ == "__main__":
    main()
