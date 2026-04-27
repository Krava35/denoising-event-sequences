from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    num_classes: int,
) -> dict:
    """Compute classification metrics from predicted probabilities.

    Args:
        y_true: [N] integer ground-truth labels.
        y_pred_proba: [N, num_classes] predicted class probabilities.
        num_classes: number of classes (2 → binary path, >2 → multiclass).

    Returns:
        dict of float-valued metrics.
    """
    y_pred = y_pred_proba.argmax(axis=1)

    if num_classes == 2:
        pos_proba = y_pred_proba[:, 1]
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, pos_proba)),
            "pr_auc": float(average_precision_score(y_true, pos_proba)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }
    else:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "roc_auc_ovr": float(
                roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro")
            ),
        }


def plot_results(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_dir: str | Path,
    prefix: str = "",
) -> None:
    """Save evaluation plots to output_dir.

    Always saves confusion_matrix.png.
    Binary: also saves roc_curve.png and pr_curve.png.
    Multiclass: also saves multiclass_roc.png.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = y_pred_proba.shape[1]
    y_pred = y_pred_proba.argmax(axis=1)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}confusion_matrix.png", dpi=150)
    plt.close(fig)

    if num_classes == 2:
        pos_proba = y_pred_proba[:, 1]

        # ── ROC curve ─────────────────────────────────────────────────────────
        fpr, tpr, _ = roc_curve(y_true, pos_proba)
        auc_val = roc_auc_score(y_true, pos_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}roc_curve.png", dpi=150)
        plt.close(fig)

        # ── PR curve ──────────────────────────────────────────────────────────
        precision, recall, _ = precision_recall_curve(y_true, pos_proba)
        ap = average_precision_score(y_true, pos_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f"AP = {ap:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}pr_curve.png", dpi=150)
        plt.close(fig)

    else:
        # ── Multiclass ROC (OvR, one curve per class) ─────────────────────────
        classes = list(range(num_classes))
        y_bin = label_binarize(y_true, classes=classes)

        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(num_classes):
            try:
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
                auc_i = roc_auc_score(y_bin[:, i], y_pred_proba[:, i])
                ax.plot(fpr_i, tpr_i, label=f"Class {i} (AUC={auc_i:.3f})")
            except ValueError:
                # Degenerate class with only one unique label in y_bin[:, i]
                pass
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Multiclass ROC (OvR)")
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}multiclass_roc.png", dpi=150)
        plt.close(fig)
