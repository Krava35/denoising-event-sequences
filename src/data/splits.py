import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def make_entity_splits(
    df: pd.DataFrame,
    entity_col: str,
    target_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> dict[str, list]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    entity_targets = df.groupby(entity_col)[target_col].first()
    entities = entity_targets.index.tolist()
    targets = entity_targets.values.tolist()
    n_total = len(entities)

    strat_labels = targets if stratify else None

    # Check if stratification is feasible (each class needs >= 2 samples per fold)
    if stratify:
        from collections import Counter

        counts = Counter(targets)
        if min(counts.values()) < 2:
            logger.warning(
                "Some classes have < 2 entities; falling back to non-stratified split"
            )
            strat_labels = None

    trainval, test_ids, trainval_labels, _ = train_test_split(
        entities,
        targets,
        test_size=test_ratio,
        random_state=seed,
        stratify=strat_labels,
    )

    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    strat_trainval = trainval_labels if (strat_labels is not None) else None

    if strat_trainval is not None:
        from collections import Counter

        counts = Counter(trainval_labels)
        if min(counts.values()) < 2:
            logger.warning(
                "Some classes have < 2 entities in train+val; "
                "falling back to non-stratified val split"
            )
            strat_trainval = None

    train_ids, val_ids = train_test_split(
        trainval,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=strat_trainval,
    )

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert train_set.isdisjoint(val_set), "Train/val overlap detected"
    assert train_set.isdisjoint(test_set), "Train/test overlap detected"
    assert val_set.isdisjoint(test_set), "Val/test overlap detected"
    assert len(train_ids) + len(val_ids) + len(test_ids) == n_total, (
        "Not all entities covered"
    )

    logger.info(
        "Splits: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_ids),
        100 * len(train_ids) / n_total,
        len(val_ids),
        100 * len(val_ids) / n_total,
        len(test_ids),
        100 * len(test_ids) / n_total,
    )

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def save_splits(splits: dict[str, list], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(splits, f, indent=2)
    sizes = {k: len(v) for k, v in splits.items()}
    logger.info("Saved splits to %s: %s", path, sizes)


def load_splits(path: str | Path) -> dict[str, list]:
    path = Path(path)
    with path.open() as f:
        splits = json.load(f)
    logger.info("Loaded splits from %s", path)
    return splits
