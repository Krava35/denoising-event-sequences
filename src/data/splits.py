import json
import logging
import tempfile
from pathlib import Path

import numpy as np
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)
    n_entities = 1000
    entity_ids = [f"e_{i:04d}" for i in range(n_entities)]
    targets = rng.integers(0, 2, size=n_entities).tolist()

    # ~5 events per entity
    rows = []
    for eid, tgt in zip(entity_ids, targets):
        for _ in range(5):
            rows.append({"entity_id": eid, "event_type": rng.integers(0, 10), "target": tgt})
    df = pd.DataFrame(rows)

    splits = make_entity_splits(
        df,
        entity_col="entity_id",
        target_col="target",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        stratify=True,
    )

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert train_set.isdisjoint(val_set), "FAIL: train/val leakage"
    assert train_set.isdisjoint(test_set), "FAIL: train/test leakage"
    assert val_set.isdisjoint(test_set), "FAIL: val/test leakage"

    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == n_entities, f"FAIL: coverage {total} != {n_entities}"

    assert abs(len(splits["train"]) / n_entities - 0.70) < 0.02, "FAIL: train ratio"
    assert abs(len(splits["val"]) / n_entities - 0.15) < 0.02, "FAIL: val ratio"
    assert abs(len(splits["test"]) / n_entities - 0.15) < 0.02, "FAIL: test ratio"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    save_splits(splits, tmp_path)
    loaded = load_splits(tmp_path)
    assert loaded.keys() == splits.keys()
    for split_name in splits:
        assert set(loaded[split_name]) == set(splits[split_name]), (
            f"FAIL: round-trip mismatch in {split_name}"
        )

    print("All assertions passed.")
