from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from src.data.preprocessing import EventPreprocessor

logger = logging.getLogger(__name__)

# Max look-back window for pretrain random sampling
_PRETRAIN_WINDOW = 512


class EventSequenceDataset(Dataset):
    def __init__(
        self,
        df_events: pd.DataFrame,
        entity_ids: list,
        preprocessor: EventPreprocessor,
        config: dict,
        mode: str = "pretrain",
    ) -> None:
        if mode not in ("pretrain", "finetune", "eval"):
            raise ValueError(f"mode must be pretrain|finetune|eval, got '{mode}'")

        data_cfg = config.get("data", {})
        self.max_seq_len: int = int(data_cfg.get("max_seq_len", 256))
        self.mode = mode

        target_col: str = data_cfg.get("target_col", "target")
        entity_col: str = preprocessor.entity_col
        timestamp_col: str = preprocessor.timestamp_col
        num_cols: list[str] = preprocessor.numerical_cols
        cat_cols: list[str] = preprocessor.categorical_cols
        event_type_col: str = preprocessor.event_type_col

        entity_set = set(entity_ids)
        df_subset = df_events[df_events[entity_col].isin(entity_set)].copy()

        transformed = preprocessor.transform(df_subset)
        # Sorting keys — keep original values before transform touches them
        transformed["_ec"] = df_subset[entity_col].values
        transformed["_tc"] = df_subset[timestamp_col].values
        transformed = transformed.sort_values(["_ec", "_tc"], kind="stable")

        # One label per entity from original data
        if target_col in df_events.columns:
            entity_labels: dict = (
                df_events[df_events[entity_col].isin(entity_set)]
                .groupby(entity_col)[target_col]
                .first()
                .to_dict()
            )
        else:
            entity_labels = {}

        groups = transformed.groupby("_ec", sort=False)

        self._samples: list[dict] = []
        for eid in entity_ids:
            if eid not in groups.groups:
                logger.warning("Entity %s has no events, skipping", eid)
                continue

            ent = groups.get_group(eid)
            N = len(ent)

            event_type = ent[event_type_col].to_numpy(dtype=np.int64)
            time_delta = ent["time_delta"].to_numpy(dtype=np.float32)
            num = (
                ent[num_cols].to_numpy(dtype=np.float32)
                if num_cols
                else np.empty((N, 0), dtype=np.float32)
            )
            cat = (
                ent[cat_cols].to_numpy(dtype=np.int64)
                if cat_cols
                else np.empty((N, 0), dtype=np.int64)
            )

            self._samples.append(
                {
                    "event_type": event_type,
                    "time_delta": time_delta,
                    "num": num,
                    "cat": cat,
                    "label": int(entity_labels.get(eid, -1)),
                    "entity_id": str(eid),
                }
            )

        logger.info(
            "EventSequenceDataset: %d entities, mode=%s, max_seq_len=%d",
            len(self._samples),
            mode,
            self.max_seq_len,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]
        n = len(sample["event_type"])

        if self.mode == "pretrain":
            pool_start = max(0, n - _PRETRAIN_WINDOW)
            pool_len = n - pool_start
            if pool_len > self.max_seq_len:
                offset = int(np.random.randint(0, pool_len - self.max_seq_len + 1))
                start = pool_start + offset
                end = start + self.max_seq_len
            else:
                start = pool_start
                end = n
        else:
            # finetune / eval: deterministic last max_seq_len events
            start = max(0, n - self.max_seq_len)
            end = n

        L = end - start
        return {
            "event_type": torch.from_numpy(sample["event_type"][start:end]),
            "time_delta": torch.from_numpy(sample["time_delta"][start:end]),
            "num_features": torch.from_numpy(sample["num"][start:end]),
            "cat_features": torch.from_numpy(sample["cat"][start:end]),
            "attention_mask": torch.ones(L, dtype=torch.bool),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "entity_id": sample["entity_id"],
        }


# ── __main__ tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import copy
    import logging

    logging.basicConfig(level=logging.INFO)

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from src.data.collate import collate_fn
    from src.data.preprocessing import EventPreprocessor

    rng = np.random.default_rng(42)
    n_entities = 50
    events_per_entity = 30

    entity_ids = [f"e_{i:04d}" for i in range(n_entities)]
    targets = rng.integers(0, 2, size=n_entities)

    rows = []
    base_time = pd.Timestamp("2023-01-01")
    for i, eid in enumerate(entity_ids):
        t = base_time + pd.Timedelta(days=int(rng.integers(0, 30)))
        for _ in range(events_per_entity):
            t += pd.Timedelta(hours=int(rng.integers(1, 24)))
            rows.append(
                {
                    "entity_id": eid,
                    "timestamp": t,
                    "event_type": int(rng.integers(0, 10)),
                    "amount": float(rng.exponential(100)),
                    "num_feature": float(rng.normal(0, 1)),
                    "cat_col": int(rng.integers(0, 5)),
                    "target": int(targets[i]),
                }
            )

    df = pd.DataFrame(rows)

    config = {
        "data": {
            "event_type_col": "event_type",
            "timestamp_col": "timestamp",
            "numerical_cols": ["amount", "num_feature"],
            "categorical_cols": ["cat_col"],
            "group_col": "entity_id",
            "target_col": "target",
            "max_seq_len": 20,
            "min_vocab_count": 1,
        }
    }

    train_ids = entity_ids[:35]
    test_ids = entity_ids[35:]

    prep = EventPreprocessor(config)
    prep.fit(df, train_ids)

    # ── 1. Basic shape checks ────────────────────────────────────────────────
    dataset_ft = EventSequenceDataset(df, entity_ids, prep, config, mode="finetune")
    dataset_pt = EventSequenceDataset(df, entity_ids, prep, config, mode="pretrain")

    assert len(dataset_ft) == n_entities
    assert len(dataset_pt) == n_entities

    sample = dataset_ft[0]
    # All 30 events → truncated to max_seq_len=20
    assert sample["event_type"].shape == (20,), f"Expected (20,), got {sample['event_type'].shape}"
    assert sample["event_type"].dtype == torch.long
    assert sample["time_delta"].shape == (20,)
    assert sample["time_delta"].dtype == torch.float32
    assert sample["num_features"].shape == (20, 2)
    assert sample["num_features"].dtype == torch.float32
    assert sample["cat_features"].shape == (20, 1)
    assert sample["cat_features"].dtype == torch.long
    assert sample["attention_mask"].shape == (20,)
    assert sample["attention_mask"].dtype == torch.bool
    assert sample["attention_mask"].all(), "all tokens must be real (no padding in __getitem__)"
    assert sample["label"].shape == ()
    assert isinstance(sample["entity_id"], str)

    # ── 2. No NaN in float tensors ───────────────────────────────────────────
    for i in range(len(dataset_ft)):
        s = dataset_ft[i]
        assert not torch.isnan(s["time_delta"]).any(), f"NaN time_delta entity {i}"
        assert not torch.isnan(s["num_features"]).any(), f"NaN num_features entity {i}"

    # ── 3. Pretrain: random window ≤ max_seq_len ────────────────────────────
    for i in range(10):
        s = dataset_pt[0]
        assert s["event_type"].shape[0] <= config["data"]["max_seq_len"]

    # ── 4. Label propagation ─────────────────────────────────────────────────
    for i, eid in enumerate(entity_ids):
        s = dataset_ft[i]
        assert s["label"].item() == int(targets[i]), f"Label mismatch for entity {eid}"

    # ── 5. collate_fn: uniform batch (no padding) ────────────────────────────
    batch_uniform = [dataset_ft[i] for i in range(8)]
    col = collate_fn(batch_uniform)
    B, L = 8, 20
    assert col["event_type"].shape == (B, L)
    assert col["time_delta"].shape == (B, L)
    assert col["num_features"].shape == (B, L, 2)
    assert col["cat_features"].shape == (B, L, 1)
    assert col["attention_mask"].shape == (B, L)
    assert col["label"].shape == (B,)
    assert col["attention_mask"].all(), "no padding → all True"
    assert not torch.isnan(col["time_delta"]).any()
    assert not torch.isnan(col["num_features"]).any()

    # ── 6. collate_fn: mixed batch (forces padding) ──────────────────────────
    # Construct a short sample by trimming a real sample to length 8
    short = copy.deepcopy(dataset_ft[0])
    short_len = 8
    for k in ("event_type", "time_delta", "num_features", "cat_features", "attention_mask"):
        short[k] = short[k][:short_len]

    mixed = [dataset_ft[i] for i in range(4)] + [short]
    col_m = collate_fn(mixed)
    B_m = 5
    max_L_m = 20  # longest = 20

    assert col_m["event_type"].shape == (B_m, max_L_m)
    assert col_m["attention_mask"].shape == (B_m, max_L_m)

    # padding mask: real tokens True, padded positions False
    short_idx = 4  # position of the short sample
    assert col_m["attention_mask"][short_idx, :short_len].all(), "real tokens must be True"
    assert not col_m["attention_mask"][short_idx, short_len:].any(), "padding must be False"

    # PAD positions = 0
    assert (col_m["event_type"][short_idx, short_len:] == 0).all(), "event_type PAD != 0"
    assert (col_m["cat_features"][short_idx, short_len:] == 0).all(), "cat_features PAD != 0"
    assert (col_m["time_delta"][short_idx, short_len:] == 0.0).all(), "time_delta PAD != 0"
    assert (col_m["num_features"][short_idx, short_len:] == 0.0).all(), "num_features PAD != 0"

    # no NaN anywhere
    assert not torch.isnan(col_m["time_delta"]).any()
    assert not torch.isnan(col_m["num_features"]).any()

    # ── 7. DataLoader smoke test ─────────────────────────────────────────────
    loader = DataLoader(dataset_ft, batch_size=8, shuffle=True, collate_fn=collate_fn)
    for batch_dl in loader:
        assert batch_dl["event_type"].ndim == 2
        assert batch_dl["attention_mask"].ndim == 2
        assert batch_dl["num_features"].ndim == 3
        break

    print("All assertions passed.")
