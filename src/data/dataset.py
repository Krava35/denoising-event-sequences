from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.forecasting import (
    build_forecast_stats,
    compute_derived_time_features,
    get_derived_time_feature_names,
    has_valid_forecast_cut,
    make_forecast_targets,
    sample_forecast_cut,
)

if TYPE_CHECKING:
    from src.data.preprocessing import EventPreprocessor

logger = logging.getLogger(__name__)


class EventSequenceDataset(Dataset):
    def __init__(
        self,
        df_events: pd.DataFrame,
        entity_ids: list,
        preprocessor: EventPreprocessor,
        config: dict,
        mode: str = "pretrain",
        forecast_stats: dict | None = None,
    ) -> None:
        if mode not in ("pretrain", "finetune", "eval", "forecast"):
            raise ValueError(f"mode must be pretrain|finetune|eval|forecast, got '{mode}'")

        self.config = config

        data_cfg = config.get("data", {})
        self.max_seq_len: int = int(data_cfg.get("max_seq_len", 256))
        self.mode = mode
        self._needs_forecast_targets = (
            mode == "forecast"
            or (
                mode == "pretrain"
                and bool(config.get("forecasting", {}).get("pretrain_aux_enabled", False))
            )
        )
        self._time_feature_names = get_derived_time_feature_names(config)
        self.forecast_stats = (
            forecast_stats
            if forecast_stats is not None or not self._needs_forecast_targets
            else build_forecast_stats(df_events, entity_ids, preprocessor, config)
        )

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
            if self._time_feature_names:
                time_features = compute_derived_time_features(ent["_tc"])
                num = np.concatenate([num, time_features], axis=1)

            cat = (
                ent[cat_cols].to_numpy(dtype=np.int64)
                if cat_cols
                else np.empty((N, 0), dtype=np.int64)
            )

            forecast_len = min(N, self.max_seq_len) if mode == "pretrain" else N
            if self._needs_forecast_targets and not has_valid_forecast_cut(forecast_len, config):
                logger.warning("Entity %s is too short for forecast mode, skipping", eid)
                continue

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

    def _format_forecast_targets(self, forecast_targets: dict) -> dict:
        return {
            "future_event_type_profile": torch.from_numpy(
                forecast_targets["future_event_type_profile"]
            ),
            "future_count_bucket": torch.tensor(
                forecast_targets["future_count_bucket"], dtype=torch.long
            ),
            "future_amount_stats": torch.from_numpy(
                forecast_targets["future_amount_stats"]
            ),
            "future_gap_bucket": torch.tensor(
                forecast_targets["future_gap_bucket"], dtype=torch.long
            ),
            "future_cat_profiles": [
                torch.from_numpy(x) for x in forecast_targets["future_cat_profiles"]
            ],
        }

    def _get_window(self, n: int, mode: str) -> tuple[int, int]:
        """
        Compute (start, end) slice indices for sequence truncation.

        pretrain mode → sliding_window: random crop from full history.
        finetune/eval → last_events: deterministic tail.

        Motivation: for long sequences (age_group median=863, gender median=324)
        sliding_window lets the model see different parts of history across epochs
        during pretraining, while eval always uses the tail for a consistent
        prediction setting.
        """
        if n <= self.max_seq_len:
            return 0, n

        if mode == "pretrain":
            strategy = self.config.get("data", {}).get("truncation_pretrain", "sliding_window")
        else:
            strategy = self.config.get("data", {}).get("truncation_eval", "last_events")

        if strategy == "sliding_window":
            start = random.randint(0, n - self.max_seq_len)
            return start, start + self.max_seq_len
        elif strategy == "last_events":
            return n - self.max_seq_len, n
        elif strategy == "first_events":
            return 0, self.max_seq_len
        else:
            raise ValueError(f"Unknown truncation strategy: {strategy}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]
        n = len(sample["event_type"])

        if self.mode == "forecast":
            cut = sample_forecast_cut(n, self.config)
            start = max(0, cut - self.max_seq_len)
            end = cut
            forecast_targets = make_forecast_targets(sample, cut, self.forecast_stats)
            L = end - start
            return {
                "event_type": torch.from_numpy(sample["event_type"][start:end]),
                "time_delta": torch.from_numpy(sample["time_delta"][start:end]),
                "num_features": torch.from_numpy(sample["num"][start:end]),
                "cat_features": torch.from_numpy(sample["cat"][start:end]),
                "attention_mask": torch.ones(L, dtype=torch.bool),
                "label": torch.tensor(sample["label"], dtype=torch.long),
                "entity_id": sample["entity_id"],
                "forecast_targets": self._format_forecast_targets(forecast_targets),
                "forecast_cut": torch.tensor(L, dtype=torch.long),
            }

        start, end = self._get_window(n, self.mode)

        L = end - start
        result = {
            "event_type": torch.from_numpy(sample["event_type"][start:end]),
            "time_delta": torch.from_numpy(sample["time_delta"][start:end]),
            "num_features": torch.from_numpy(sample["num"][start:end]),
            "cat_features": torch.from_numpy(sample["cat"][start:end]),
            "attention_mask": torch.ones(L, dtype=torch.bool),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "entity_id": sample["entity_id"],
        }
        if self.mode == "pretrain" and self._needs_forecast_targets:
            cut = sample_forecast_cut(L, self.config)
            window_sample = {
                "event_type": sample["event_type"][start:end],
                "time_delta": sample["time_delta"][start:end],
                "num": sample["num"][start:end],
                "cat": sample["cat"][start:end],
            }
            forecast_targets = make_forecast_targets(
                window_sample,
                cut,
                self.forecast_stats,
            )
            result["forecast_targets"] = self._format_forecast_targets(forecast_targets)
            result["forecast_cut"] = torch.tensor(cut, dtype=torch.long)
        return result
