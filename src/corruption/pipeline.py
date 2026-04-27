from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.corruption.categorical import corrupt_categorical_features, corrupt_event_type
from src.corruption.continuous import corrupt_numerical_features, corrupt_time_delta
from src.corruption.event_masking import mask_whole_events

if TYPE_CHECKING:
    from src.corruption.transition_matrix import TransitionMatrix


class CorruptionPipeline:
    """Orchestrate all corruption steps for a single clean batch.

    Apply order:
      1. mask_whole_events   — first, so later steps never overwrite event-level masks
      2. corrupt_event_type
      3. corrupt_categorical_features
      4. corrupt_time_delta
      5. corrupt_numerical_features

    Steps are skipped silently when the corresponding key is absent from the batch
    or when corruption_prob / selected_prob is 0.
    """

    def __init__(
        self,
        config: dict,
        transition_matrix: TransitionMatrix | None = None,
        vocab_sizes: dict | None = None,
        time_transform: str = "log1p",
    ) -> None:
        """
        Args:
            config: corruption sub-dict from the YAML config
                    (i.e. config["corruption"] or the whole config — both work
                    because only known sub-keys are accessed).
            transition_matrix: fitted TransitionMatrix or None.
            vocab_sizes: dict with keys "event_type" (int) and "cat_features"
                         (list[int]) used for random-replacement sampling.
            time_transform: value of data.time_transform ('log1p' or 'none').
                            Passed to corrupt_time_delta to warn when Gaussian
                            noise is applied to raw (non-log) time deltas.
        """
        self._cfg = config
        self._tm = transition_matrix
        self._vocab_sizes = vocab_sizes or {}
        self._time_transform = time_transform

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _section(self, *keys: str) -> dict:
        """Safely navigate nested config keys, returning {} if missing."""
        node = self._cfg
        for k in keys:
            if not isinstance(node, dict):
                return {}
            node = node.get(k, {})
        return node if isinstance(node, dict) else {}

    def _get(self, default, *keys: str):
        """Get a scalar from the config by path, returning default if missing."""
        node = self._cfg
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
            if not isinstance(node, dict):
                return node
        return node

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(self, clean_batch: dict) -> tuple[dict, dict, dict]:
        """Corrupt a clean batch and return (corrupted, targets, masks).

        Args:
            clean_batch: dict produced by the DataLoader.  Must contain
                         'attention_mask': BoolTensor[B, L].

        Returns:
            corrupted_batch: same keys as clean_batch, values corrupted.
            targets: original values used for reconstruction loss.
            masks: boolean masks indicating which positions were corrupted.
        """
        attention_mask = clean_batch["attention_mask"]

        batch = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                 for k, v in clean_batch.items()}

        # Snapshot originals from clean_batch BEFORE any corruption step.
        # Targets must always reflect the pre-corruption values so that
        # event-level masking (step 1) does not contaminate them.
        _target_keys = ("event_type", "time_delta", "num_features", "cat_features")
        targets: dict = {
            k: clean_batch[k].clone()
            for k in _target_keys
            if k in clean_batch and isinstance(clean_batch[k], torch.Tensor)
        }
        masks: dict = {}

        # ── 1. Event-level masking ─────────────────────────────────────
        el_cfg = self._section("event_level_masking")
        el_prob = el_cfg.get("prob", 0.0)
        if el_prob > 0.0:
            batch, event_level_mask = mask_whole_events(
                batch,
                attention_mask,
                event_mask_prob=el_prob,
                mask_tokens=el_cfg.get("mask_tokens"),
            )
        else:
            B, L = attention_mask.shape
            event_level_mask = torch.zeros(B, L, dtype=torch.bool,
                                           device=attention_mask.device)
        masks["event_level"] = event_level_mask

        # ── 2. Event type ──────────────────────────────────────────────
        if "event_type" in batch:
            et_cfg = self._section("event_type")
            vocab_size = self._vocab_sizes.get("event_type", 100)
            use_tm = et_cfg.get("use_transition_aware_replacement", False)

            corrupted_et, pred_mask_et, _ = corrupt_event_type(
                batch["event_type"],
                attention_mask,
                selected_prob=et_cfg.get("selected_prob", 0.40),
                mask_prob=et_cfg.get("mask_prob", 0.28),
                transition_replace_prob=et_cfg.get("transition_replace_prob", 0.00),
                random_replace_prob=et_cfg.get("random_replace_prob", 0.10),
                keep_predict_prob=et_cfg.get("keep_predict_prob", 0.02),
                vocab_size=vocab_size,
                transition_matrix=self._tm if use_tm else None,
                excluded_mask=event_level_mask,
            )
            batch["event_type"] = corrupted_et
            masks["event_type"] = pred_mask_et

        # ── 3. Categorical features ────────────────────────────────────
        if "cat_features" in batch:
            cf_cfg = self._section("categorical_features")
            cat_vocab_sizes = self._vocab_sizes.get("cat_features")

            corrupted_cf, pred_mask_cf, _ = corrupt_categorical_features(
                batch["cat_features"],
                attention_mask,
                mask_prob=cf_cfg.get("mask_prob", 0.15),
                random_replace_prob=cf_cfg.get("random_replace_prob", 0.05),
                vocab_sizes=cat_vocab_sizes,
                excluded_mask=event_level_mask,
            )
            batch["cat_features"] = corrupted_cf
            masks["cat_features"] = pred_mask_cf

        # ── 4. Time delta ──────────────────────────────────────────────
        if "time_delta" in batch:
            tn_cfg = self._section("time_noise")
            corruption_prob = tn_cfg.get("corruption_prob", 0.0)
            if corruption_prob > 0.0:
                corrupted_td, time_mask, _ = corrupt_time_delta(
                    batch["time_delta"],
                    attention_mask,
                    corruption_prob=corruption_prob,
                    min_std=tn_cfg.get("min_std", 0.05),
                    max_std=tn_cfg.get("max_std", 0.30),
                    sampling_level=tn_cfg.get("sampling_level", "batch"),
                    already_log_transformed=(self._time_transform == "log1p"),
                )
                batch["time_delta"] = corrupted_td
                masks["time_delta"] = time_mask
            else:
                B, L = attention_mask.shape
                masks["time_delta"] = torch.zeros(B, L, dtype=torch.bool,
                                                  device=attention_mask.device)

        # ── 5. Numerical features ──────────────────────────────────────
        if "num_features" in batch:
            nn_cfg = self._section("numerical_noise")
            corruption_prob = nn_cfg.get("corruption_prob", 0.0)
            if corruption_prob > 0.0:
                corrupted_nf, num_mask, _ = corrupt_numerical_features(
                    batch["num_features"],
                    attention_mask,
                    corruption_prob=corruption_prob,
                    min_std=nn_cfg.get("min_std", 0.03),
                    max_std=nn_cfg.get("max_std", 0.15),
                    sampling_level=nn_cfg.get("sampling_level", "batch"),
                )
                batch["num_features"] = corrupted_nf
                masks["num_features"] = num_mask
            else:
                B, L = attention_mask.shape
                masks["num_features"] = torch.zeros(B, L, dtype=torch.bool,
                                                    device=attention_mask.device)

        return batch, targets, masks
