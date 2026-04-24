from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class TransitionMatrix:
    """Frozen transition matrix for transition-aware categorical replacement."""

    def __init__(
        self,
        vocab_size: int,
        smoothing_alpha: float = 0.1,
        min_count: int = 5,
        fallback: str = "frequency_aware",
    ) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2 to sample replacement != original")
        if smoothing_alpha < 0.0:
            raise ValueError("smoothing_alpha must be >= 0.0")
        if min_count < 0:
            raise ValueError("min_count must be >= 0")
        if fallback not in {"frequency_aware", "uniform"}:
            raise ValueError("fallback must be one of {'frequency_aware', 'uniform'}")

        self.vocab_size = int(vocab_size)
        self.smoothing_alpha = float(smoothing_alpha)
        self.min_count = int(min_count)
        self.fallback = fallback

        self.transition_counts: np.ndarray | None = None
        self.transition_probs: np.ndarray | None = None
        self.frequency_distribution = np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float64)
        self.low_count_mask = np.zeros(self.vocab_size, dtype=bool)

        self.coverage = 0.0
        self.covered_types = 0
        self.fallback_type_count = self.vocab_size

        self._sampling_probs: np.ndarray | None = None
        self._fitted = False

    def fit(
        self,
        sequences: list[list[int]],
        attention_masks: list[list[bool]] | list[np.ndarray] | None = None,
    ) -> None:
        if attention_masks is not None and len(attention_masks) != len(sequences):
            raise ValueError("attention_masks must have the same length as sequences")

        counts = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
        event_counts = np.zeros(self.vocab_size, dtype=np.float64)

        for idx, sequence in enumerate(sequences):
            seq_arr = np.asarray(sequence, dtype=np.int64)
            if seq_arr.ndim != 1:
                raise ValueError("Each sequence must be one-dimensional")
            if seq_arr.size == 0:
                continue

            self._validate_ids(seq_arr)

            if attention_masks is None:
                valid = np.ones(seq_arr.shape[0], dtype=bool)
            else:
                valid = np.asarray(attention_masks[idx], dtype=bool)
                if valid.ndim != 1 or valid.shape[0] != seq_arr.shape[0]:
                    raise ValueError(
                        "Each attention mask must be one-dimensional and match sequence length"
                    )

            if valid.any():
                np.add.at(event_counts, seq_arr[valid], 1.0)

            if seq_arr.shape[0] < 2:
                continue

            pair_mask = valid[:-1] & valid[1:]
            if not pair_mask.any():
                continue

            prev_types = seq_arr[:-1][pair_mask]
            next_types = seq_arr[1:][pair_mask]
            np.add.at(counts, (prev_types, next_types), 1.0)

        raw_row_totals = counts.sum(axis=1)
        smoothed = counts + self.smoothing_alpha
        smoothed_row_totals = smoothed.sum(axis=1, keepdims=True)
        probs = np.divide(
            smoothed,
            smoothed_row_totals,
            out=np.full_like(smoothed, 1.0 / self.vocab_size),
            where=smoothed_row_totals > 0.0,
        )

        total_events = float(event_counts.sum())
        if total_events > 0.0:
            frequency_distribution = event_counts / total_events
        else:
            frequency_distribution = np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float64)

        low_count_mask = raw_row_totals < float(self.min_count)
        fallback_distribution = self._fallback_distribution(frequency_distribution)
        probs[low_count_mask] = fallback_distribution

        row_sums = probs.sum(axis=1, keepdims=True)
        probs = np.divide(
            probs,
            row_sums,
            out=np.full_like(probs, 1.0 / self.vocab_size),
            where=row_sums > 0.0,
        )

        self.transition_counts = counts
        self.transition_probs = probs
        self.frequency_distribution = frequency_distribution
        self.low_count_mask = low_count_mask
        self.covered_types = int((~low_count_mask).sum())
        self.fallback_type_count = int(low_count_mask.sum())
        self.coverage = self.covered_types / float(self.vocab_size)
        self._sampling_probs = self._build_sampling_probs(probs)
        self._fitted = True

    def sample_replacement(self, event_type_id: int) -> int:
        self._ensure_fitted()
        event_type = int(event_type_id)
        self._validate_ids(np.asarray([event_type], dtype=np.int64))
        probs = self._sampling_probs[event_type]
        return int(np.random.choice(self.vocab_size, p=probs))

    def sample_replacement_batch(
        self,
        event_types: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> torch.LongTensor:
        self._ensure_fitted()

        if event_types.ndim != 2 or mask.ndim != 2:
            raise ValueError("event_types and mask must be rank-2 tensors [B, L]")
        if event_types.shape != mask.shape:
            raise ValueError("event_types and mask must have the same shape")

        result = event_types.clone()
        if not bool(mask.any()):
            return result

        selected = event_types[mask]
        selected_cpu = selected.detach().to(dtype=torch.long, device="cpu")
        self._validate_ids(selected_cpu.numpy())

        replacements = selected_cpu.clone()
        unique_ids = torch.unique(selected_cpu)
        for event_type in unique_ids.tolist():
            positions = selected_cpu == int(event_type)
            n_samples = int(positions.sum().item())
            sampled = np.random.choice(
                self.vocab_size,
                size=n_samples,
                p=self._sampling_probs[int(event_type)],
            )
            replacements[positions] = torch.from_numpy(sampled).to(dtype=torch.long)

        result[mask] = replacements.to(device=event_types.device)
        return result

    def save(self, npy_path: str, meta_path: str) -> None:
        self._ensure_fitted()

        npy = Path(npy_path)
        meta = Path(meta_path)
        npy.parent.mkdir(parents=True, exist_ok=True)
        meta.parent.mkdir(parents=True, exist_ok=True)

        np.save(npy, self.transition_probs)
        payload = {
            "vocab_size": self.vocab_size,
            "smoothing_alpha": self.smoothing_alpha,
            "min_count": self.min_count,
            "fallback": self.fallback,
            "coverage": self.coverage,
            "covered_types": self.covered_types,
            "fallback_type_count": self.fallback_type_count,
            "low_count_mask": self.low_count_mask.tolist(),
            "frequency_distribution": self.frequency_distribution.tolist(),
        }
        with meta.open("w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, npy_path: str, meta_path: str) -> TransitionMatrix:
        matrix = np.load(npy_path)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Transition matrix must be square [V, V]")

        with Path(meta_path).open() as f:
            payload = json.load(f)

        vocab_size = int(payload.get("vocab_size", matrix.shape[0]))
        if matrix.shape != (vocab_size, vocab_size):
            raise ValueError(
                "Matrix shape does not match vocab_size in metadata: "
                f"{matrix.shape} vs vocab_size={vocab_size}"
            )

        instance = cls(
            vocab_size=vocab_size,
            smoothing_alpha=float(payload.get("smoothing_alpha", 0.1)),
            min_count=int(payload.get("min_count", 5)),
            fallback=str(payload.get("fallback", "frequency_aware")),
        )

        instance.transition_probs = matrix.astype(np.float64, copy=True)
        instance.transition_counts = np.zeros_like(instance.transition_probs)
        instance.low_count_mask = np.asarray(
            payload.get("low_count_mask", [False] * vocab_size),
            dtype=bool,
        )
        if instance.low_count_mask.shape != (vocab_size,):
            raise ValueError("low_count_mask length must match vocab_size")

        frequency_raw = np.asarray(
            payload.get("frequency_distribution", [1.0 / vocab_size] * vocab_size),
            dtype=np.float64,
        )
        if frequency_raw.shape != (vocab_size,):
            raise ValueError("frequency_distribution length must match vocab_size")
        freq_sum = float(frequency_raw.sum())
        if freq_sum <= 0.0:
            instance.frequency_distribution = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64)
        else:
            instance.frequency_distribution = frequency_raw / freq_sum

        instance.coverage = float(payload.get("coverage", float((~instance.low_count_mask).mean())))
        instance.covered_types = int(payload.get("covered_types", int((~instance.low_count_mask).sum())))
        instance.fallback_type_count = int(
            payload.get("fallback_type_count", int(instance.low_count_mask.sum()))
        )
        instance._sampling_probs = instance._build_sampling_probs(instance.transition_probs)
        instance._fitted = True
        return instance

    def _ensure_fitted(self) -> None:
        if not self._fitted or self.transition_probs is None or self._sampling_probs is None:
            raise RuntimeError("Call fit() or load() before sampling/saving")

    def _validate_ids(self, event_type_ids: np.ndarray) -> None:
        if event_type_ids.size == 0:
            return
        min_id = int(event_type_ids.min())
        max_id = int(event_type_ids.max())
        if min_id < 0 or max_id >= self.vocab_size:
            raise ValueError(
                f"Event type ids must be in [0, {self.vocab_size - 1}], got [{min_id}, {max_id}]"
            )

    def _fallback_distribution(self, frequency_distribution: np.ndarray) -> np.ndarray:
        if self.fallback == "frequency_aware":
            if float(frequency_distribution.sum()) > 0.0:
                return frequency_distribution
            return np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float64)
        return np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float64)

    def _build_sampling_probs(self, probs: np.ndarray) -> np.ndarray:
        sampling = probs.copy()
        diag = np.arange(self.vocab_size)
        sampling[diag, diag] = 0.0

        row_sums = sampling.sum(axis=1, keepdims=True)
        sampling = np.divide(
            sampling,
            row_sums,
            out=np.zeros_like(sampling),
            where=row_sums > 0.0,
        )

        for row_idx in np.where(row_sums.squeeze(1) <= 0.0)[0]:
            sampling[row_idx] = self._uniform_excluding_self(int(row_idx))

        return sampling

    def _uniform_excluding_self(self, event_type_id: int) -> np.ndarray:
        probs = np.full(self.vocab_size, 1.0 / (self.vocab_size - 1), dtype=np.float64)
        probs[event_type_id] = 0.0
        return probs
