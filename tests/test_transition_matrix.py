from __future__ import annotations

import numpy as np
import pytest
import torch

from src.corruption.transition_matrix import TransitionMatrix


def test_train_only() -> None:
    matrix = TransitionMatrix(vocab_size=4, smoothing_alpha=0.0, min_count=1)
    matrix.fit([[0, 1, 0, 1], [1, 0]])

    assert matrix.transition_counts is not None
    assert matrix.transition_counts[2].sum() == 0.0
    assert matrix.transition_counts[3].sum() == 0.0

    # Fallback uses train-event frequencies only: 0 and 1 are the only observed tokens.
    expected = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(matrix.transition_probs[2], expected)
    np.testing.assert_allclose(matrix.transition_probs[3], expected)


def test_row_normalization() -> None:
    matrix = TransitionMatrix(vocab_size=3, smoothing_alpha=0.1, min_count=0)
    matrix.fit([[0, 1, 2, 1, 0], [2, 2, 0]])

    row_sums = matrix.transition_probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-9)


def test_no_self_replacement() -> None:
    matrix = TransitionMatrix(vocab_size=5, smoothing_alpha=0.1, min_count=1)
    matrix.fit([[0, 1, 2, 3, 4, 0, 2, 4, 1, 3]])

    for event_type in range(5):
        for _ in range(200):
            sampled = matrix.sample_replacement(event_type)
            assert sampled != event_type

    event_types = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long)
    mask = torch.ones_like(event_types, dtype=torch.bool)
    replaced = matrix.sample_replacement_batch(event_types, mask)
    assert torch.all(replaced != event_types)


def test_fallback() -> None:
    matrix = TransitionMatrix(vocab_size=5, smoothing_alpha=0.1, min_count=3)
    matrix.fit([[0, 1, 0, 1, 0, 1], [2]])

    # Event frequencies in train events: [3, 3, 1, 0, 0] / 7
    expected_fallback = np.array([3 / 7, 3 / 7, 1 / 7, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(matrix.transition_probs[1], expected_fallback)
    np.testing.assert_allclose(matrix.transition_probs[2], expected_fallback)
    np.testing.assert_allclose(matrix.transition_probs[3], expected_fallback)
    np.testing.assert_allclose(matrix.transition_probs[4], expected_fallback)

    assert matrix.low_count_mask.tolist() == [False, True, True, True, True]
    assert matrix.covered_types == 1
    assert matrix.fallback_type_count == 4
    assert matrix.coverage == pytest.approx(0.2)


def test_save_load(tmp_path) -> None:
    matrix = TransitionMatrix(vocab_size=4, smoothing_alpha=0.25, min_count=2)
    matrix.fit([[0, 1, 2, 1, 0, 2], [2, 3, 1]])

    npy_path = tmp_path / "transition_matrix.npy"
    meta_path = tmp_path / "transition_matrix_meta.json"
    matrix.save(str(npy_path), str(meta_path))

    loaded = TransitionMatrix.load(str(npy_path), str(meta_path))

    assert loaded.vocab_size == matrix.vocab_size
    assert loaded.smoothing_alpha == matrix.smoothing_alpha
    assert loaded.min_count == matrix.min_count
    assert loaded.fallback == matrix.fallback
    assert loaded.covered_types == matrix.covered_types
    assert loaded.fallback_type_count == matrix.fallback_type_count
    assert loaded.coverage == pytest.approx(matrix.coverage)
    np.testing.assert_allclose(loaded.transition_probs, matrix.transition_probs)
    np.testing.assert_array_equal(loaded.low_count_mask, matrix.low_count_mask)
    np.testing.assert_allclose(loaded.frequency_distribution, matrix.frequency_distribution)
