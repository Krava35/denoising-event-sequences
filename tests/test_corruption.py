import pytest
import torch

from src.corruption.categorical import (
    _NUM_SPECIAL,
    _sample_random_tokens,
    corrupt_categorical_features,
    corrupt_event_type,
)
from src.corruption.continuous import (
    corrupt_numerical_features,
    corrupt_time_delta,
)
from src.corruption.event_masking import mask_whole_events

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_batch():
    torch.manual_seed(0)
    B, L = 4, 12
    event_type = torch.randint(_NUM_SPECIAL, 30, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.bool)
    # Last 3 positions of row 2 are padding
    attention_mask[2, 9:] = False
    event_type[2, 9:] = 0
    return event_type, attention_mask


@pytest.fixture
def large_batch():
    """Large enough to test probability calibration."""
    torch.manual_seed(42)
    B, L = 256, 128
    event_type = torch.randint(_NUM_SPECIAL, 80, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.bool)
    return event_type, attention_mask


@pytest.fixture
def cat_batch():
    torch.manual_seed(7)
    B, L, _ = 4, 12, 3
    vocab_sizes = [20, 35, 50]
    cat = torch.stack(
        [torch.randint(_NUM_SPECIAL, vs, (B, L)) for vs in vocab_sizes], dim=-1
    )
    attention_mask = torch.ones(B, L, dtype=torch.bool)
    attention_mask[1, 10:] = False
    cat[1, 10:] = 0
    return cat, attention_mask, vocab_sizes


# ---------------------------------------------------------------------------
# _sample_random_tokens
# ---------------------------------------------------------------------------

class TestSampleRandomTokens:
    def test_shape(self, small_batch):
        event_type, _ = small_batch
        result = _sample_random_tokens(event_type, vocab_size=50)
        assert result.shape == event_type.shape

    def test_never_equals_original(self):
        torch.manual_seed(1)
        original = torch.randint(_NUM_SPECIAL, 30, (32, 64))
        result = _sample_random_tokens(original, vocab_size=50)
        assert (result != original).all()

    def test_range(self):
        torch.manual_seed(2)
        original = torch.randint(_NUM_SPECIAL, 30, (32, 64))
        result = _sample_random_tokens(original, vocab_size=50)
        assert (result >= _NUM_SPECIAL).all()
        assert (result < 50).all()

    def test_original_is_special_token(self):
        # original in [0, _NUM_SPECIAL) — still must not match and must be in range
        original = torch.zeros(10, 10, dtype=torch.long)  # PAD tokens
        result = _sample_random_tokens(original, vocab_size=20)
        assert (result >= _NUM_SPECIAL).all()
        assert (result < 20).all()
        assert (result != original).all()


# ---------------------------------------------------------------------------
# corrupt_event_type — shapes and return types
# ---------------------------------------------------------------------------

class TestCorruptEventTypeShapes:
    def test_output_shapes(self, small_batch):
        event_type, attention_mask = small_batch
        corrupted, pred_mask, original = corrupt_event_type(
            event_type, attention_mask, vocab_size=50
        )
        assert corrupted.shape == event_type.shape
        assert pred_mask.shape == event_type.shape
        assert original.shape == event_type.shape

    def test_dtypes(self, small_batch):
        event_type, attention_mask = small_batch
        corrupted, pred_mask, original = corrupt_event_type(
            event_type, attention_mask, vocab_size=50
        )
        assert corrupted.dtype == torch.long
        assert pred_mask.dtype == torch.bool
        assert original.dtype == torch.long

    def test_original_unchanged(self, small_batch):
        event_type, attention_mask = small_batch
        _, _, original = corrupt_event_type(event_type, attention_mask, vocab_size=50)
        assert (original == event_type).all()


# ---------------------------------------------------------------------------
# corrupt_event_type — padding invariant
# ---------------------------------------------------------------------------

class TestCorruptEventTypePadding:
    def test_padding_not_in_prediction(self, small_batch):
        event_type, attention_mask = small_batch
        _, pred_mask, _ = corrupt_event_type(event_type, attention_mask, vocab_size=50)
        assert pred_mask[~attention_mask].sum() == 0

    def test_padding_not_corrupted(self, small_batch):
        event_type, attention_mask = small_batch
        corrupted, _, original = corrupt_event_type(
            event_type, attention_mask, vocab_size=50
        )
        assert (corrupted[~attention_mask] == original[~attention_mask]).all()

    def test_all_padding_batch(self):
        event_type = torch.zeros(3, 8, dtype=torch.long)
        attention_mask = torch.zeros(3, 8, dtype=torch.bool)
        corrupted, pred_mask, original = corrupt_event_type(
            event_type, attention_mask, vocab_size=20
        )
        assert pred_mask.sum() == 0
        assert (corrupted == original).all()


# ---------------------------------------------------------------------------
# corrupt_event_type — mask operation
# ---------------------------------------------------------------------------

class TestCorruptEventTypeMaskOp:
    def test_masked_positions_equal_mask_token(self, small_batch):
        torch.manual_seed(10)
        event_type, attention_mask = small_batch
        mask_token_id = 2
        corrupted, pred_mask, original = corrupt_event_type(
            event_type, attention_mask,
            vocab_size=50,
            mask_token_id=mask_token_id,
        )
        # Every position where corrupted == mask_token_id AND it's real
        # must be in pred_mask; and its original must differ (we corrupted it)
        mask_applied = (corrupted == mask_token_id) & attention_mask
        assert (pred_mask[mask_applied]).all()

    def test_mask_token_only_on_real_positions(self, small_batch):
        torch.manual_seed(11)
        event_type, attention_mask = small_batch
        corrupted, _, _ = corrupt_event_type(
            event_type, attention_mask, vocab_size=50, mask_token_id=2
        )
        # No padding position should become mask_token_id (unless it was already 2 in input)
        pad_positions = ~attention_mask
        assert (corrupted[pad_positions] == 0).all()  # pad_token_id=0


# ---------------------------------------------------------------------------
# corrupt_event_type — random operation
# ---------------------------------------------------------------------------

class TestCorruptEventTypeRandomOp:
    def test_random_tokens_in_valid_range(self, large_batch):
        torch.manual_seed(20)
        event_type, attention_mask = large_batch
        corrupted, pred_mask, original = corrupt_event_type(
            event_type, attention_mask, vocab_size=80
        )
        # All non-pad tokens in corrupted must be in [0, vocab_size)
        real = attention_mask
        assert (corrupted[real] >= 0).all()
        assert (corrupted[real] < 80).all()


# ---------------------------------------------------------------------------
# corrupt_event_type — probability calibration
# ---------------------------------------------------------------------------

class TestCorruptEventTypeProbCalibration:
    def test_selection_rate(self, large_batch):
        torch.manual_seed(30)
        event_type, attention_mask = large_batch
        selected_prob = 0.40

        _, pred_mask, _ = corrupt_event_type(
            event_type, attention_mask,
            selected_prob=selected_prob,
            mask_prob=0.28,
            transition_replace_prob=0.08,
            random_replace_prob=0.02,
            keep_predict_prob=0.02,
            vocab_size=80,
        )
        n_real = attention_mask.sum().item()
        n_selected = pred_mask.sum().item()
        actual_rate = n_selected / n_real
        assert abs(actual_rate - selected_prob) < 0.02, (
            f"Selection rate {actual_rate:.4f} too far from {selected_prob}"
        )

    def test_mask_operation_rate(self, large_batch):
        torch.manual_seed(31)
        event_type, attention_mask = large_batch
        mask_prob = 0.28
        mask_token_id = 2

        corrupted, _, _ = corrupt_event_type(
            event_type, attention_mask,
            mask_prob=mask_prob,
            vocab_size=80,
            mask_token_id=mask_token_id,
        )
        n_real = attention_mask.sum().item()
        # Count positions where event_type was not mask_token_id but corrupted is
        became_mask = (corrupted == mask_token_id) & (event_type != mask_token_id) & attention_mask
        actual_rate = became_mask.sum().item() / n_real
        assert abs(actual_rate - mask_prob) < 0.02, (
            f"Mask rate {actual_rate:.4f} too far from {mask_prob}"
        )


# ---------------------------------------------------------------------------
# corrupt_event_type — determinism
# ---------------------------------------------------------------------------

class TestCorruptEventTypeDeterminism:
    def test_deterministic_with_seed(self, small_batch):
        event_type, attention_mask = small_batch
        torch.manual_seed(99)
        r1 = corrupt_event_type(event_type, attention_mask, vocab_size=50)
        torch.manual_seed(99)
        r2 = corrupt_event_type(event_type, attention_mask, vocab_size=50)
        assert (r1[0] == r2[0]).all()
        assert (r1[1] == r2[1]).all()
        assert (r1[2] == r2[2]).all()


# ---------------------------------------------------------------------------
# corrupt_event_type — validation
# ---------------------------------------------------------------------------

class TestCorruptEventTypeValidation:
    def test_prob_sum_mismatch_raises(self, small_batch):
        event_type, attention_mask = small_batch
        with pytest.raises(AssertionError, match="must equal selected_prob"):
            corrupt_event_type(
                event_type, attention_mask,
                selected_prob=0.40,
                mask_prob=0.30,  # sum = 0.42, not 0.40
                transition_replace_prob=0.08,
                random_replace_prob=0.02,
                keep_predict_prob=0.02,
                vocab_size=50,
            )

    def test_vocab_size_too_small_raises(self, small_batch):
        event_type, attention_mask = small_batch
        with pytest.raises(AssertionError, match="vocab_size must be"):
            corrupt_event_type(event_type, attention_mask, vocab_size=_NUM_SPECIAL)

    def test_shape_mismatch_raises(self):
        event_type = torch.randint(5, 20, (4, 12))
        attention_mask = torch.ones(4, 10, dtype=torch.bool)  # wrong L
        with pytest.raises(AssertionError):
            corrupt_event_type(event_type, attention_mask, vocab_size=50)


# ---------------------------------------------------------------------------
# corrupt_event_type — transition_matrix=None fallback
# ---------------------------------------------------------------------------

class TestCorruptEventTypeTransitionFallback:
    def test_transition_fallback_produces_valid_tokens(self, large_batch):
        torch.manual_seed(40)
        event_type, attention_mask = large_batch
        # Force high transition prob and no transition matrix
        corrupted, pred_mask, original = corrupt_event_type(
            event_type, attention_mask,
            selected_prob=0.40,
            mask_prob=0.00,
            transition_replace_prob=0.40,
            random_replace_prob=0.00,
            keep_predict_prob=0.00,
            vocab_size=80,
            transition_matrix=None,
        )
        # Positions in pred_mask should have valid tokens, not equal original
        trans_positions = pred_mask & attention_mask
        assert (corrupted[trans_positions] >= _NUM_SPECIAL).all()
        assert (corrupted[trans_positions] < 80).all()
        assert (corrupted[trans_positions] != original[trans_positions]).all()


# ---------------------------------------------------------------------------
# corrupt_categorical_features — shapes and return types
# ---------------------------------------------------------------------------

class TestCorruptCatShapes:
    def test_output_shapes(self, cat_batch):
        cat, attention_mask, vocab_sizes = cat_batch
        corrupted, pred_mask, original = corrupt_categorical_features(
            cat, attention_mask, vocab_sizes=vocab_sizes
        )
        assert corrupted.shape == cat.shape
        assert pred_mask.shape == cat.shape
        assert original.shape == cat.shape

    def test_dtypes(self, cat_batch):
        cat, attention_mask, vocab_sizes = cat_batch
        corrupted, pred_mask, original = corrupt_categorical_features(
            cat, attention_mask, vocab_sizes=vocab_sizes
        )
        assert corrupted.dtype == torch.long
        assert pred_mask.dtype == torch.bool
        assert original.dtype == torch.long

    def test_original_unchanged(self, cat_batch):
        cat, attention_mask, vocab_sizes = cat_batch
        _, _, original = corrupt_categorical_features(
            cat, attention_mask, vocab_sizes=vocab_sizes
        )
        assert (original == cat).all()


# ---------------------------------------------------------------------------
# corrupt_categorical_features — padding invariant
# ---------------------------------------------------------------------------

class TestCorruptCatPadding:
    def test_padding_not_in_prediction(self, cat_batch):
        cat, attention_mask, vocab_sizes = cat_batch
        _, pred_mask, _ = corrupt_categorical_features(
            cat, attention_mask, vocab_sizes=vocab_sizes
        )
        pad_expanded = ~attention_mask.unsqueeze(-1).expand_as(pred_mask)
        assert pred_mask[pad_expanded].sum() == 0

    def test_padding_not_corrupted(self, cat_batch):
        cat, attention_mask, vocab_sizes = cat_batch
        corrupted, _, original = corrupt_categorical_features(
            cat, attention_mask, vocab_sizes=vocab_sizes
        )
        pad_expanded = ~attention_mask.unsqueeze(-1).expand_as(corrupted)
        assert (corrupted[pad_expanded] == original[pad_expanded]).all()

    def test_all_padding_batch(self):
        cat = torch.zeros(3, 8, 2, dtype=torch.long)
        attention_mask = torch.zeros(3, 8, dtype=torch.bool)
        corrupted, pred_mask, original = corrupt_categorical_features(
            cat, attention_mask, vocab_sizes=[20, 30]
        )
        assert pred_mask.sum() == 0
        assert (corrupted == original).all()


# ---------------------------------------------------------------------------
# corrupt_categorical_features — mask operation
# ---------------------------------------------------------------------------

class TestCorruptCatMaskOp:
    def test_masked_positions_equal_mask_token(self, cat_batch):
        torch.manual_seed(50)
        cat, attention_mask, vocab_sizes = cat_batch
        mask_token_id = 3
        corrupted, pred_mask, _ = corrupt_categorical_features(
            cat, attention_mask,
            mask_token_id=mask_token_id,
            vocab_sizes=vocab_sizes,
        )
        became_mask = (corrupted == mask_token_id) & attention_mask.unsqueeze(-1)
        assert (pred_mask[became_mask]).all()


# ---------------------------------------------------------------------------
# corrupt_categorical_features — random operation
# ---------------------------------------------------------------------------

class TestCorruptCatRandomOp:
    def test_random_tokens_in_valid_range(self):
        torch.manual_seed(60)
        B, L, _ = 32, 64, 4
        vocab_sizes = [20, 30, 40, 50]
        cat = torch.stack(
            [torch.randint(_NUM_SPECIAL, vs, (B, L)) for vs in vocab_sizes], dim=-1
        )
        attention_mask = torch.ones(B, L, dtype=torch.bool)

        corrupted, pred_mask, original = corrupt_categorical_features(
            cat, attention_mask,
            mask_prob=0.00,
            random_replace_prob=0.20,
            vocab_sizes=vocab_sizes,
        )
        # All random positions: value in [_NUM_SPECIAL, vocab_sizes[j]) and != original
        for j, vs in enumerate(vocab_sizes):
            rand_j = pred_mask[:, :, j]
            if rand_j.sum() == 0:
                continue
            vals = corrupted[:, :, j][rand_j]
            orig_vals = original[:, :, j][rand_j]
            assert (vals >= _NUM_SPECIAL).all(), f"feature {j}: value below _NUM_SPECIAL"
            assert (vals < vs).all(), f"feature {j}: value >= vocab_size"
            assert (vals != orig_vals).all(), f"feature {j}: random token == original"


# ---------------------------------------------------------------------------
# corrupt_categorical_features — probability calibration
# ---------------------------------------------------------------------------

class TestCorruptCatProbCalibration:
    def test_mask_rate(self):
        torch.manual_seed(70)
        B, L, N = 128, 128, 2
        cat = torch.randint(_NUM_SPECIAL, 40, (B, L, N))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        mask_prob = 0.15
        mask_token_id = 3

        corrupted, _, _ = corrupt_categorical_features(
            cat, attention_mask,
            mask_prob=mask_prob,
            random_replace_prob=0.00,
            mask_token_id=mask_token_id,
        )
        n_real = attention_mask.sum().item() * N
        became_mask = (
            (corrupted == mask_token_id) & (cat != mask_token_id)
            & attention_mask.unsqueeze(-1)
        )
        actual_rate = became_mask.sum().item() / n_real
        assert abs(actual_rate - mask_prob) < 0.02, (
            f"Mask rate {actual_rate:.4f} too far from {mask_prob}"
        )


# ---------------------------------------------------------------------------
# corrupt_categorical_features — determinism
# ---------------------------------------------------------------------------

class TestCorruptCatDeterminism:
    def test_deterministic_with_seed(self, cat_batch):
        cat, attention_mask, vocab_sizes = cat_batch
        torch.manual_seed(88)
        r1 = corrupt_categorical_features(cat, attention_mask, vocab_sizes=vocab_sizes)
        torch.manual_seed(88)
        r2 = corrupt_categorical_features(cat, attention_mask, vocab_sizes=vocab_sizes)
        assert (r1[0] == r2[0]).all()
        assert (r1[1] == r2[1]).all()


# ---------------------------------------------------------------------------
# corrupt_categorical_features — validation
# ---------------------------------------------------------------------------

class TestCorruptCatValidation:
    def test_vocab_sizes_length_mismatch_raises(self, cat_batch):
        cat, attention_mask, _ = cat_batch
        with pytest.raises(AssertionError, match="must equal N_cat"):
            corrupt_categorical_features(cat, attention_mask, vocab_sizes=[20, 30])

    def test_attention_mask_shape_mismatch_raises(self, cat_batch):
        cat, _, vocab_sizes = cat_batch
        bad_mask = torch.ones(cat.shape[0], cat.shape[1] + 1, dtype=torch.bool)
        with pytest.raises(AssertionError):
            corrupt_categorical_features(cat, bad_mask, vocab_sizes=vocab_sizes)

    def test_vocab_sizes_none_skips_random(self):
        torch.manual_seed(91)
        B, L, N = 8, 16, 2
        cat = torch.randint(_NUM_SPECIAL, 20, (B, L, N))
        attention_mask = torch.ones(B, L, dtype=torch.bool)

        corrupted, pred_mask, original = corrupt_categorical_features(
            cat, attention_mask,
            mask_prob=0.00,
            random_replace_prob=0.30,
            vocab_sizes=None,  # no vocab_sizes → skip random, keep original
        )
        # is_rand positions in pred_mask but corrupted == original (BERT-keep fallback)
        rand_positions = pred_mask
        assert (corrupted[rand_positions] == original[rand_positions]).all()

    def test_zero_cat_features(self):
        cat = torch.zeros(4, 8, 0, dtype=torch.long)
        attention_mask = torch.ones(4, 8, dtype=torch.bool)
        corrupted, pred_mask, original = corrupt_categorical_features(
            cat, attention_mask, vocab_sizes=[]
        )
        assert corrupted.shape == (4, 8, 0)
        assert pred_mask.sum() == 0


# ===========================================================================
# corrupt_time_delta
# ===========================================================================

class TestCorruptTimeDeltaShapes:
    def test_output_shapes(self):
        torch.manual_seed(0)
        B, L = 4, 12
        x = torch.randn(B, L)
        mask = torch.ones(B, L, dtype=torch.bool)
        corrupted, time_mask, original = corrupt_time_delta(x, mask)
        assert corrupted.shape == (B, L)
        assert time_mask.shape == (B, L)
        assert original.shape == (B, L)

    def test_dtypes(self):
        x = torch.randn(3, 10)
        mask = torch.ones(3, 10, dtype=torch.bool)
        corrupted, time_mask, original = corrupt_time_delta(x, mask)
        assert corrupted.dtype == torch.float32
        assert time_mask.dtype == torch.bool
        assert original.dtype == torch.float32

    def test_original_unchanged(self):
        x = torch.randn(4, 8)
        mask = torch.ones(4, 8, dtype=torch.bool)
        _, _, original = corrupt_time_delta(x, mask)
        assert (original == x).all()


class TestCorruptTimeDeltaPadding:
    def test_padding_not_corrupted(self):
        torch.manual_seed(1)
        B, L = 4, 12
        x = torch.randn(B, L)
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        attention_mask[2, 8:] = False
        x[2, 8:] = 0.0

        corrupted, time_mask, original = corrupt_time_delta(x, attention_mask)
        assert (corrupted[~attention_mask] == original[~attention_mask]).all()
        assert time_mask[~attention_mask].sum() == 0

    def test_all_padding_returns_unchanged(self):
        x = torch.randn(3, 8)
        mask = torch.zeros(3, 8, dtype=torch.bool)
        corrupted, time_mask, original = corrupt_time_delta(x, mask)
        assert (corrupted == original).all()
        assert time_mask.sum() == 0


class TestCorruptTimeDeltaNoise:
    def test_corrupted_positions_differ_from_original(self):
        torch.manual_seed(2)
        B, L = 16, 32
        x = torch.randn(B, L)
        mask = torch.ones(B, L, dtype=torch.bool)
        corrupted, time_mask, original = corrupt_time_delta(
            x, mask, corruption_prob=1.0
        )
        # With prob 1.0 all positions are selected; noise is non-zero almost surely
        assert not (corrupted == original).all()

    def test_uncorrupted_positions_unchanged(self):
        torch.manual_seed(3)
        B, L = 8, 20
        x = torch.randn(B, L)
        mask = torch.ones(B, L, dtype=torch.bool)
        corrupted, time_mask, original = corrupt_time_delta(x, mask)
        unselected = ~time_mask & mask
        assert (corrupted[unselected] == original[unselected]).all()

    def test_noise_magnitude_within_expected_range(self):
        torch.manual_seed(4)
        B, L = 128, 256
        x = torch.zeros(B, L)
        mask = torch.ones(B, L, dtype=torch.bool)
        min_std, max_std = 0.05, 0.30
        corrupted, time_mask, original = corrupt_time_delta(
            x, mask, corruption_prob=1.0, min_std=min_std, max_std=max_std
        )
        diffs = (corrupted - original)[time_mask]
        # std of diffs should be in [min_std, max_std] (sigma is uniform between them)
        std = diffs.std().item()
        # Expected mean sigma ≈ 0.175; std of N(0, sigma) has std = sigma
        assert min_std * 0.5 < std < max_std * 2.0, (
            f"noise std {std:.4f} outside expected range"
        )


class TestCorruptTimeDeltaCalibration:
    def test_corruption_rate(self):
        torch.manual_seed(5)
        B, L = 256, 128
        x = torch.randn(B, L)
        mask = torch.ones(B, L, dtype=torch.bool)
        corruption_prob = 0.30
        _, time_mask, _ = corrupt_time_delta(x, mask, corruption_prob=corruption_prob)
        actual = time_mask.float().mean().item()
        assert abs(actual - corruption_prob) < 0.02, (
            f"corruption rate {actual:.4f} too far from {corruption_prob}"
        )


class TestCorruptTimeDeltaSamplingLevel:
    def test_batch_level_same_sigma_across_rows(self):
        # With sampling_level='batch', sigma is identical for all rows.
        # Check that all-zero input gets equal |noise| std across rows.
        torch.manual_seed(6)
        B, L = 8, 512
        x = torch.zeros(B, L)
        mask = torch.ones(B, L, dtype=torch.bool)
        corrupted, time_mask, original = corrupt_time_delta(
            x, mask, corruption_prob=1.0,
            min_std=0.10, max_std=0.10,  # pin sigma=0.10
            sampling_level="batch",
        )
        diffs = corrupted - original  # [B, L]
        row_stds = diffs.std(dim=1)  # [B]
        # All rows share the same sigma, so row stds should be close
        assert row_stds.max() - row_stds.min() < 0.02

    def test_sequence_level_accepted(self):
        torch.manual_seed(7)
        x = torch.randn(4, 16)
        mask = torch.ones(4, 16, dtype=torch.bool)
        corrupted, _, _ = corrupt_time_delta(
            x, mask, sampling_level="sequence"
        )
        assert corrupted.shape == x.shape


class TestCorruptTimeDeltaDeterminism:
    def test_deterministic_with_seed(self):
        x = torch.randn(4, 12)
        mask = torch.ones(4, 12, dtype=torch.bool)
        torch.manual_seed(42)
        r1 = corrupt_time_delta(x, mask)
        torch.manual_seed(42)
        r2 = corrupt_time_delta(x, mask)
        assert (r1[0] == r2[0]).all()
        assert (r1[1] == r2[1]).all()


class TestCorruptTimeDeltaValidation:
    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            corrupt_time_delta(
                torch.randn(4, 12), torch.ones(4, 10, dtype=torch.bool)
            )

    def test_invalid_sampling_level_raises(self):
        with pytest.raises(AssertionError, match="sampling_level"):
            corrupt_time_delta(
                torch.randn(4, 12), torch.ones(4, 12, dtype=torch.bool),
                sampling_level="token",
            )

    def test_min_std_greater_than_max_std_raises(self):
        with pytest.raises(AssertionError):
            corrupt_time_delta(
                torch.randn(4, 12), torch.ones(4, 12, dtype=torch.bool),
                min_std=0.30, max_std=0.05,
            )


# ===========================================================================
# corrupt_numerical_features
# ===========================================================================

class TestCorruptNumericalShapes:
    def test_output_shapes(self):
        torch.manual_seed(10)
        B, L, N = 4, 12, 5
        x = torch.randn(B, L, N)
        mask = torch.ones(B, L, dtype=torch.bool)
        corrupted, num_mask, original = corrupt_numerical_features(x, mask)
        assert corrupted.shape == (B, L, N)
        assert num_mask.shape == (B, L)
        assert original.shape == (B, L, N)

    def test_dtypes(self):
        x = torch.randn(3, 10, 4)
        mask = torch.ones(3, 10, dtype=torch.bool)
        corrupted, num_mask, original = corrupt_numerical_features(x, mask)
        assert corrupted.dtype == torch.float32
        assert num_mask.dtype == torch.bool
        assert original.dtype == torch.float32

    def test_original_unchanged(self):
        x = torch.randn(4, 8, 3)
        mask = torch.ones(4, 8, dtype=torch.bool)
        _, _, original = corrupt_numerical_features(x, mask)
        assert (original == x).all()


class TestCorruptNumericalPadding:
    def test_padding_not_corrupted(self):
        torch.manual_seed(11)
        B, L, N = 4, 12, 3
        x = torch.randn(B, L, N)
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        attention_mask[1, 9:] = False
        x[1, 9:] = 0.0

        corrupted, num_mask, original = corrupt_numerical_features(x, attention_mask)
        pad_exp = ~attention_mask.unsqueeze(-1).expand(B, L, N)
        assert (corrupted[pad_exp] == original[pad_exp]).all()
        assert num_mask[~attention_mask].sum() == 0

    def test_all_padding_returns_unchanged(self):
        x = torch.randn(3, 8, 4)
        mask = torch.zeros(3, 8, dtype=torch.bool)
        corrupted, num_mask, original = corrupt_numerical_features(x, mask)
        assert (corrupted == original).all()
        assert num_mask.sum() == 0


class TestCorruptNumericalNoise:
    def test_all_features_noised_at_selected_position(self):
        torch.manual_seed(12)
        B, L, N = 8, 16, 6
        x = torch.randn(B, L, N)
        mask = torch.ones(B, L, dtype=torch.bool)
        corrupted, num_mask, original = corrupt_numerical_features(
            x, mask, corruption_prob=1.0
        )
        # Every feature at every selected position must differ (noise is non-zero a.s.)
        selected_exp = num_mask.unsqueeze(-1).expand(B, L, N)
        assert not (corrupted[selected_exp] == original[selected_exp]).all()

    def test_uncorrupted_positions_unchanged(self):
        torch.manual_seed(13)
        B, L, N = 8, 16, 4
        x = torch.randn(B, L, N)
        mask = torch.ones(B, L, dtype=torch.bool)
        corrupted, num_mask, original = corrupt_numerical_features(x, mask)
        unsel = ~num_mask.unsqueeze(-1).expand(B, L, N)
        assert (corrupted[unsel] == original[unsel]).all()

    def test_mask_is_per_position_not_per_feature(self):
        # num_mask is [B, L], not [B, L, N]: same position selected for all features
        torch.manual_seed(14)
        x = torch.randn(4, 12, 5)
        mask = torch.ones(4, 12, dtype=torch.bool)
        _, num_mask, _ = corrupt_numerical_features(x, mask)
        assert num_mask.shape == (4, 12)


class TestCorruptNumericalCalibration:
    def test_corruption_rate(self):
        torch.manual_seed(15)
        B, L, N = 256, 128, 3
        x = torch.randn(B, L, N)
        mask = torch.ones(B, L, dtype=torch.bool)
        corruption_prob = 0.20
        _, num_mask, _ = corrupt_numerical_features(
            x, mask, corruption_prob=corruption_prob
        )
        actual = num_mask.float().mean().item()
        assert abs(actual - corruption_prob) < 0.02, (
            f"corruption rate {actual:.4f} too far from {corruption_prob}"
        )


class TestCorruptNumericalDeterminism:
    def test_deterministic_with_seed(self):
        x = torch.randn(4, 12, 3)
        mask = torch.ones(4, 12, dtype=torch.bool)
        torch.manual_seed(99)
        r1 = corrupt_numerical_features(x, mask)
        torch.manual_seed(99)
        r2 = corrupt_numerical_features(x, mask)
        assert (r1[0] == r2[0]).all()
        assert (r1[1] == r2[1]).all()


class TestCorruptNumericalValidation:
    def test_wrong_rank_raises(self):
        with pytest.raises(AssertionError, match="rank-3"):
            corrupt_numerical_features(
                torch.randn(4, 12), torch.ones(4, 12, dtype=torch.bool)
            )

    def test_mask_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            corrupt_numerical_features(
                torch.randn(4, 12, 3), torch.ones(4, 10, dtype=torch.bool)
            )

    def test_invalid_sampling_level_raises(self):
        with pytest.raises(AssertionError, match="sampling_level"):
            corrupt_numerical_features(
                torch.randn(4, 12, 3), torch.ones(4, 12, dtype=torch.bool),
                sampling_level="feature",
            )


# ===========================================================================
# mask_whole_events
# ===========================================================================

def _make_batch(B=4, L=12, N_num=3, N_cat=2, seed=0):
    torch.manual_seed(seed)
    return {
        "event_type": torch.randint(5, 30, (B, L)),
        "time_delta": torch.randn(B, L).abs(),
        "num_features": torch.randn(B, L, N_num),
        "cat_features": torch.randint(5, 20, (B, L, N_cat)),
    }


def _make_mask(B=4, L=12, pad_row=2, pad_from=9):
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[pad_row, pad_from:] = False
    return mask


class TestMaskWholeEventsShapes:
    def test_output_keys_preserved(self):
        batch = _make_batch()
        mask = _make_mask()
        out, event_mask = mask_whole_events(batch, mask)
        assert set(out.keys()) == set(batch.keys())

    def test_event_mask_shape_and_dtype(self):
        batch = _make_batch()
        attention_mask = _make_mask()
        _, event_mask = mask_whole_events(batch, attention_mask)
        assert event_mask.shape == attention_mask.shape
        assert event_mask.dtype == torch.bool

    def test_output_tensor_shapes_unchanged(self):
        batch = _make_batch()
        attention_mask = _make_mask()
        out, _ = mask_whole_events(batch, attention_mask)
        for key in batch:
            assert out[key].shape == batch[key].shape

    def test_non_tensor_values_passed_through(self):
        batch = _make_batch()
        batch["entity_id"] = ["a", "b", "c", "d"]
        attention_mask = _make_mask()
        out, _ = mask_whole_events(batch, attention_mask)
        assert out["entity_id"] == ["a", "b", "c", "d"]


class TestMaskWholeEventsPadding:
    def test_padding_positions_never_masked(self):
        batch = _make_batch()
        attention_mask = _make_mask()
        _, event_mask = mask_whole_events(batch, attention_mask)
        assert event_mask[~attention_mask].sum() == 0

    def test_padding_values_unchanged(self):
        torch.manual_seed(1)
        batch = _make_batch()
        attention_mask = _make_mask()
        # Zero out all padding slots
        for key in ("event_type", "time_delta"):
            batch[key][~attention_mask] = 0
        out, _ = mask_whole_events(batch, attention_mask)
        assert (out["event_type"][~attention_mask] == 0).all()
        assert (out["time_delta"][~attention_mask] == 0.0).all()

    def test_all_padding_produces_no_masked_events(self):
        batch = _make_batch()
        attention_mask = torch.zeros(4, 12, dtype=torch.bool)
        out, event_mask = mask_whole_events(batch, attention_mask)
        assert event_mask.sum() == 0
        # Batch tensors unchanged
        assert (out["event_type"] == batch["event_type"]).all()


class TestMaskWholeEventsFillValues:
    def test_event_type_filled_with_mask_event(self):
        torch.manual_seed(2)
        batch = _make_batch()
        attention_mask = torch.ones(4, 12, dtype=torch.bool)
        out, event_mask = mask_whole_events(
            batch, attention_mask, event_mask_prob=1.0
        )
        assert (out["event_type"][event_mask] == 4).all()

    def test_time_delta_filled_with_zero(self):
        torch.manual_seed(3)
        batch = _make_batch()
        attention_mask = torch.ones(4, 12, dtype=torch.bool)
        out, event_mask = mask_whole_events(
            batch, attention_mask, event_mask_prob=1.0
        )
        assert (out["time_delta"][event_mask] == 0.0).all()

    def test_num_features_filled_with_zero(self):
        torch.manual_seed(4)
        batch = _make_batch()
        attention_mask = torch.ones(4, 12, dtype=torch.bool)
        out, event_mask = mask_whole_events(
            batch, attention_mask, event_mask_prob=1.0
        )
        assert (out["num_features"][event_mask] == 0.0).all()

    def test_cat_features_filled_with_mask_cat(self):
        torch.manual_seed(5)
        batch = _make_batch()
        attention_mask = torch.ones(4, 12, dtype=torch.bool)
        out, event_mask = mask_whole_events(
            batch, attention_mask, event_mask_prob=1.0
        )
        assert (out["cat_features"][event_mask] == 3).all()

    def test_custom_mask_tokens_override(self):
        torch.manual_seed(6)
        batch = _make_batch()
        attention_mask = torch.ones(4, 12, dtype=torch.bool)
        out, event_mask = mask_whole_events(
            batch, attention_mask,
            event_mask_prob=1.0,
            mask_tokens={"event_type": 99, "time_delta": -1.0},
        )
        assert (out["event_type"][event_mask] == 99).all()
        assert (out["time_delta"][event_mask] == -1.0).all()
        # Unoverridden keys use defaults
        assert (out["cat_features"][event_mask] == 3).all()


class TestMaskWholeEventsUnmaskedPositions:
    def test_unmasked_positions_unchanged(self):
        torch.manual_seed(7)
        batch = _make_batch()
        attention_mask = torch.ones(4, 12, dtype=torch.bool)
        out, event_mask = mask_whole_events(batch, attention_mask)
        keep = ~event_mask
        assert (out["event_type"][keep] == batch["event_type"][keep]).all()
        assert (out["time_delta"][keep] == batch["time_delta"][keep]).all()
        assert (out["num_features"][keep] == batch["num_features"][keep]).all()
        assert (out["cat_features"][keep] == batch["cat_features"][keep]).all()

    def test_input_batch_not_mutated(self):
        torch.manual_seed(8)
        batch = _make_batch()
        originals = {k: v.clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        attention_mask = torch.ones(4, 12, dtype=torch.bool)
        mask_whole_events(batch, attention_mask, event_mask_prob=0.5)
        for key, orig in originals.items():
            assert (batch[key] == orig).all(), f"Input batch['{key}'] was mutated"


class TestMaskWholeEventsCalibration:
    def test_masking_rate(self):
        torch.manual_seed(9)
        B, L = 256, 128
        batch = {
            "event_type": torch.randint(5, 50, (B, L)),
            "time_delta": torch.randn(B, L),
        }
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        event_mask_prob = 0.10
        _, event_mask = mask_whole_events(batch, attention_mask, event_mask_prob=event_mask_prob)
        actual = event_mask.float().mean().item()
        assert abs(actual - event_mask_prob) < 0.02, (
            f"masking rate {actual:.4f} too far from {event_mask_prob}"
        )


class TestMaskWholeEventsDeterminism:
    def test_deterministic_with_seed(self):
        batch = _make_batch()
        attention_mask = _make_mask()
        torch.manual_seed(42)
        _, m1 = mask_whole_events(batch, attention_mask)
        torch.manual_seed(42)
        _, m2 = mask_whole_events(batch, attention_mask)
        assert (m1 == m2).all()


class TestMaskWholeEventsEdgeCases:
    def test_prob_zero_masks_nothing(self):
        batch = _make_batch()
        attention_mask = torch.ones(4, 12, dtype=torch.bool)
        out, event_mask = mask_whole_events(batch, attention_mask, event_mask_prob=0.0)
        assert event_mask.sum() == 0
        assert (out["event_type"] == batch["event_type"]).all()

    def test_prob_one_masks_all_real(self):
        torch.manual_seed(10)
        batch = _make_batch()
        attention_mask = _make_mask()
        out, event_mask = mask_whole_events(batch, attention_mask, event_mask_prob=1.0)
        # All real positions must be masked
        assert (event_mask == attention_mask).all()

    def test_batch_with_missing_keys(self):
        # Only event_type present — other keys absent, should not crash
        torch.manual_seed(11)
        batch = {"event_type": torch.randint(5, 20, (4, 12))}
        attention_mask = torch.ones(4, 12, dtype=torch.bool)
        out, event_mask = mask_whole_events(batch, attention_mask, event_mask_prob=0.5)
        assert "event_type" in out
        assert "time_delta" not in out

    def test_invalid_prob_raises(self):
        batch = _make_batch()
        attention_mask = torch.ones(4, 12, dtype=torch.bool)
        with pytest.raises(AssertionError):
            mask_whole_events(batch, attention_mask, event_mask_prob=1.5)


# ===========================================================================
# CorruptionPipeline
# ===========================================================================

from src.corruption.pipeline import CorruptionPipeline  # noqa: E402


def _base_config(**overrides) -> dict:
    """Minimal corruption config matching base.yaml defaults."""
    cfg = {
        "event_level_masking": {"prob": 0.10},
        "event_type": {
            "selected_prob": 0.40,
            "mask_prob": 0.28,
            "transition_replace_prob": 0.00,
            "random_replace_prob": 0.10,
            "keep_predict_prob": 0.02,
            "use_transition_aware_replacement": False,
        },
        "categorical_features": {"mask_prob": 0.15, "random_replace_prob": 0.05},
        "time_noise": {
            "corruption_prob": 0.30,
            "min_std": 0.05,
            "max_std": 0.30,
            "sampling_level": "batch",
        },
        "numerical_noise": {
            "corruption_prob": 0.20,
            "min_std": 0.03,
            "max_std": 0.15,
            "sampling_level": "batch",
        },
    }
    cfg.update(overrides)
    return cfg


def _full_batch(B=4, L=12, N_num=3, N_cat=2, seed=0):
    torch.manual_seed(seed)
    attention_mask = torch.ones(B, L, dtype=torch.bool)
    attention_mask[2, 9:] = False
    event_type = torch.randint(5, 50, (B, L))
    time_delta = torch.randn(B, L).abs()
    num_features = torch.randn(B, L, N_num)
    cat_features = torch.randint(5, 20, (B, L, N_cat))
    # Zero-out padding positions — matches what collate_fn produces
    event_type[~attention_mask] = 0
    time_delta[~attention_mask] = 0.0
    num_features[~attention_mask] = 0.0
    cat_features[~attention_mask] = 0
    return {
        "event_type": event_type,
        "time_delta": time_delta,
        "num_features": num_features,
        "cat_features": cat_features,
        "attention_mask": attention_mask,
        "label": torch.randint(0, 2, (B,)),
    }


class TestPipelineOutputStructure:
    def test_returns_three_dicts(self):
        batch = _full_batch()
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        result = pipe(batch)
        assert len(result) == 3

    def test_corrupted_has_same_keys(self):
        batch = _full_batch()
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        corrupted, _, _ = pipe(batch)
        assert set(corrupted.keys()) == set(batch.keys())

    def test_targets_keys(self):
        batch = _full_batch()
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        _, targets, _ = pipe(batch)
        assert set(targets.keys()) >= {"event_type", "time_delta", "num_features", "cat_features"}

    def test_masks_keys(self):
        batch = _full_batch()
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        _, _, masks = pipe(batch)
        assert set(masks.keys()) >= {
            "event_type", "time_delta", "num_features", "cat_features", "event_level"
        }

    def test_tensor_shapes_preserved(self):
        batch = _full_batch()
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        corrupted, targets, masks = pipe(batch)
        for key in ("event_type", "time_delta", "num_features", "cat_features"):
            assert corrupted[key].shape == batch[key].shape
            assert targets[key].shape == batch[key].shape

    def test_mask_shapes(self):
        batch = _full_batch()
        B, L = batch["attention_mask"].shape
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        _, _, masks = pipe(batch)
        assert masks["event_level"].shape == (B, L)
        assert masks["event_type"].shape == (B, L)
        assert masks["time_delta"].shape == (B, L)
        assert masks["num_features"].shape == (B, L)
        # cat_features mask is [B, L, N_cat]
        assert masks["cat_features"].shape == batch["cat_features"].shape


class TestPipelinePaddingInvariant:
    def test_padding_not_corrupted(self):
        torch.manual_seed(1)
        batch = _full_batch()
        attention_mask = batch["attention_mask"]
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        corrupted, _, _ = pipe(batch)

        pad = ~attention_mask
        assert (corrupted["event_type"][pad] == batch["event_type"][pad]).all()
        assert (corrupted["time_delta"][pad] == batch["time_delta"][pad]).all()
        pad3 = pad.unsqueeze(-1).expand_as(batch["num_features"])
        assert (corrupted["num_features"][pad3] == batch["num_features"][pad3]).all()

    def test_event_level_mask_not_set_on_padding(self):
        batch = _full_batch()
        attention_mask = batch["attention_mask"]
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        _, _, masks = pipe(batch)
        assert masks["event_level"][~attention_mask].sum() == 0


class TestPipelineInputNotMutated:
    def test_clean_batch_unchanged(self):
        batch = _full_batch()
        originals = {k: v.clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        pipe(batch)
        for key, orig in originals.items():
            assert (batch[key] == orig).all(), f"clean_batch['{key}'] was mutated"


class TestPipelineTargetsAreOriginalValues:
    def test_targets_equal_clean_batch(self):
        """targets must hold pre-corruption values."""
        torch.manual_seed(2)
        batch = _full_batch()
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        _, targets, _ = pipe(batch)
        assert (targets["event_type"] == batch["event_type"]).all()
        assert (targets["time_delta"] == batch["time_delta"]).all()
        assert (targets["num_features"] == batch["num_features"]).all()
        assert (targets["cat_features"] == batch["cat_features"]).all()


class TestPipelineEventLevelMaskingIsFirst:
    def test_event_level_positions_have_mask_event_type(self):
        """Positions selected by event-level masking must have MASK_EVENT (id=4) in corrupted."""
        torch.manual_seed(3)
        batch = _full_batch()
        # Force event-level masking to 100%, all other corruption off
        cfg = _base_config()
        cfg["event_level_masking"]["prob"] = 1.0
        cfg["event_type"]["selected_prob"] = 0.0
        cfg["event_type"]["mask_prob"] = 0.0
        cfg["event_type"]["random_replace_prob"] = 0.0
        cfg["event_type"]["keep_predict_prob"] = 0.0
        cfg["time_noise"]["corruption_prob"] = 0.0
        cfg["numerical_noise"]["corruption_prob"] = 0.0
        cfg["categorical_features"]["mask_prob"] = 0.0
        cfg["categorical_features"]["random_replace_prob"] = 0.0

        pipe = CorruptionPipeline(cfg, vocab_sizes={"event_type": 50})
        corrupted, _, masks = pipe(batch)

        real = batch["attention_mask"]
        assert (masks["event_level"] == real).all()
        assert (corrupted["event_type"][real] == 4).all()
        assert (corrupted["time_delta"][real] == 0.0).all()


class TestPipelineSkipsAbsentKeys:
    def test_batch_without_num_features(self):
        batch = _full_batch()
        del batch["num_features"]
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        corrupted, targets, masks = pipe(batch)
        assert "num_features" not in corrupted
        assert "num_features" not in targets

    def test_batch_without_cat_features(self):
        batch = _full_batch()
        del batch["cat_features"]
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50})
        corrupted, targets, masks = pipe(batch)
        assert "cat_features" not in corrupted
        assert "cat_features" not in targets


class TestPipelineZeroCorruptionProb:
    def test_time_delta_unchanged_when_prob_zero(self):
        torch.manual_seed(4)
        batch = _full_batch()
        cfg = _base_config()
        cfg["time_noise"]["corruption_prob"] = 0.0
        cfg["event_level_masking"]["prob"] = 0.0  # isolate: only time noise disabled
        pipe = CorruptionPipeline(cfg, vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        corrupted, _, masks = pipe(batch)
        assert (corrupted["time_delta"] == batch["time_delta"]).all()
        assert masks["time_delta"].sum() == 0

    def test_num_features_unchanged_when_prob_zero(self):
        torch.manual_seed(5)
        batch = _full_batch()
        cfg = _base_config()
        cfg["numerical_noise"]["corruption_prob"] = 0.0
        cfg["event_level_masking"]["prob"] = 0.0  # isolate: only numerical noise disabled
        pipe = CorruptionPipeline(cfg, vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        corrupted, _, masks = pipe(batch)
        assert (corrupted["num_features"] == batch["num_features"]).all()
        assert masks["num_features"].sum() == 0


class TestPipelineNonTensorPassThrough:
    def test_label_and_entity_id_preserved(self):
        batch = _full_batch()
        batch["entity_id"] = ["a", "b", "c", "d"]
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        corrupted, _, _ = pipe(batch)
        assert (corrupted["label"] == batch["label"]).all()
        assert corrupted["entity_id"] == ["a", "b", "c", "d"]


class TestPipelineDeterminism:
    def test_deterministic_with_seed(self):
        batch = _full_batch()
        pipe = CorruptionPipeline(_base_config(), vocab_sizes={"event_type": 50, "cat_features": [20, 20]})
        torch.manual_seed(77)
        c1, t1, m1 = pipe(batch)
        torch.manual_seed(77)
        c2, t2, m2 = pipe(batch)
        for key in ("event_type", "time_delta", "num_features", "cat_features"):
            assert (c1[key] == c2[key]).all(), f"corrupted['{key}'] not deterministic"
            assert (m1[key] == m2[key]).all(), f"masks['{key}'] not deterministic"
        assert (m1["event_level"] == m2["event_level"]).all()


# ===========================================================================
# Specification tests  (B=16, L=32, vocab_size=20)
# ===========================================================================


class _MockTransitionMatrix:
    """Deterministic transition for mixture-proportion testing.

    Maps token i → (i − _NUM_SPECIAL + 1) % n_valid + _NUM_SPECIAL.
    Proof that output ≠ original:
      j = (i−5+1)%15+5;  j==i  ↔  (i−4)%15==i−5  ↔  1≡0 (mod 15)  — impossible.
    Proof that output ≠ mask_token_id(=2):
      j ∈ [5, 19], mask_token_id=2 ∉ [5, 19].
    """

    def __init__(self, vocab_size: int) -> None:
        self._n_valid = vocab_size - _NUM_SPECIAL
        self.n_replaced = 0

    def sample_replacement_batch(
        self, event_types: torch.LongTensor, mask: torch.BoolTensor
    ) -> torch.LongTensor:
        result = event_types.clone()
        selected = event_types[mask].long()
        replaced = (selected - _NUM_SPECIAL + 1) % self._n_valid + _NUM_SPECIAL
        result[mask] = replaced
        self.n_replaced += int(mask.sum().item())
        return result


def test_event_type_selected_prob():
    """Mean fraction of selected positions over 1000 batches should be 0.40 ± 0.03."""
    B, L, vocab_size = 16, 32, 20
    torch.manual_seed(0)
    attention_mask = torch.ones(B, L, dtype=torch.bool)
    event_type = torch.randint(_NUM_SPECIAL, vocab_size, (B, L))
    n_real = attention_mask.sum().item()

    rates = []
    for _ in range(1000):
        _, pred_mask, _ = corrupt_event_type(
            event_type, attention_mask,
            selected_prob=0.40,
            mask_prob=0.28,
            transition_replace_prob=0.08,
            random_replace_prob=0.02,
            keep_predict_prob=0.02,
            vocab_size=vocab_size,
        )
        rates.append(pred_mask.sum().item() / n_real)

    mean_rate = sum(rates) / len(rates)
    assert abs(mean_rate - 0.40) < 0.03, (
        f"Mean selected_prob {mean_rate:.4f} not in 0.40 ± 0.03"
    )


def test_event_type_mixture():
    """Among selected positions, operation fractions match configured probabilities."""
    B, L, vocab_size = 16, 32, 20
    torch.manual_seed(1)
    attention_mask = torch.ones(B, L, dtype=torch.bool)

    n_sel = n_mask = n_trans = n_keep = 0

    for _ in range(1000):
        event_type = torch.randint(_NUM_SPECIAL, vocab_size, (B, L))
        tm = _MockTransitionMatrix(vocab_size)

        corrupted, pred_mask, original = corrupt_event_type(
            event_type, attention_mask,
            selected_prob=0.40,
            mask_prob=0.28,
            transition_replace_prob=0.08,
            random_replace_prob=0.02,
            keep_predict_prob=0.02,
            mask_token_id=2,
            vocab_size=vocab_size,
            transition_matrix=tm,
        )

        n_sel  += pred_mask.sum().item()
        n_mask += ((corrupted == 2) & pred_mask).sum().item()
        n_trans += tm.n_replaced
        n_keep += ((corrupted == original) & pred_mask).sum().item()

    # Remaining = random (decomposition is exhaustive and mutually exclusive)
    n_rand = n_sel - n_mask - n_trans - n_keep

    f_mask  = n_mask  / n_sel
    f_trans = n_trans / n_sel
    f_rand  = n_rand  / n_sel
    f_keep  = n_keep  / n_sel

    assert abs(f_mask  - 0.70) < 0.05, f"MASK  {f_mask:.4f} ≠ 0.70 ± 0.05"
    assert abs(f_trans - 0.20) < 0.05, f"trans {f_trans:.4f} ≠ 0.20 ± 0.05"
    assert abs(f_rand  - 0.05) < 0.03, f"rand  {f_rand:.4f} ≠ 0.05 ± 0.03"
    assert abs(f_keep  - 0.05) < 0.03, f"keep  {f_keep:.4f} ≠ 0.05 ± 0.03"


def test_no_padding_corruption():
    """Padding positions (attention_mask=False) must never change in any corruption function."""
    B, L, vocab_size = 16, 32, 20
    torch.manual_seed(2)

    attention_mask = torch.ones(B, L, dtype=torch.bool)
    attention_mask[:, -8:] = False  # last 8 positions are padding

    event_type   = torch.randint(_NUM_SPECIAL, vocab_size, (B, L))
    time_delta   = torch.randn(B, L).abs()
    num_features = torch.randn(B, L, 3)
    cat_features = torch.randint(_NUM_SPECIAL, vocab_size, (B, L, 2))

    # Mirror what collate_fn does: zero-fill padding
    event_type[~attention_mask]   = 0
    time_delta[~attention_mask]   = 0.0
    num_features[~attention_mask] = 0.0
    cat_features[~attention_mask] = 0

    pad   = ~attention_mask
    pad3n = pad.unsqueeze(-1).expand_as(num_features)
    pad3c = pad.unsqueeze(-1).expand_as(cat_features)

    for _ in range(20):
        c_et, _, _ = corrupt_event_type(
            event_type, attention_mask,
            selected_prob=0.40, mask_prob=0.28,
            transition_replace_prob=0.08, random_replace_prob=0.02,
            keep_predict_prob=0.02, vocab_size=vocab_size,
        )
        assert (c_et[pad] == event_type[pad]).all(), "event_type: padding mutated"

        c_td, _, _ = corrupt_time_delta(
            time_delta, attention_mask, corruption_prob=0.30
        )
        assert (c_td[pad] == time_delta[pad]).all(), "time_delta: padding mutated"

        c_nf, _, _ = corrupt_numerical_features(
            num_features, attention_mask, corruption_prob=0.20
        )
        assert (c_nf[pad3n] == num_features[pad3n]).all(), "num_features: padding mutated"

        c_cf, _, _ = corrupt_categorical_features(
            cat_features, attention_mask,
            vocab_sizes=[vocab_size, vocab_size],
        )
        assert (c_cf[pad3c] == cat_features[pad3c]).all(), "cat_features: padding mutated"


def test_time_batch_sigma():
    """With sampling_level='batch', sigma is shared across all sequences.

    Within a batch, per-row noise std estimates are all drawn from the same sigma,
    so their spread (max−min) is driven only by estimation noise.
    With sampling_level='sequence', each row has its own independent sigma,
    producing a much larger spread.
    """
    B, L = 16, 32
    x    = torch.zeros(B, L)
    mask = torch.ones(B, L, dtype=torch.bool)

    n_runs = 200
    batch_spreads: list[float] = []
    seq_spreads:   list[float] = []

    for _ in range(n_runs):
        for level, collector in [("batch", batch_spreads), ("sequence", seq_spreads)]:
            corrupted, _, _ = corrupt_time_delta(
                x, mask,
                corruption_prob=1.0,
                min_std=0.05, max_std=0.30,
                sampling_level=level,
            )
            row_stds = corrupted.std(dim=1)  # [B] — one sigma estimate per row
            collector.append((row_stds.max() - row_stds.min()).item())

    mean_batch = sum(batch_spreads) / n_runs
    mean_seq   = sum(seq_spreads)   / n_runs

    # Batch: shared sigma → within-batch spread ≈ estimation noise only (~0.08)
    # Sequence: independent sigmas U(0.05, 0.30) → spread ≈ 0.21
    assert mean_batch < mean_seq, (
        f"batch spread {mean_batch:.4f} should be < sequence spread {mean_seq:.4f}"
    )
    assert mean_batch < 0.15, (
        f"Batch sigma inconsistent within batch: mean spread = {mean_batch:.4f}"
    )


def test_masks_not_in_batch():
    """corrupted_batch must not contain any mask or internal corruption keys."""
    batch = _full_batch()
    pipe  = CorruptionPipeline(
        _base_config(),
        vocab_sizes={"event_type": 50, "cat_features": [20, 20]},
    )
    corrupted, _, _ = pipe(batch)

    forbidden = {
        "corruption_mask", "prediction_mask",
        "event_level_mask", "event_level",
        "time_mask", "num_mask", "cat_mask",
    }
    overlap = forbidden & set(corrupted.keys())
    assert not overlap, f"corrupted_batch contains mask keys: {overlap}"

    # corrupted_batch must have exactly the same keys as clean_batch
    assert set(corrupted.keys()) == set(batch.keys())


def test_pipeline_shapes():
    """CorruptionPipeline returns tensors with the same shapes as the input batch."""
    B, L, vocab_size = 16, 32, 20
    torch.manual_seed(3)

    attention_mask = torch.ones(B, L, dtype=torch.bool)
    attention_mask[2, 20:] = False

    batch = {
        "event_type":   torch.randint(_NUM_SPECIAL, vocab_size, (B, L)),
        "time_delta":   torch.randn(B, L).abs(),
        "num_features": torch.randn(B, L, 3),
        "cat_features": torch.randint(_NUM_SPECIAL, vocab_size, (B, L, 2)),
        "attention_mask": attention_mask,
        "label": torch.randint(0, 2, (B,)),
    }
    for key in ("event_type", "time_delta", "num_features", "cat_features"):
        batch[key][~attention_mask] = 0

    pipe = CorruptionPipeline(
        _base_config(),
        vocab_sizes={"event_type": vocab_size, "cat_features": [vocab_size, vocab_size]},
    )
    corrupted, targets, masks = pipe(batch)

    for key in ("event_type", "time_delta", "num_features", "cat_features"):
        assert corrupted[key].shape == batch[key].shape, f"corrupted['{key}'] shape mismatch"
        assert targets[key].shape   == batch[key].shape, f"targets['{key}'] shape mismatch"

    assert masks["event_level"].shape  == (B, L)
    assert masks["event_type"].shape   == (B, L)
    assert masks["time_delta"].shape   == (B, L)
    assert masks["num_features"].shape == (B, L)
    assert masks["cat_features"].shape == batch["cat_features"].shape


def test_padding_preserved():
    """After the full pipeline, padding positions are identical to clean_batch."""
    B, L, vocab_size = 16, 32, 20
    torch.manual_seed(4)

    attention_mask = torch.ones(B, L, dtype=torch.bool)
    attention_mask[3, 20:] = False
    attention_mask[7, 15:] = False

    batch = {
        "event_type":   torch.randint(_NUM_SPECIAL, vocab_size, (B, L)),
        "time_delta":   torch.randn(B, L).abs(),
        "num_features": torch.randn(B, L, 3),
        "cat_features": torch.randint(_NUM_SPECIAL, vocab_size, (B, L, 2)),
        "attention_mask": attention_mask,
        "label": torch.randint(0, 2, (B,)),
    }
    for key in ("event_type", "time_delta", "num_features", "cat_features"):
        batch[key][~attention_mask] = 0

    pipe = CorruptionPipeline(
        _base_config(),
        vocab_sizes={"event_type": vocab_size, "cat_features": [vocab_size, vocab_size]},
    )
    corrupted, _, _ = pipe(batch)

    pad = ~attention_mask
    assert (corrupted["event_type"][pad] == batch["event_type"][pad]).all()
    assert (corrupted["time_delta"][pad] == batch["time_delta"][pad]).all()

    pad3 = pad.unsqueeze(-1).expand_as(batch["num_features"])
    assert (corrupted["num_features"][pad3] == batch["num_features"][pad3]).all()

    pad3 = pad.unsqueeze(-1).expand_as(batch["cat_features"])
    assert (corrupted["cat_features"][pad3] == batch["cat_features"][pad3]).all()


def test_event_level_mask_not_overwritten():
    """Event-level masked positions must keep MASK_EVENT=4, not be overwritten by corrupt_event_type.

    With event_mask_prob=1.0, ALL real positions are event-level masked.
    corrupt_event_type and corrupt_categorical_features must skip those positions
    entirely via excluded_mask, so the token values set by mask_whole_events survive.
    """
    MASK_EVENT = 4
    MASK_TYPE = 2
    MASK_CAT = 3
    B, L = 4, 12

    attention_mask = torch.ones(B, L, dtype=torch.bool)
    batch = {
        "event_type": torch.randint(5, 50, (B, L)),
        "time_delta": torch.randn(B, L).abs(),
        "num_features": torch.randn(B, L, 2),
        "cat_features": torch.randint(5, 20, (B, L, 2)),
        "attention_mask": attention_mask,
        "label": torch.zeros(B, dtype=torch.long),
    }

    # prob=1.0 guarantees every real position is event-level masked deterministically
    cfg = _base_config()
    cfg["event_level_masking"]["prob"] = 1.0

    pipe = CorruptionPipeline(
        cfg,
        vocab_sizes={"event_type": 50, "cat_features": [20, 20]},
    )
    corrupted, _, masks = pipe(batch)

    # Every real position was event-level masked
    assert masks["event_level"].all(), "prob=1.0 must mask all real positions"

    # event_type: must stay MASK_EVENT=4, never be overwritten with MASK_TYPE=2
    assert (corrupted["event_type"] == MASK_EVENT).all(), (
        f"Event-level masked positions must keep MASK_EVENT={MASK_EVENT}, "
        f"not be overwritten by corrupt_event_type"
    )
    assert (corrupted["event_type"] != MASK_TYPE).all(), (
        f"MASK_TYPE={MASK_TYPE} must not appear when all positions are event-level masked"
    )

    # cat_features: must stay MASK_CAT=3
    assert (corrupted["cat_features"] == MASK_CAT).all(), (
        f"Event-level masked cat_features must keep MASK_CAT={MASK_CAT}, "
        f"not be overwritten by corrupt_categorical_features"
    )
