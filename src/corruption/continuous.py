from __future__ import annotations

import logging

import torch
from torch import BoolTensor, FloatTensor

logger = logging.getLogger(__name__)


def corrupt_time_delta(
    log_time_delta: FloatTensor,
    attention_mask: BoolTensor,
    corruption_prob: float = 0.30,
    min_std: float = 0.05,
    max_std: float = 0.30,
    sampling_level: str = "batch",
    already_log_transformed: bool = True,
) -> tuple[FloatTensor, BoolTensor, FloatTensor]:
    """Add Gaussian noise to log-scaled time deltas for denoising pretraining.

    sigma is sampled once per batch (sampling_level='batch') or once per
    sequence (sampling_level='sequence'), giving a curriculum-like effect
    where all corrupted positions in the same unit share the same noise scale.

    Args:
        log_time_delta: FloatTensor[B, L] — log-scaled inter-event times.
        attention_mask: BoolTensor[B, L] — True for real positions.
        corruption_prob: fraction of real positions to corrupt.
        min_std: lower bound of sigma uniform prior.
        max_std: upper bound of sigma uniform prior.
        sampling_level: 'batch' (one sigma for whole batch) or
                        'sequence' (one sigma per sequence in the batch).

    Returns:
        corrupted_log_delta: FloatTensor[B, L]
        time_mask: BoolTensor[B, L] — True for corrupted positions.
        original_log_delta: FloatTensor[B, L] — clean copy of input.
    """
    assert log_time_delta.shape == attention_mask.shape, (
        "log_time_delta and attention_mask must have identical shape"
    )
    assert 0.0 <= corruption_prob <= 1.0, "corruption_prob must be in [0, 1]"
    assert 0.0 < min_std <= max_std, "need 0 < min_std <= max_std"
    assert sampling_level in ("batch", "sequence"), (
        "sampling_level must be 'batch' or 'sequence'"
    )

    if not already_log_transformed and corruption_prob > 0.0:
        logger.warning(
            "corrupt_time_delta: corruption_prob=%.2f but already_log_transformed=False. "
            "Gaussian noise will be applied to raw (non-log) time deltas, which may "
            "produce out-of-distribution values. Set time_transform='log1p' or "
            "disable time noise for this dataset.",
            corruption_prob,
        )

    B, L = log_time_delta.shape
    device = log_time_delta.device
    original = log_time_delta.clone()

    # Sample sigma
    if sampling_level == "batch":
        sigma = torch.empty(1, device=device).uniform_(min_std, max_std)  # scalar-like
        sigma = sigma.expand(B, 1)  # broadcast over L later
    else:  # sequence: one sigma per row
        sigma = torch.empty(B, 1, device=device).uniform_(min_std, max_std)

    # Select positions to corrupt: Bernoulli over real positions only
    selected = (torch.rand(B, L, device=device) < corruption_prob) & attention_mask

    # Additive Gaussian noise: x_tilde = x + sigma * eps
    eps = torch.randn(B, L, device=device)
    noise = sigma * eps  # broadcasts [B,1] × [B,L] → [B,L]

    corrupted = torch.where(selected, original + noise, original)

    return corrupted, selected, original


def corrupt_numerical_features(
    num_features: FloatTensor,
    attention_mask: BoolTensor,
    corruption_prob: float = 0.20,
    min_std: float = 0.03,
    max_std: float = 0.15,
    sampling_level: str = "batch",
) -> tuple[FloatTensor, BoolTensor, FloatTensor]:
    """Add Gaussian noise to numerical features for denoising pretraining.

    sigma is sampled once per batch (or per sequence), then applied
    independently to every feature dimension at selected positions.

    Args:
        num_features: FloatTensor[B, L, N] — normalized numerical features.
        attention_mask: BoolTensor[B, L] — True for real positions.
        corruption_prob: fraction of real positions to corrupt.
        min_std: lower bound of sigma uniform prior.
        max_std: upper bound of sigma uniform prior.
        sampling_level: 'batch' (one sigma for whole batch) or
                        'sequence' (one sigma per sequence in the batch).

    Returns:
        corrupted_num: FloatTensor[B, L, N]
        num_mask: BoolTensor[B, L] — True for positions where noise was added.
        original_num: FloatTensor[B, L, N] — clean copy of input.
    """
    assert num_features.ndim == 3, "num_features must be rank-3 [B, L, N]"
    assert attention_mask.shape == num_features.shape[:2], (
        "attention_mask shape must match (B, L) of num_features"
    )
    assert 0.0 <= corruption_prob <= 1.0, "corruption_prob must be in [0, 1]"
    assert 0.0 < min_std <= max_std, "need 0 < min_std <= max_std"
    assert sampling_level in ("batch", "sequence"), (
        "sampling_level must be 'batch' or 'sequence'"
    )

    B, L, N = num_features.shape
    device = num_features.device
    original = num_features.clone()

    # Sample sigma
    if sampling_level == "batch":
        sigma = torch.empty(1, device=device).uniform_(min_std, max_std)
        sigma = sigma.expand(B, 1, 1)  # broadcast over L and N
    else:  # sequence
        sigma = torch.empty(B, 1, 1, device=device).uniform_(min_std, max_std)

    # Select positions: [B, L] — same mask applied to all N features at a position
    selected = (torch.rand(B, L, device=device) < corruption_prob) & attention_mask

    # Expand mask to [B, L, N] for where()
    selected_expanded = selected.unsqueeze(-1).expand(B, L, N)

    eps = torch.randn(B, L, N, device=device)
    noise = sigma * eps  # [B,1,1] × [B,L,N] → [B,L,N]

    corrupted = torch.where(selected_expanded, original + noise, original)

    return corrupted, selected, original
