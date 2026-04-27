from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import BoolTensor, LongTensor

if TYPE_CHECKING:
    from src.corruption.transition_matrix import TransitionMatrix

# PAD=0, UNK=1, MASK_TYPE=2, MASK_CAT=3, MASK_EVENT=4 — reserved in all vocabs
_NUM_SPECIAL = 5


def _sample_random_tokens(
    original: LongTensor,
    vocab_size: int,
    low: int = _NUM_SPECIAL,
) -> LongTensor:
    """Sample a random valid token != original for every position.

    Uses the shift trick: sample r in [low, vocab_size-2], then r += 1 where r >= original.
    Result is uniform over [low, vocab_size-1] \\ {original}.
    Requires vocab_size >= low + 2.
    """
    r = torch.randint(low, vocab_size - 1, original.shape, device=original.device)
    r = torch.where(r >= original, r + 1, r)
    return r


def corrupt_event_type(
    event_type: LongTensor,
    attention_mask: BoolTensor,
    selected_prob: float = 0.40,
    mask_prob: float = 0.28,
    transition_replace_prob: float = 0.08,
    random_replace_prob: float = 0.02,
    keep_predict_prob: float = 0.02,
    mask_token_id: int = 2,
    vocab_size: int = 100,
    transition_matrix: TransitionMatrix | None = None,
    pad_token_id: int = 0,
    excluded_mask: BoolTensor | None = None,
) -> tuple[LongTensor, BoolTensor, LongTensor]:
    """Corrupt event-type tokens for denoising pretraining.

    Probabilities are absolute (not conditional on selection):
      mask_prob               = selected_prob × 0.70  (replace with MASK token)
      transition_replace_prob = selected_prob × 0.20  (transition-aware replacement)
      random_replace_prob     = selected_prob × 0.05  (uniform random valid token)
      keep_predict_prob       = selected_prob × 0.05  (keep original, still predict)

    Args:
        event_type: LongTensor[B, L] — token ids, padding has value pad_token_id.
        attention_mask: BoolTensor[B, L] — True for real positions, False for padding.
        selected_prob: total fraction of real positions included in prediction.
        mask_prob: absolute prob of MASK operation.
        transition_replace_prob: absolute prob of transition-aware replacement.
        random_replace_prob: absolute prob of random replacement.
        keep_predict_prob: absolute prob of keep-but-predict operation.
        mask_token_id: token id used as MASK (MASK_TYPE=2).
        vocab_size: total vocabulary size including special tokens.
        transition_matrix: fitted TransitionMatrix or None (fallback to random).
        pad_token_id: token id for padding positions.

    Returns:
        corrupted_event_type: LongTensor[B, L]
        prediction_mask: BoolTensor[B, L] — True for positions included in loss.
        original_event_type: LongTensor[B, L] — clean copy of input.
    """
    total = mask_prob + transition_replace_prob + random_replace_prob + keep_predict_prob
    assert abs(total - selected_prob) < 1e-6, (
        f"mask_prob + transition_replace_prob + random_replace_prob + keep_predict_prob "
        f"must equal selected_prob, got {total:.6f} vs {selected_prob:.6f}"
    )
    assert vocab_size > _NUM_SPECIAL, (
        f"vocab_size must be > {_NUM_SPECIAL} (number of reserved special tokens)"
    )
    assert event_type.shape == attention_mask.shape, (
        "event_type and attention_mask must have identical shape"
    )

    original = event_type.clone()
    corrupted = event_type.clone()

    # Positions eligible for corruption: real tokens not already event-level masked
    eligible = attention_mask
    if excluded_mask is not None:
        eligible = attention_mask & ~excluded_mask

    # Assign each position to one of five operations via cumsum thresholds on U[0,1)
    u = torch.rand(event_type.shape, device=event_type.device)
    t1 = mask_prob
    t2 = t1 + transition_replace_prob
    t3 = t2 + random_replace_prob
    t4 = t3 + keep_predict_prob  # == selected_prob

    is_mask = (u < t1) & eligible
    is_trans = (u >= t1) & (u < t2) & eligible
    is_random = (u >= t2) & (u < t3) & eligible
    is_keep = (u >= t3) & (u < t4) & eligible

    prediction_mask = is_mask | is_trans | is_random | is_keep

    # MASK operation
    corrupted = torch.where(is_mask, torch.full_like(corrupted, mask_token_id), corrupted)

    # TRANSITION operation — lazy guard avoids unnecessary CPU/GPU round-trip
    if is_trans.any():
        if transition_matrix is not None:
            # sample_replacement_batch returns clone(original) with replacements at is_trans
            trans_result = transition_matrix.sample_replacement_batch(original, is_trans)
            corrupted = torch.where(is_trans, trans_result, corrupted)
        else:
            rand_tokens = _sample_random_tokens(original, vocab_size)
            corrupted = torch.where(is_trans, rand_tokens, corrupted)

    # RANDOM operation
    if is_random.any():
        rand_tokens = _sample_random_tokens(original, vocab_size)
        corrupted = torch.where(is_random, rand_tokens, corrupted)

    # KEEP operation — no change needed; original already in corrupted

    # Reinforce padding: attention_mask guards all op masks, but be explicit
    corrupted = torch.where(
        attention_mask, corrupted, torch.full_like(corrupted, pad_token_id)
    )

    return corrupted, prediction_mask, original


def corrupt_categorical_features(
    cat_features: LongTensor,
    attention_mask: BoolTensor,
    mask_prob: float = 0.15,
    random_replace_prob: float = 0.05,
    mask_token_id: int = 3,
    vocab_sizes: list[int] | None = None,
    excluded_mask: BoolTensor | None = None,
) -> tuple[LongTensor, BoolTensor, LongTensor]:
    """Corrupt categorical features for denoising pretraining.

    For each position (b, l, j):
      mask_prob           → replace with mask_token_id
      random_replace_prob → replace with a random valid token for feature j
      remaining           → keep unchanged (not included in prediction)

    Operations are mutually exclusive (sampled from the same U[0,1)).

    Args:
        cat_features: LongTensor[B, L, N_cat] — categorical feature token ids.
        attention_mask: BoolTensor[B, L] — True for real positions, False for padding.
        mask_prob: fraction of real positions to mask per feature.
        random_replace_prob: fraction of real positions to randomly replace per feature.
        mask_token_id: token id used as MASK (MASK_CAT=3).
        vocab_sizes: list of vocab sizes per feature (length N_cat).
                     If None, random replacement is skipped (BERT-keep fallback).

    Returns:
        corrupted: LongTensor[B, L, N_cat]
        prediction_mask: BoolTensor[B, L, N_cat] — True for positions in loss.
        original: LongTensor[B, L, N_cat] — clean copy of input.
    """
    B, L, N_cat = cat_features.shape
    assert attention_mask.shape == (B, L), (
        f"attention_mask shape {attention_mask.shape} must match (B={B}, L={L})"
    )
    if vocab_sizes is not None:
        assert len(vocab_sizes) == N_cat, (
            f"len(vocab_sizes)={len(vocab_sizes)} must equal N_cat={N_cat}"
        )

    original = cat_features.clone()
    corrupted = cat_features.clone()

    u = torch.rand(B, L, N_cat, device=cat_features.device)

    # Positions eligible for corruption: real tokens not already event-level masked
    eligible = attention_mask
    if excluded_mask is not None:
        eligible = attention_mask & ~excluded_mask

    # Zero-copy broadcast: [B, L] → [B, L, N_cat]
    attn = eligible.unsqueeze(-1).expand(B, L, N_cat)

    is_mask = (u < mask_prob) & attn
    is_rand = (u >= mask_prob) & (u < mask_prob + random_replace_prob) & attn
    prediction_mask = is_mask | is_rand

    # MASK operation
    corrupted = torch.where(is_mask, torch.full_like(corrupted, mask_token_id), corrupted)

    # RANDOM operation — loop over N_cat (small) to respect per-feature vocab_sizes
    if is_rand.any() and vocab_sizes is not None:
        for j in range(N_cat):
            col_rand = is_rand[:, :, j]
            if not col_rand.any():
                continue
            rand_tokens = _sample_random_tokens(original[:, :, j], vocab_sizes[j])
            corrupted[:, :, j] = torch.where(col_rand, rand_tokens, corrupted[:, :, j])
    # vocab_sizes=None: is_rand positions retain original value (BERT-keep fallback);
    # prediction_mask still marks them so the loss is computed on those positions.

    return corrupted, prediction_mask, original
