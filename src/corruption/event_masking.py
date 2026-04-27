from __future__ import annotations

import torch
from torch import BoolTensor

# Default mask token ids — match preprocessing.py conventions
_MASK_EVENT_ID = 4
_MASK_CAT_ID = 3

_DEFAULT_MASK_TOKENS: dict = {
    "event_type": _MASK_EVENT_ID,
    "time_delta": 0.0,
    "num_features": 0.0,
    "cat_features": _MASK_CAT_ID,
}


def mask_whole_events(
    batch: dict,
    attention_mask: BoolTensor,
    event_mask_prob: float = 0.10,
    mask_tokens: dict | None = None,
) -> tuple[dict, BoolTensor]:
    """Replace every feature of selected events with mask values.

    Selected positions are drawn once per call via Bernoulli(event_mask_prob)
    restricted to real (non-padding) positions. Sequence length is unchanged.

    The returned event_level_mask is NOT passed to the encoder — it is used
    only to compute the optional existence_head loss.

    Args:
        batch: dict containing any subset of
               'event_type'   LongTensor[B, L]
               'time_delta'   FloatTensor[B, L]
               'num_features' FloatTensor[B, L, N]
               'cat_features' LongTensor[B, L, N_cat]
               Keys absent from the dict are silently skipped.
        attention_mask: BoolTensor[B, L] — True for real positions.
        event_mask_prob: fraction of real events to mask.
        mask_tokens: override mask fill values per key.
                     Defaults to {event_type: 4, time_delta: 0.0,
                     num_features: 0.0, cat_features: 3}.

    Returns:
        masked_batch: dict with same keys as batch, corrupted in-place copy.
        event_level_mask: BoolTensor[B, L] — True for masked events.
    """
    assert 0.0 <= event_mask_prob <= 1.0, "event_mask_prob must be in [0, 1]"

    tokens = {**_DEFAULT_MASK_TOKENS, **(mask_tokens or {})}

    device = attention_mask.device
    B, L = attention_mask.shape

    # Select whole events: Bernoulli over real positions only
    event_mask = (torch.rand(B, L, device=device) < event_mask_prob) & attention_mask

    masked_batch: dict = {}

    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            masked_batch[key] = value
            continue

        out = value.clone()
        fill = tokens.get(key)

        if fill is None or not event_mask.any():
            masked_batch[key] = out
            continue

        if value.ndim == 2:          # [B, L]
            out[event_mask] = fill
        elif value.ndim == 3:        # [B, L, *]
            out[event_mask] = fill   # broadcasts fill to all feature dims

        masked_batch[key] = out

    return masked_batch, event_mask
