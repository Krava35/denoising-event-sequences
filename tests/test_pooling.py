from __future__ import annotations

import pytest
import torch

from src.models.pooling import GatedAttentionPooling, get_pooling

B, L, H = 4, 12, 64


@pytest.mark.parametrize("pooling_type", ["cls", "mean", "max", "attention", "gated_attention"])
def test_pooling_output_shape(pooling_type: str) -> None:
    pool = get_pooling(pooling_type, H)
    x = torch.randn(B, L, H)
    mask = torch.ones(B, L, dtype=torch.bool)
    out = pool(x, mask)
    assert out.shape == (B, H)


@pytest.mark.parametrize("pooling_type", ["cls", "mean", "max", "attention", "gated_attention"])
def test_pooling_partial_mask(pooling_type: str) -> None:
    pool = get_pooling(pooling_type, H)
    x = torch.randn(B, L, H)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[1, 6:] = False  # вторая последовательность короче
    out = pool(x, mask)
    assert out.shape == (B, H)
    assert out.isfinite().all(), f"{pooling_type}: non-finite values with partial mask"


def test_gated_attention_different_lengths() -> None:
    pool = GatedAttentionPooling(H)
    pool.eval()
    torch.manual_seed(42)
    x = torch.randn(1, L, H)

    mask_short = torch.zeros(1, L, dtype=torch.bool)
    mask_short[0, :4] = True   # только первые 4 токена

    mask_long = torch.ones(1, L, dtype=torch.bool)  # все L токенов

    with torch.no_grad():
        out_short = pool(x, mask_short)
        out_long = pool(x, mask_long)

    assert not torch.allclose(out_short, out_long), (
        "GatedAttentionPooling должен давать разные результаты для разной длины"
    )


def test_get_pooling_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown pooling type"):
        get_pooling("unknown", H)
