from __future__ import annotations

import torch

from src.generation import (
    build_conditional_generation_batch,
    compute_generation_metrics,
    decode_generated_suffix,
    generate_suffix,
)
from src.models.dme_encoder import DMEEncoder

B, L = 2, 8

CONFIG = {
    "data": {"use_time_features": False},
    "model": {
        "hidden_dim": 32,
        "num_layers": 1,
        "num_heads": 4,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "activation": "gelu",
        "event_type_emb_dim": 16,
        "cat_emb_dim": 8,
        "num_projection_dim": 16,
        "time_projection_dim": 16,
        "max_seq_len": 16,
    },
    "pooling": {"type": "mean"},
    "pretraining": {"objective": "diffusion"},
    "diffusion": {"num_steps": 4, "beta_start": 1e-4, "beta_end": 2e-2},
    "generation": {
        "enabled": True,
        "suffix_len": 3,
        "sampler": "ddim_lite",
        "num_sampling_steps": 2,
        "temperature_event_type": 1.0,
        "temperature_cat": 1.0,
        "top_k_event_type": 5,
    },
}

VOCAB_INFO = {
    "event_type_vocab_size": 12,
    "cat_vocab_sizes": [9, 10],
    "num_num_features": 2,
    "num_classes": 2,
}


class FakePreprocessor:
    event_type_col = "event_type"
    categorical_cols = ["merchant", "channel"]
    numerical_cols = ["amount", "balance"]
    scaler_params = {
        "time_delta": {"center": 0.0, "scale": 1.0},
        "amount": {"center": 0.0, "scale": 1.0},
        "balance": {"center": 0.0, "scale": 1.0},
    }
    time_transform = "none"
    amount_transform = "robust_scaler"
    _amount_cols: set[str] = set()
    vocab = {
        "event_type": {"purchase": 5, "refund": 6, "view": 7},
        "merchant": {"m1": 5, "m2": 6},
        "channel": {"web": 5, "mobile": 6},
    }


def _make_clean_batch() -> dict:
    torch.manual_seed(0)
    attention_mask = torch.ones(B, L, dtype=torch.bool)
    return {
        "event_type": torch.tensor(
            [
                [5, 6, 7, 8, 9, 10, 11, 5],
                [6, 7, 8, 9, 10, 11, 5, 6],
            ],
            dtype=torch.long,
        ),
        "time_delta": torch.randn(B, L),
        "num_features": torch.randn(B, L, 2),
        "cat_features": torch.tensor(
            [
                [[5, 5], [6, 6], [7, 7], [8, 8], [5, 9], [6, 5], [7, 6], [8, 7]],
                [[6, 5], [7, 6], [8, 7], [5, 8], [6, 9], [7, 5], [8, 6], [5, 7]],
            ],
            dtype=torch.long,
        ),
        "attention_mask": attention_mask,
        "label": torch.zeros(B, dtype=torch.long),
        "entity_id": ["a", "b"],
    }


def test_build_conditional_generation_batch_targets_future_suffix() -> None:
    batch = _make_clean_batch()
    generation_seed = build_conditional_generation_batch(
        batch,
        CONFIG,
        suffix_len=3,
        prefix_lengths=torch.tensor([4, 5]),
    )

    gen_batch = generation_seed["batch"]
    prefix_mask = generation_seed["prefix_mask"]
    suffix_mask = generation_seed["suffix_mask"]
    target_mask = generation_seed["target_mask"]

    assert gen_batch["event_type"].shape == (B, 8)
    assert prefix_mask.sum().item() == 9
    assert suffix_mask.sum().item() == 6
    assert target_mask.sum().item() == 6
    assert torch.equal(gen_batch["event_type"][0, :4], batch["event_type"][0, :4])
    assert torch.equal(gen_batch["event_type"][1, :5], batch["event_type"][1, :5])
    assert torch.equal(generation_seed["targets"]["event_type"][0, 4:7], batch["event_type"][0, 4:7])
    assert torch.equal(generation_seed["targets"]["event_type"][1, 5:8], batch["event_type"][1, 5:8])


def test_generate_suffix_metrics_and_decode() -> None:
    torch.manual_seed(7)
    model = DMEEncoder(CONFIG, VOCAB_INFO)
    batch = _make_clean_batch()
    generation_seed = build_conditional_generation_batch(
        batch,
        CONFIG,
        suffix_len=3,
        prefix_lengths=torch.tensor([4, 5]),
    )

    generated = generate_suffix(model, generation_seed, CONFIG, VOCAB_INFO)
    gen_batch = generated["generated_batch"]
    prefix_mask = generated["prefix_mask"]
    suffix_mask = generated["suffix_mask"]

    assert torch.equal(
        gen_batch["event_type"][prefix_mask],
        generation_seed["batch"]["event_type"][prefix_mask],
    )
    assert int(gen_batch["event_type"][suffix_mask].min().item()) >= 5
    assert int(gen_batch["cat_features"][suffix_mask].min().item()) >= 5

    metrics = compute_generation_metrics(generated, VOCAB_INFO)
    assert metrics["suffix_positions"] == 6
    assert metrics["target_positions"] == 6
    assert metrics["invalid_event_token_rate"] == 0.0
    assert metrics["invalid_cat_token_rate"] == 0.0
    assert metrics["time_delta_mae_normalized"] >= 0.0

    rows = decode_generated_suffix(generated, FakePreprocessor(), CONFIG)
    assert len(rows) == 6
    assert {"entity_id", "sample_id", "step", "event_type_id", "time_delta"}.issubset(rows[0])
