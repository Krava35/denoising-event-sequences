from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.models.dme_encoder import DMEEncoder
from src.models.pooling import get_pooling
from src.models.tokenizer import MixedEventTokenizer
from src.models.transformer_encoder import TimeAwareTransformerEncoder

B, L = 4, 16

CONFIG = {
    "model": {
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dim_feedforward": 128,
        "dropout": 0.0,
        "activation": "gelu",
        "event_type_emb_dim": 32,
        "cat_emb_dim": 16,
        "num_projection_dim": 32,
        "time_projection_dim": 32,
        "max_seq_len": 64,
    },
    "pooling": {"type": "gated_attention"},
}

VOCAB_INFO = {
    "event_type_vocab_size": 20,
    "cat_vocab_sizes": [8, 6],
    "num_num_features": 3,
    "num_classes": 2,
}

H = CONFIG["model"]["hidden_dim"]


@pytest.fixture()
def batch() -> dict:
    torch.manual_seed(0)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, 12:] = False  # паддинг для первой последовательности
    return {
        "event_type": torch.randint(1, 20, (B, L)),
        "time_delta": torch.rand(B, L) * 100,
        "num_features": torch.randn(B, L, 3),
        "cat_features": torch.randint(1, 6, (B, L, 2)),
        "attention_mask": mask,
    }


@pytest.fixture()
def model() -> DMEEncoder:
    torch.manual_seed(0)
    return DMEEncoder(CONFIG, VOCAB_INFO)


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def test_tokenizer_output_shape(batch: dict) -> None:
    tok = MixedEventTokenizer(
        event_type_vocab_size=20,
        event_type_emb_dim=32,
        cat_vocab_sizes=[8, 6],
        cat_emb_dim=16,
        num_num_features=3,
        num_projection_dim=32,
        time_projection_dim=32,
        hidden_dim=H,
        max_seq_len=64,
    )
    out = tok(
        event_type=batch["event_type"],
        time_delta=batch["time_delta"],
        num_features=batch["num_features"],
        cat_features=batch["cat_features"],
        attention_mask=batch["attention_mask"],
    )
    assert out.shape == (B, L, H)
    assert out.isfinite().all(), "tokenizer output contains NaN/Inf"


def test_tokenizer_projects_preprocessed_time_delta_without_log_or_clamp(batch: dict) -> None:
    class CaptureTimeProjection(nn.Module):
        def __init__(self, out_dim: int) -> None:
            super().__init__()
            self.out_dim = out_dim
            self.seen: torch.Tensor | None = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.seen = x.detach().clone()
            return torch.zeros(*x.shape[:-1], self.out_dim, device=x.device, dtype=x.dtype)

    tok = MixedEventTokenizer(
        event_type_vocab_size=20,
        event_type_emb_dim=32,
        cat_vocab_sizes=[8, 6],
        cat_emb_dim=16,
        num_num_features=3,
        num_projection_dim=32,
        time_projection_dim=32,
        hidden_dim=H,
        max_seq_len=64,
        dropout=0.0,
    )
    capture = CaptureTimeProjection(out_dim=32)
    tok.time_projection = capture

    scaled_time = torch.linspace(-2.5, 2.5, B * L).reshape(B, L)
    out = tok(
        event_type=batch["event_type"],
        time_delta=scaled_time,
        num_features=batch["num_features"],
        cat_features=batch["cat_features"],
        attention_mask=batch["attention_mask"],
    )

    assert out.shape == (B, L, H)
    assert out.isfinite().all(), "tokenizer output contains NaN/Inf"
    assert capture.seen is not None
    assert torch.equal(capture.seen.squeeze(-1), scaled_time)


# ── Encoder ───────────────────────────────────────────────────────────────────

def test_encoder_output_shape(batch: dict) -> None:
    enc = TimeAwareTransformerEncoder(
        hidden_dim=H, num_layers=2, num_heads=4, dim_feedforward=128, dropout=0.0
    )
    x = torch.randn(B, L, H)
    out = enc(x, batch["attention_mask"])
    assert out.shape == (B, L, H)


# ── Pooling ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pooling_type", ["cls", "mean", "max", "attention", "gated_attention"])
def test_pooling_shapes(pooling_type: str, batch: dict) -> None:
    pool = get_pooling(pooling_type, H)
    x = torch.randn(B, L, H)
    out = pool(x, batch["attention_mask"])
    assert out.shape == (B, H)


# ── DMEEncoder pretrain ───────────────────────────────────────────────────────

def test_pretrain_mode(model: DMEEncoder, batch: dict) -> None:
    out = model(batch, mode="pretrain")

    assert out["event_type_logits"].shape == (B, L, 20)
    assert out["time_delta_pred"].shape   == (B, L, 1)
    assert out["existence_logits"].shape  == (B, L, 1)
    assert out["num_pred"].shape          == (B, L, 3)
    assert out["hidden_states"].shape     == (B, L, H)

    cat_logits = out["cat_logits"]
    assert len(cat_logits) == 2
    assert cat_logits[0].shape == (B, L, 8)
    assert cat_logits[1].shape == (B, L, 6)

    for key, val in out.items():
        tensors = val if isinstance(val, list) else [val]
        for t in tensors:
            assert t.isfinite().all(), f"pretrain output '{key}' contains NaN/Inf"


def test_diffusion_pretrain_mode(batch: dict) -> None:
    config = copy.deepcopy(CONFIG)
    config["pretraining"] = {"objective": "diffusion"}
    config["diffusion"] = {"num_steps": 32}
    model = DMEEncoder(config, VOCAB_INFO)
    diffusion_batch = {**batch, "diffusion_t": torch.randint(1, 33, (B,))}

    out = model(diffusion_batch, mode="pretrain")

    assert out["event_type_logits"].shape == (B, L, 20)
    assert out["time_delta_pred"].shape == (B, L, 1)
    assert out["time_delta_eps_pred"].shape == (B, L, 1)
    assert out["num_pred"].shape == (B, L, 3)
    assert out["num_eps_pred"].shape == (B, L, 3)
    assert out["hidden_states"].shape == (B, L, H)

    for key, val in out.items():
        tensors = val if isinstance(val, list) else [val]
        for t in tensors:
            assert t.isfinite().all(), f"diffusion output '{key}' contains NaN/Inf"


def test_diffusion_d3pm_pretrain_mode(batch: dict) -> None:
    config = copy.deepcopy(CONFIG)
    config["pretraining"] = {"objective": "diffusion"}
    config["diffusion"] = {"num_steps": 32}
    config["d3pm"] = {"enabled": True, "apply_to": ["event_type"]}
    model = DMEEncoder(config, VOCAB_INFO)
    diffusion_batch = {**batch, "diffusion_t": torch.randint(1, 33, (B,))}

    out = model(diffusion_batch, mode="pretrain")

    assert out["event_type_prev_logits"].shape == (B, L, 20)
    assert out["event_type_prev_logits"].isfinite().all()
    assert "event_type_prev_head" in model.count_parameters()


# ── DMEEncoder finetune ───────────────────────────────────────────────────────

def test_finetune_mode(model: DMEEncoder, batch: dict) -> None:
    out = model(batch, mode="finetune")
    assert out["logits"].shape         == (B, VOCAB_INFO["num_classes"])
    assert out["representation"].shape == (B, H)


# ── fp16 ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fp16 LayerNorm not supported on CPU; use torch.autocast for GPU training",
)
def test_fp16_compatibility(model: DMEEncoder, batch: dict) -> None:
    device = torch.device("cuda")
    model_h = model.to(device).half()
    batch_h = {
        k: (v.half() if v.is_floating_point() else v).to(device)
        for k, v in batch.items()
    }

    out = model_h(batch_h, mode="finetune")

    assert out["logits"].shape == (B, VOCAB_INFO["num_classes"])
    assert out["logits"].dtype == torch.float16


# ── Parameter count ───────────────────────────────────────────────────────────

def test_parameter_count(model: DMEEncoder) -> None:
    params = model.count_parameters()

    assert isinstance(params, dict)
    assert params["total"] > 0
    assert params["total"] == sum(p.numel() for p in model.parameters())

    expected_keys = {
        "tokenizer", "encoder", "pooling",
        "event_type_head", "time_delta_head", "existence_head",
        "numerical_head", "cat_head", "total",
    }
    if model.classifier is not None:
        expected_keys.add("classifier")
    assert expected_keys == set(params.keys())
