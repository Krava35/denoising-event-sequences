from __future__ import annotations

import torch

from src.corruption.diffusion import (
    ConditionalSuffixDiffusionPipeline,
    DiffusionCorruptionPipeline,
    DiffusionSchedule,
)


def _make_batch() -> dict:
    torch.manual_seed(0)
    B, L = 4, 8
    attention_mask = torch.ones(B, L, dtype=torch.bool)
    attention_mask[1, 6:] = False
    event_type = torch.randint(5, 20, (B, L))
    event_type[~attention_mask] = 0
    cat_features = torch.stack(
        [
            torch.randint(5, 12, (B, L)),
            torch.randint(5, 15, (B, L)),
        ],
        dim=-1,
    )
    cat_features[~attention_mask] = 0
    return {
        "event_type": event_type,
        "time_delta": torch.randn(B, L),
        "num_features": torch.randn(B, L, 3),
        "cat_features": cat_features,
        "attention_mask": attention_mask,
    }


def test_diffusion_schedule_is_monotonic() -> None:
    schedule = DiffusionSchedule(num_steps=10, beta_start=1e-4, beta_end=2e-2)

    assert schedule.alpha_bars.shape == (11,)
    assert torch.isfinite(schedule.alpha_bars).all()
    assert schedule.alpha_bars[0].item() == 1.0
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])


def test_diffusion_pipeline_shapes_and_padding() -> None:
    torch.manual_seed(42)
    batch = _make_batch()
    pipe = DiffusionCorruptionPipeline(
        {
            "diffusion": {
                "num_steps": 12,
                "beta_start": 1e-4,
                "beta_end": 2e-2,
                "discrete_mask_fraction": 0.8,
            }
        },
        vocab_sizes={"event_type": 20, "cat_features": [12, 15]},
    )

    corrupted, targets, masks = pipe(batch)

    assert corrupted["diffusion_t"].shape == (batch["event_type"].shape[0],)
    assert int(corrupted["diffusion_t"].min().item()) >= 1
    assert int(corrupted["diffusion_t"].max().item()) <= 12

    for key in ("event_type", "time_delta", "num_features", "cat_features"):
        assert corrupted[key].shape == batch[key].shape
        assert targets[key].shape == batch[key].shape

    assert targets["time_delta_eps"].shape == batch["time_delta"].shape
    assert targets["num_features_eps"].shape == batch["num_features"].shape
    assert masks["event_type"].equal(batch["attention_mask"])
    assert masks["time_delta"].equal(batch["attention_mask"])
    assert masks["num_features"].equal(batch["attention_mask"])
    assert masks["cat_features"].shape == batch["cat_features"].shape

    pad = ~batch["attention_mask"]
    assert (corrupted["event_type"][pad] == 0).all()
    assert (corrupted["cat_features"][pad] == 0).all()
    assert torch.isfinite(corrupted["time_delta"]).all()
    assert torch.isfinite(corrupted["num_features"]).all()


def test_diffusion_pipeline_d3pm_event_type_prev_target() -> None:
    torch.manual_seed(43)
    batch = _make_batch()
    pipe = DiffusionCorruptionPipeline(
        {
            "diffusion": {
                "num_steps": 12,
                "beta_start": 1e-4,
                "beta_end": 2e-2,
                "discrete_mask_fraction": 0.8,
            },
            "d3pm": {
                "enabled": True,
                "apply_to": ["event_type"],
                "loss_weight_event_type_prev": 0.25,
            },
        },
        vocab_sizes={"event_type": 20, "cat_features": [12, 15]},
    )

    corrupted, targets, masks = pipe(batch)

    assert corrupted["event_type"].shape == batch["event_type"].shape
    assert targets["d3pm_event_type_prev"].shape == batch["event_type"].shape
    assert masks["d3pm_event_type_prev"].equal(batch["attention_mask"])
    assert int(targets["d3pm_event_type_prev"][batch["attention_mask"]].min().item()) >= 0
    assert (targets["d3pm_event_type_prev"][~batch["attention_mask"]] == 0).all()


def test_continuous_corruption_matches_closed_form() -> None:
    torch.manual_seed(7)
    values = torch.randn(2, 4)
    attention_mask = torch.ones(2, 4, dtype=torch.bool)
    alpha_bar = torch.tensor([0.25, 0.81])
    pipe = DiffusionCorruptionPipeline({"diffusion": {"num_steps": 4}})

    corrupted, eps = pipe._corrupt_continuous(values, attention_mask, alpha_bar)
    expected = torch.sqrt(alpha_bar).unsqueeze(-1) * values + torch.sqrt(
        1.0 - alpha_bar
    ).unsqueeze(-1) * eps

    assert torch.allclose(corrupted, expected)


def test_conditional_suffix_diffusion_keeps_prefix_clean() -> None:
    torch.manual_seed(123)
    batch = _make_batch()
    pipe = ConditionalSuffixDiffusionPipeline(
        {
            "diffusion": {
                "num_steps": 12,
                "beta_start": 1e-4,
                "beta_end": 2e-2,
                "discrete_mask_fraction": 1.0,
            },
            "generation": {
                "suffix_len": 2,
                "prefix_min_ratio": 0.5,
                "prefix_max_ratio": 0.5,
            },
        },
        vocab_sizes={"event_type": 20, "cat_features": [12, 15]},
    )

    corrupted, targets, masks = pipe(batch)
    prefix_mask = masks["generation_prefix"]
    suffix_mask = masks["generation_suffix"]
    effective_attention = prefix_mask | suffix_mask

    assert corrupted["attention_mask"].equal(effective_attention)
    assert not bool((prefix_mask & suffix_mask).any())
    assert bool(suffix_mask.any())
    assert masks["event_type"].equal(suffix_mask)
    assert masks["time_delta"].equal(suffix_mask)
    assert masks["num_features"].equal(suffix_mask)
    assert masks["cat_features"].equal(suffix_mask.unsqueeze(-1).expand_as(batch["cat_features"]))

    assert torch.equal(corrupted["event_type"][prefix_mask], batch["event_type"][prefix_mask])
    assert torch.equal(corrupted["cat_features"][prefix_mask], batch["cat_features"][prefix_mask])
    assert torch.allclose(corrupted["time_delta"][prefix_mask], batch["time_delta"][prefix_mask])
    assert torch.allclose(
        corrupted["num_features"][prefix_mask],
        batch["num_features"][prefix_mask],
    )

    assert (corrupted["event_type"][~effective_attention] == 0).all()
    assert (corrupted["cat_features"][~effective_attention] == 0).all()
    assert targets["event_type"].equal(batch["event_type"])


def test_conditional_suffix_uses_forecast_cut_as_prefix() -> None:
    torch.manual_seed(124)
    batch = _make_batch()
    batch["forecast_cut"] = torch.tensor([3, 4, 5, 6], dtype=torch.long)
    pipe = ConditionalSuffixDiffusionPipeline(
        {
            "diffusion": {
                "num_steps": 12,
                "beta_start": 1e-4,
                "beta_end": 2e-2,
                "discrete_mask_fraction": 1.0,
            },
            "generation": {
                "suffix_len": 2,
                "prefix_min_ratio": 0.5,
                "prefix_max_ratio": 0.5,
            },
        },
        vocab_sizes={"event_type": 20, "cat_features": [12, 15]},
    )

    _, _, masks = pipe(batch)

    assert torch.equal(masks["generation_prefix"].sum(dim=1), batch["forecast_cut"])
