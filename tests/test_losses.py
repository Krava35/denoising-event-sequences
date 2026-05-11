from __future__ import annotations

import torch

from src.training.losses import compute_diffusion_pretraining_loss, compute_pretraining_loss

B, L = 4, 16
V_TYPE = 20
N_NUM = 3
N_CAT = 2
V_CAT = [8, 6]

CONFIG = {
    "loss": {
        "lambda_event_type": 1.0,
        "lambda_time": 1.0,
        "lambda_num": 0.5,
        "lambda_cat": 0.5,
        "lambda_exist": 0.1,
        "lambda_time_eps": 0.1,
        "lambda_num_eps": 0.1,
    }
}

_COMPONENTS = ["total", "event_type", "time_delta", "numerical", "categorical", "existence"]


def _make_outputs(device: str = "cpu") -> dict:
    return {
        "event_type_logits": torch.randn(B, L, V_TYPE, device=device, requires_grad=True),
        "time_delta_pred": torch.randn(B, L, 1, device=device, requires_grad=True),
        "num_pred": torch.randn(B, L, N_NUM, device=device, requires_grad=True),
        "cat_logits": [
            torch.randn(B, L, V_CAT[j], device=device, requires_grad=True)
            for j in range(N_CAT)
        ],
        "existence_logits": torch.randn(B, L, 1, device=device, requires_grad=True),
    }


def _make_diffusion_outputs(device: str = "cpu") -> dict:
    outputs = _make_outputs(device)
    outputs["time_delta_eps_pred"] = torch.randn(B, L, 1, device=device, requires_grad=True)
    outputs["num_eps_pred"] = torch.randn(B, L, N_NUM, device=device, requires_grad=True)
    return outputs


def _make_targets(device: str = "cpu") -> dict:
    return {
        "event_type": torch.randint(0, V_TYPE, (B, L), device=device),
        "time_delta": torch.randn(B, L, device=device),
        "num_features": torch.randn(B, L, N_NUM, device=device),
        "cat_features": torch.stack(
            [torch.randint(0, V_CAT[j], (B, L), device=device) for j in range(N_CAT)],
            dim=-1,
        ),
    }


def _make_diffusion_targets(device: str = "cpu") -> dict:
    targets = _make_targets(device)
    targets["time_delta_eps"] = torch.randn(B, L, device=device)
    targets["num_features_eps"] = torch.randn(B, L, N_NUM, device=device)
    return targets


def _full_masks(device: str = "cpu") -> dict:
    return {
        "event_type": torch.ones(B, L, dtype=torch.bool, device=device),
        "time_delta": torch.ones(B, L, dtype=torch.bool, device=device),
        "num_features": torch.ones(B, L, dtype=torch.bool, device=device),
        "cat_features": torch.ones(B, L, N_CAT, dtype=torch.bool, device=device),
        "event_level": torch.zeros(B, L, dtype=torch.bool, device=device),
        "attention_mask": torch.ones(B, L, dtype=torch.bool, device=device),
    }


def _full_diffusion_masks(device: str = "cpu") -> dict:
    masks = _full_masks(device)
    masks["time_delta_eps"] = torch.ones(B, L, dtype=torch.bool, device=device)
    masks["num_features_eps"] = torch.ones(B, L, dtype=torch.bool, device=device)
    return masks


def _padded_masks(pad_start: int = 12, device: str = "cpu") -> dict:
    m = _full_masks(device)
    m["attention_mask"][:, pad_start:] = False
    m["event_type"][:, pad_start:] = False
    m["time_delta"][:, pad_start:] = False
    m["num_features"][:, pad_start:] = False
    m["cat_features"][:, pad_start:, :] = False
    m["event_level"][:, pad_start:] = False
    return m


def _padded_diffusion_masks(pad_start: int = 12, device: str = "cpu") -> dict:
    m = _padded_masks(pad_start, device)
    m["time_delta_eps"] = torch.ones(B, L, dtype=torch.bool, device=device)
    m["num_features_eps"] = torch.ones(B, L, dtype=torch.bool, device=device)
    m["time_delta_eps"][:, pad_start:] = False
    m["num_features_eps"][:, pad_start:] = False
    return m


def test_ignores_padding() -> None:
    torch.manual_seed(42)
    outputs = _make_outputs()
    targets = _make_targets()
    masks = _padded_masks()

    result_ref = compute_pretraining_loss(outputs, targets, masks, CONFIG)

    # Modify targets at padding positions with garbage values
    targets_mod = {k: v.clone() for k, v in targets.items()}
    targets_mod["event_type"][:, 12:] = V_TYPE - 1
    targets_mod["time_delta"][:, 12:] = 99999.0
    targets_mod["num_features"][:, 12:] = 99999.0
    targets_mod["cat_features"][:, 12:] = V_CAT[0] - 1

    result_mod = compute_pretraining_loss(outputs, targets_mod, masks, CONFIG)

    for key in _COMPONENTS:
        assert torch.allclose(result_ref[key], result_mod[key]), (
            f"Loss '{key}' changed when garbage was written to padding positions"
        )


def test_empty_mask() -> None:
    torch.manual_seed(42)
    outputs = _make_outputs()
    targets = _make_targets()

    masks_empty = {
        "event_type": torch.zeros(B, L, dtype=torch.bool),
        "time_delta": torch.zeros(B, L, dtype=torch.bool),
        "num_features": torch.zeros(B, L, dtype=torch.bool),
        "cat_features": torch.zeros(B, L, N_CAT, dtype=torch.bool),
        "event_level": torch.zeros(B, L, dtype=torch.bool),
        "attention_mask": torch.zeros(B, L, dtype=torch.bool),
    }

    result = compute_pretraining_loss(outputs, targets, masks_empty, CONFIG)

    for key in _COMPONENTS:
        assert not torch.isnan(result[key]), f"'{key}' loss is NaN"
        assert result[key].item() == 0.0, f"'{key}' loss should be 0.0, got {result[key].item()}"


def test_loss_components() -> None:
    torch.manual_seed(42)
    outputs = _make_outputs()
    targets = _make_targets()
    masks = _full_masks()

    result = compute_pretraining_loss(outputs, targets, masks, CONFIG)

    for key in ["event_type", "time_delta", "numerical", "categorical", "existence"]:
        assert result[key].item() > 0.0, f"'{key}' loss should be positive"
    assert result["total"].item() > 0.0, "total loss should be positive"


def test_gradient_flow() -> None:
    torch.manual_seed(42)
    outputs = _make_outputs()
    targets = _make_targets()
    masks = _full_masks()

    result = compute_pretraining_loss(outputs, targets, masks, CONFIG)
    result["total"].backward()

    assert outputs["event_type_logits"].grad is not None
    assert outputs["time_delta_pred"].grad is not None
    assert outputs["num_pred"].grad is not None
    for j, logits_j in enumerate(outputs["cat_logits"]):
        assert logits_j.grad is not None, f"cat_logits[{j}].grad is None"
    assert outputs["existence_logits"].grad is not None


def test_diffusion_loss_components_and_gradients() -> None:
    torch.manual_seed(42)
    outputs = _make_diffusion_outputs()
    targets = _make_diffusion_targets()
    masks = _full_diffusion_masks()

    result = compute_diffusion_pretraining_loss(outputs, targets, masks, CONFIG)
    result["total"].backward()

    for key in [
        "event_type",
        "time_delta",
        "numerical",
        "categorical",
        "time_delta_eps",
        "numerical_eps",
    ]:
        assert result[key].item() > 0.0, f"'{key}' loss should be positive"
    assert result["total"].item() > 0.0
    assert outputs["time_delta_eps_pred"].grad is not None
    assert outputs["num_eps_pred"].grad is not None


def test_diffusion_loss_ignores_padding() -> None:
    torch.manual_seed(42)
    outputs = _make_diffusion_outputs()
    targets = _make_diffusion_targets()
    masks = _padded_diffusion_masks()

    result_ref = compute_diffusion_pretraining_loss(outputs, targets, masks, CONFIG)

    targets_mod = {k: v.clone() for k, v in targets.items()}
    targets_mod["event_type"][:, 12:] = V_TYPE - 1
    targets_mod["time_delta"][:, 12:] = 99999.0
    targets_mod["time_delta_eps"][:, 12:] = 99999.0
    targets_mod["num_features"][:, 12:] = 99999.0
    targets_mod["num_features_eps"][:, 12:] = 99999.0
    targets_mod["cat_features"][:, 12:] = V_CAT[0] - 1

    result_mod = compute_diffusion_pretraining_loss(outputs, targets_mod, masks, CONFIG)

    for key in [
        "total",
        "event_type",
        "time_delta",
        "numerical",
        "categorical",
        "time_delta_eps",
        "numerical_eps",
    ]:
        assert torch.allclose(result_ref[key], result_mod[key])
