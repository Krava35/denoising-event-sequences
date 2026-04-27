from __future__ import annotations

import csv
import json
import runpy
import sys
from pathlib import Path

import pytest
import torch
import yaml

from src.utils.config import load_config, save_config
from src.utils.logging import MetricsLogger
from src.utils.seed import get_device, set_seed


def test_seed_reproducibility_and_device() -> None:
    set_seed(42)
    first = torch.rand(8)
    set_seed(42)
    second = torch.rand(8)
    assert torch.equal(first, second)

    device = get_device()
    assert isinstance(device, torch.device)
    assert str(device) in {"cpu", "cuda", "mps"}


def test_config_load_merge_and_save(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    override_path = tmp_path / "override.yaml"
    out_path = tmp_path / "saved.yaml"

    base_data = {
        "data": {"max_seq_len": 256, "min_seq_len": 2},
        "corruption": {"event_type": {"mask_prob": 0.25}},
        "model": {"hidden_dim": 128, "num_layers": 4},
        "training": {"batch_size": 64, "lr": 3e-4},
        "project": {"name": "demo"},
    }
    override_data = {"model": {"num_layers": 6}, "training": {"batch_size": 32}}
    base_path.write_text(yaml.safe_dump(base_data))
    override_path.write_text(yaml.safe_dump(override_data))

    merged = load_config(str(base_path), override=str(override_path))
    assert merged["model"]["num_layers"] == 6
    assert merged["model"]["hidden_dim"] == 128
    assert merged["training"]["batch_size"] == 32
    assert merged["training"]["lr"] == pytest.approx(3e-4)

    save_config(merged, str(out_path))
    saved = load_config(str(out_path))
    assert "_metadata" in saved
    assert "saved_at" in saved["_metadata"]
    assert saved["model"]["num_layers"] == 6


def test_config_validation_for_required_keys(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump({"project": {"name": "x"}}))

    with pytest.raises(KeyError):
        load_config(str(bad_path))


def test_metrics_logger_records_snapshot_and_csv(tmp_path: Path) -> None:
    logger = MetricsLogger(output_dir=str(tmp_path), experiment_name="exp")
    logger.log_step(step=1, metrics={"loss": 0.9, "lr": 1e-3})
    logger.log_epoch(epoch=1, metrics={"val_loss": 0.7, "val_f1": 0.5})
    logger.log_config({"model": {"hidden_dim": 64}})

    metrics_path = tmp_path / "exp" / "metrics.jsonl"
    with metrics_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 2
    assert rows[0]["step"] == 1
    assert rows[1]["epoch"] == 1
    assert "timestamp" in rows[0]

    snapshot_path = tmp_path / "exp" / "config_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["model"]["hidden_dim"] == 64
    assert "saved_at" in snapshot

    csv_path = tmp_path / "exp" / "metrics.csv"
    logger.to_csv(str(csv_path))
    with csv_path.open() as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 2
    assert csv_rows[0]["loss"] == "0.9"
    assert csv_rows[1]["val_f1"] == "0.5"


@pytest.mark.parametrize(
    "module_name",
    ["src.utils.seed", "src.utils.config", "src.utils.logging"],
)
def test_utils_main_blocks_execute(module_name: str) -> None:
    sys.modules.pop(module_name, None)
    runpy.run_module(module_name, run_name="__main__")
