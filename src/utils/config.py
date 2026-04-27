import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("data", "corruption", "model", "training")


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str, override: str | None = None) -> dict:
    with open(path) as f:
        config: dict = yaml.safe_load(f) or {}
    logger.info("Loaded config from %s", path)

    if override is not None:
        with open(override) as f:
            override_data: dict = yaml.safe_load(f) or {}
        config = _deep_merge(config, override_data)
        logger.info("Merged override from %s", override)

    missing = [k for k in _REQUIRED_KEYS if k not in config]
    if missing:
        raise KeyError(f"Config is missing required top-level keys: {missing}")

    return config


def _load_yaml(path: str) -> dict:
    """Load YAML without required-key validation (for partial override configs)."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_experiment_config(
    base_path: str = "configs/base.yaml",
    dataset_path: str | None = None,
    ablation_path: str | None = None,
) -> dict:
    """
    Load config with three-level merge.

    Priority (highest → lowest):
      ablation_path > dataset_path > base_path

    Example:
      config = load_experiment_config(
          base_path="configs/base.yaml",
          dataset_path="configs/datasets/rosbank.yaml",
          ablation_path="configs/ablations/A3_transition_aware.yaml",
      )
    """
    config = load_config(base_path)
    if dataset_path:
        config = _deep_merge(config, _load_yaml(dataset_path))
        logger.info("Merged dataset config from %s", dataset_path)
    if ablation_path:
        config = _deep_merge(config, _load_yaml(ablation_path))
        logger.info("Merged ablation config from %s", ablation_path)
    return config


def save_config(config: dict, path: str) -> None:
    output = {
        "_metadata": {"saved_at": datetime.now(timezone.utc).isoformat()},
        **config,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info("Saved config to %s", path)


if __name__ == "__main__":
    import os
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")

    base_data = {
        "data": {"max_seq_len": 256},
        "corruption": {"event_type": {"mask_prob": 0.28}},
        "model": {"hidden_dim": 256, "num_layers": 4},
        "training": {"batch_size": 128, "lr": 3e-4},
        "project": {"name": "test"},
    }
    override_data = {
        "model": {"num_layers": 6},
        "training": {"batch_size": 64},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "base.yaml")
        override_path = os.path.join(tmpdir, "override.yaml")
        out_path = os.path.join(tmpdir, "saved.yaml")

        with open(base_path, "w") as f:
            yaml.dump(base_data, f)
        with open(override_path, "w") as f:
            yaml.dump(override_data, f)

        # load without override
        cfg = load_config(base_path)
        assert cfg["model"]["hidden_dim"] == 256
        assert cfg["training"]["batch_size"] == 128

        # load with override — override wins on scalars, base preserved elsewhere
        cfg = load_config(base_path, override=override_path)
        assert cfg["model"]["num_layers"] == 6, "override must win"
        assert cfg["model"]["hidden_dim"] == 256, "base key must be preserved"
        assert cfg["training"]["batch_size"] == 64, "override must win"
        assert cfg["training"]["lr"] == 3e-4, "base key must be preserved"

        # save and reload
        save_config(cfg, out_path)
        reloaded = load_config(out_path)
        assert "_metadata" in reloaded
        assert "saved_at" in reloaded["_metadata"]
        assert reloaded["model"]["hidden_dim"] == 256

        # missing required keys must raise
        bad = {"project": {"name": "x"}}
        bad_path = os.path.join(tmpdir, "bad.yaml")
        with open(bad_path, "w") as f:
            yaml.dump(bad, f)
        try:
            load_config(bad_path)
            raise AssertionError("Expected KeyError")
        except KeyError as e:
            print(f"Validation check passed: {e}")

        # load_experiment_config: three-level merge
        dataset_data = {"data": {"max_seq_len": 512}, "training": {"task": "binary"}}
        ablation_data = {"corruption": {"event_type": {"use_transition_aware_replacement": True}}}
        dataset_path2 = os.path.join(tmpdir, "dataset.yaml")
        ablation_path2 = os.path.join(tmpdir, "ablation.yaml")
        with open(dataset_path2, "w") as f:
            yaml.dump(dataset_data, f)
        with open(ablation_path2, "w") as f:
            yaml.dump(ablation_data, f)

        cfg3 = load_experiment_config(base_path, dataset_path2, ablation_path2)
        assert cfg3["data"]["max_seq_len"] == 512, "dataset must override base"
        assert cfg3["model"]["hidden_dim"] == 256, "base preserved"
        assert cfg3["training"]["task"] == "binary", "dataset field present"
        assert cfg3["corruption"]["event_type"]["use_transition_aware_replacement"] is True
        assert cfg3["corruption"]["event_type"]["mask_prob"] == 0.28, "base preserved"

        # load_experiment_config without dataset/ablation == load_config
        cfg4 = load_experiment_config(base_path)
        assert cfg4["model"]["hidden_dim"] == 256

    print("All checks passed.")
