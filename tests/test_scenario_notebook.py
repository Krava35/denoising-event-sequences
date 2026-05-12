from __future__ import annotations

import json
from pathlib import Path


def test_kaggle_scenario_notebook_structure() -> None:
    notebook_path = Path("notebooks/kaggle_scenario_module.ipynb")
    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )

    required_snippets = [
        "RAW_DATA_PATH",
        "RUN_PREPARE_DATA_IF_MISSING = True",
        "FORCE_REBUILD_PROCESSED = False",
        "SCENARIO_CONFIG_PATH",
        "scenario_recovered_config.yaml",
        "patch_rosbank_config_if_needed",
        "\"group_col\": \"cl_id\"",
        "\"event_type_col\": \"MCC\"",
        "\"target_col\": \"target_flag\"",
        "RUN_FORECAST_PRETRAIN_IF_MISSING = False",
        "Recover Processed Artifacts From Raw Data",
        "scripts\" / \"prepare_data.py",
        "scenario_prepare_data.log",
        "ensure_data_artifacts_loaded",
        "FORECAST_CKPT_PATH",
        "scenario_from_outputs",
        "scenario_from_targets",
        "compute_forecast_quality_metrics",
        "compute_forecast_baseline_metrics",
        "scenario_eval_metrics.json",
        "scenario_examples.json",
        "scenario_predictions.parquet",
        "scenario_metric_comparison.csv",
    ]
    for snippet in required_snippets:
        assert snippet in source


def test_kaggle_a9_aux_notebook_structure() -> None:
    notebook_path = Path("notebooks/kaggle_a9_d3pm_forecast_aux_experiments.ipynb")
    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )

    required_snippets = [
        "# Cell 1 - Setup & Install",
        "# Cell 2 - Paths & Runtime Knobs",
        "# Cell 3 - A9 Auxiliary Experiment Configs",
        "# Cell 4 - Data Loading & Preprocessing Artifacts",
        "# Cell 5 - Prepared Artifact Inspection",
        "# Cell 6 - Auxiliary Diffusion Smoke Test",
        "# Cell 7 - Diffusion Pretraining Experiments",
        "# Cell 8 - Diffusion Validation Metrics",
        "# Cell 9 - Fine-tuning Frozen and Full Encoder",
        "# Cell 10 - Optional Low-Label Evaluation",
        "# Cell 11 - Event Suffix Generation",
        "# Cell 12 - Artifact Summary",
        "a9_d3pm_aux",
        "a9_forecast_aux",
        "a9_d3pm_forecast_aux",
        "scripts/pretrain.py",
        "scripts/finetune.py",
        "scripts/generate_events.py",
    ]
    for snippet in required_snippets:
        assert snippet in source


def test_kaggle_final_forecast_diffusion_notebook_structure() -> None:
    notebook_path = Path("notebooks/kaggle_final_forecast_diffusion_pipeline.ipynb")
    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )

    required_snippets = [
        "# Cell 1 - Setup & Install",
        "# Cell 2 - Paths, Targets & Runtime Knobs",
        "# Cell 3 - Final Forecast-Diffusion Config Sweep",
        "# Cell 4 - Data Loading & Preprocessing Artifacts",
        "# Cell 5 - Prepared Artifact Inspection",
        "# Cell 6 - Final Diffusion Smoke Test",
        "# Cell 7 - Final Diffusion Pretraining Sweep",
        "# Cell 8 - Final Diffusion Validation Metrics",
        "# Cell 9 - Fine-tuning Frozen and Full Encoder",
        "# Cell 10 - Compare Against Checked-In Results",
        "# Cell 11 - Event Suffix Generation",
        "# Cell 12 - Artifact Summary",
        "A13_final_forecast_aux.yaml",
        "RESULTS_DME_FULL_TARGET",
        "final_w001",
        "final_w002",
        "final_w003",
        "final_legacy256_w002",
        "\"currency\"",
        "use_prev_logits_for_sampling",
        "False",
        "final_forecast_vs_checked_in_results.csv",
        "scripts/pretrain.py",
        "scripts/finetune.py",
        "scripts/generate_events.py",
    ]
    for snippet in required_snippets:
        assert snippet in source
