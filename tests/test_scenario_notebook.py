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
        "RUN_FORECAST_PRETRAIN_IF_MISSING = False",
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
