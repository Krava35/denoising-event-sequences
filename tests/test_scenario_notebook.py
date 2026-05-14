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


def test_kaggle_final_forecast_pretrain_sweep_notebook_structure() -> None:
    notebook_path = Path("notebooks/kaggle_final_forecast_pretrain_sweep.ipynb")
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
        "# Cell 3 - Final Forecast-Pretrain Config Sweep",
        "# Cell 4 - Data Loading & Preprocessing Artifacts",
        "# Cell 5 - Prepared Artifact Inspection",
        "# Cell 6 - Forecast Hybrid Smoke Test",
        "# Cell 7 - Forecast Pretraining Sweep",
        "# Cell 8 - Fine-tuning Sweep",
        "# Cell 9 - Compare Against Checked-In Results",
        "# Cell 10 - Forecast Scenario Generation Check",
        "# Cell 11 - Artifact Summary",
        "A14_final_forecast_pretrain.yaml",
        "forecast_repro",
        "forecast_alpha015",
        "forecast_alpha030",
        "forecast_reg015",
        "forecast_enc2e5",
        "\"currency\"",
        "RESULTS_DME_FULL_TARGET",
        "scripts/forecast_pretrain.py",
        "scripts/finetune.py",
        "forecast_eval_metrics",
        "scenario_examples",
        "best_forecast_generation_metrics.csv",
        "best_forecast_scenario_examples.csv",
        "final_forecast_pretrain_vs_checked_in_results.csv",
    ]
    for snippet in required_snippets:
        assert snippet in source


def test_kaggle_coles_baseline_notebook_structure() -> None:
    notebook_path = Path("notebooks/kaggle_baseline_coles_ptls.ipynb")
    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )

    required_snippets = [
        "COLES_ARTICLE_LIKE_PREP = True",
        'COLES_PRETRAIN_ENTITY_SCOPE = "train_prefix_only"',
        'COLES_DOWNSTREAM_MODES = ["classification_head", "full_finetune", "catboost"]',
        "sample_article_like_slice",
        "manual_article_like_multi_slice",
        "assert_no_future_leakage",
        "CoLESClassifier",
        "make_class_weights",
        "extract_coles_embeddings",
        "train_coles_catboost",
        "CatBoostClassifier",
        "Pool(",
        "coles_classification_head",
        "coles_full_finetune",
        "coles_catboost",
        "coles_catboost_feature_importance.csv",
    ]
    for snippet in required_snippets:
        assert snippet in source


def test_kaggle_catboost_baseline_notebook_structure() -> None:
    notebook_path = Path("notebooks/kaggle_baseline_aggregates_catboost.ipynb")
    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )

    required_snippets = [
        "CATBOOST_CATEGORICAL_FEATURES",
        "CATBOOST_FITTED_FEATURE_METADATA",
        "assert_catboost_feature_frame_is_leak_safe",
        "assert_no_future_leakage",
        "first_",
        "last_",
        "mode_",
        "recency_weighted",
        "Pool(",
        "cat_features=cat_feature_names",
        'iterations = 100 if SMOKE_RUN else 3000',
        'learning_rate=0.025',
        'bootstrap_type="Bayesian"',
        '"feature_metadata": CATBOOST_FITTED_FEATURE_METADATA',
    ]
    for snippet in required_snippets:
        assert snippet in source


def test_public_benchmark_prepare_notebook_structure() -> None:
    notebook_path = Path("notebooks/kaggle_prepare_gender_age_group_data.ipynb")
    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )

    required_snippets = [
        "# Cell 1 - Setup & Install",
        "# Cell 2 - Dataset Selection",
        "# Cell 3 - Run Preparation Script",
        "# Cell 4 - Inspect Artifacts and Split Balance",
        "# Cell 5 - Artifact Summary",
        "prepare_public_benchmark_data.py",
        "DATASETS_TO_PREPARE = [\"gender\", \"age_group\"]",
        "official hidden/public test files are ignored",
        "canonical_events.parquet",
        "transformed_events.parquet",
        "prepared_config.yaml",
    ]
    for snippet in required_snippets:
        assert snippet in source


def test_public_benchmark_catboost_notebook_structure() -> None:
    notebook_path = Path("notebooks/kaggle_baseline_aggregates_catboost_public_benchmarks.ipynb")
    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )

    required_snippets = [
        "DATASET_NAME = \"gender\"",
        "prepare_public_benchmark_data.py",
        "CATBOOST_CATEGORICAL_FEATURES",
        "CATBOOST_FITTED_FEATURE_METADATA",
        "assert_no_future_leakage",
        "assert_catboost_feature_frame_is_leak_safe",
        "recency_weighted",
        "Pool(",
        "cat_features=cat_feature_names",
        "bootstrap_type=\"Bayesian\"",
        "events_are_raw_pretransform",
        "transformed_events.parquet",
        "catboost_agg_metrics.json",
    ]
    for snippet in required_snippets:
        assert snippet in source


def test_public_benchmark_coles_notebook_structure() -> None:
    notebook_path = Path("notebooks/kaggle_baseline_coles_public_benchmarks.ipynb")
    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )

    required_snippets = [
        "COLES_ARTICLE_LIKE_PREP = True",
        "COLES_PRETRAIN_ENTITY_SCOPE = \"train_prefix_only\"",
        "COLES_DOWNSTREAM_MODES = [\"classification_head\", \"full_finetune\", \"catboost\"]",
        "events_are_raw_pretransform",
        "transformed_events.parquet",
        "sample_article_like_slice",
        "manual_article_like_multi_slice",
        "assert_no_future_leakage",
        "CoLESClassifier",
        "make_class_weights",
        "extract_coles_embeddings",
        "train_coles_catboost",
        "CatBoostClassifier",
        "coles_classification_head",
        "coles_full_finetune",
        "coles_catboost",
        "coles_catboost_feature_importance.csv",
    ]
    for snippet in required_snippets:
        assert snippet in source


def test_public_benchmark_supervised_encoder_notebook_structure() -> None:
    notebook_path = Path("notebooks/kaggle_baseline_supervised_encoder_public_benchmarks.ipynb")
    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )

    required_snippets = [
        "SUPERVISED_ENCODER_EXPERIMENTS",
        "pip\", \"install\", \"torchmetrics\"",
        "events_are_raw_pretransform",
        "transformed_events.parquet",
        "supervised_full",
        "supervised_low_10pct",
        "scripts\" / \"finetune.py",
        "--label-fraction",
        "selection_metric",
        "runtime_base[\"model\"][\"max_seq_len\"] = runtime_base[\"data\"][\"max_seq_len\"]",
        "supervised_encoder_metrics.csv",
    ]
    for snippet in required_snippets:
        assert snippet in source


def test_public_benchmark_final_dme_notebook_structure() -> None:
    notebook_path = Path("notebooks/kaggle_final_dme_public_benchmarks.ipynb")
    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )

    required_snippets = [
        "FINAL_DME_ABLATION_PATH",
        "pip\", \"install\", \"torchmetrics\"",
        "events_are_raw_pretransform",
        "transformed_events.parquet",
        "A14_final_forecast_pretrain.yaml",
        "FINAL_DME_PUBLIC_EXPERIMENTS",
        "forecast_alpha030_reg012",
        "forecast_alpha030_reg015",
        "scripts\" / \"forecast_pretrain.py",
        "scripts\" / \"finetune.py",
        "cfg[\"model\"][\"max_seq_len\"] = cfg[\"data\"][\"max_seq_len\"]",
        "best_forecast_checkpoint.pt",
        "forecast_eval_metrics.json",
        "scenario_examples.json",
        "final_dme_metrics.csv",
        "final_dme_forecast_metrics.csv",
    ]
    for snippet in required_snippets:
        assert snippet in source
