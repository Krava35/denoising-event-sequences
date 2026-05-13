from __future__ import annotations

import pandas as pd

from scripts.prepare_public_benchmark_data import (
    build_public_benchmark_config,
    make_age_group_events,
    make_gender_events,
    parse_gender_tr_datetime,
)
from src.evaluation.classification import compute_classification_metrics
from src.training.finetune import _select_validation_score


def test_parse_gender_tr_datetime_day_time_format() -> None:
    parsed = parse_gender_tr_datetime(pd.Series(["0 00:00:00", "2 03:04:05"]))

    assert parsed.iloc[0] == pd.Timestamp("2010-01-01 00:00:00")
    assert parsed.iloc[1] == pd.Timestamp("2010-01-03 03:04:05")


def test_make_gender_events_merges_labels_and_uniquifies_timestamps() -> None:
    transactions = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 3],
            "tr_datetime": ["0 01:00:00", "0 01:00:00", "1 02:00:00", "2 03:00:00"],
            "mcc_code": [5411, 5411, 6011, 9999],
            "tr_type": [1010, 1010, 2020, 3030],
            "amount": [-10.0, -20.0, 5.0, 7.0],
            "term_id": [float("nan"), 12.0, 13.0, 14.0],
        }
    )
    labels = pd.DataFrame({"customer_id": [1, 2], "gender": [0, 1]})

    events, report = make_gender_events(transactions, labels)

    assert set(events["customer_id"]) == {1, 2}
    assert events.duplicated(["customer_id", "event_timestamp"]).sum() == 0
    assert "__MISSING__" in set(events["term_id"])
    assert report["raw_duplicate_entity_timestamp_pairs"] == 1
    assert report["canonical_duplicate_entity_timestamp_pairs"] == 0


def test_make_age_group_events_merges_labels_and_uniquifies_timestamps() -> None:
    transactions = pd.DataFrame(
        {
            "client_id": [10, 10, 11, 12],
            "trans_date": [1, 1, 3, 5],
            "small_group": [1, 2, 3, 4],
            "amount_rur": [10.0, 20.0, float("nan"), 40.0],
        }
    )
    labels = pd.DataFrame({"client_id": [10, 11], "bins": [0, 3]})

    events, report = make_age_group_events(transactions, labels)

    assert set(events["client_id"]) == {10, 11}
    assert events.duplicated(["client_id", "event_timestamp"]).sum() == 0
    assert float(events.loc[events["client_id"] == 11, "amount_rur"].iloc[0]) == 0.0
    assert report["raw_duplicate_entity_timestamp_pairs"] == 1
    assert report["canonical_duplicate_entity_timestamp_pairs"] == 0


def test_public_benchmark_config_uses_canonical_timestamp_and_multiclass_metric() -> None:
    gender_cfg = build_public_benchmark_config("gender")
    age_cfg = build_public_benchmark_config("age_group")

    assert gender_cfg["data"]["timestamp_col"] == "event_timestamp"
    assert gender_cfg["training"]["num_classes"] == 2
    assert gender_cfg["training"]["selection_metric"] == "roc_auc"
    assert age_cfg["data"]["timestamp_col"] == "event_timestamp"
    assert age_cfg["training"]["num_classes"] == 4
    assert age_cfg["training"]["selection_metric"] == "macro_f1"


def test_multiclass_metrics_and_selection_metric_are_age_group_safe() -> None:
    y_true = pd.Series([0, 1, 2, 3]).to_numpy()
    y_pred_proba = pd.DataFrame(
        [
            [0.80, 0.10, 0.05, 0.05],
            [0.10, 0.70, 0.10, 0.10],
            [0.10, 0.10, 0.70, 0.10],
            [0.10, 0.10, 0.10, 0.70],
        ]
    ).to_numpy()

    metrics = compute_classification_metrics(y_true, y_pred_proba, num_classes=4)
    selection_metric, score = _select_validation_score(
        metrics, {"training": {"selection_metric": "macro_f1"}}
    )

    assert "roc_auc_ovr" in metrics
    assert "macro_pr_auc" in metrics
    assert selection_metric == "macro_f1"
    assert score == metrics["macro_f1"]
