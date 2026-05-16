from __future__ import annotations

import json
import pickle

import pandas as pd
import yaml

from scripts.build_transition_matrix import _load_train_sequences
from scripts.prepare_public_benchmark_data import (
    build_public_benchmark_config,
    make_age_group_events,
    make_gender_events,
    parse_gender_tr_datetime,
    prepare_public_benchmark_dataset,
)
from src.data.dataset import EventSequenceDataset
from src.data.preprocessing import EventPreprocessor
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
    assert gender_cfg["model"]["max_seq_len"] == gender_cfg["data"]["max_seq_len"]
    assert gender_cfg["data"]["categorical_cols"] == ["tr_type"]
    assert gender_cfg["training"]["num_classes"] == 2
    assert gender_cfg["training"]["selection_metric"] == "roc_auc"
    assert age_cfg["data"]["timestamp_col"] == "event_timestamp"
    assert age_cfg["model"]["max_seq_len"] == age_cfg["data"]["max_seq_len"]
    assert age_cfg["training"]["num_classes"] == 4
    assert age_cfg["training"]["selection_metric"] == "macro_f1"


def test_prepare_public_benchmark_dataset_writes_raw_events_and_transformed_cache(
    tmp_path,
) -> None:
    raw_dir = tmp_path / "raw" / "gender"
    raw_dir.mkdir(parents=True)

    rows = []
    labels = []
    for customer_id in range(20):
        labels.append({"customer_id": customer_id, "gender": customer_id % 2})
        for event_idx in range(6):
            rows.append(
                {
                    "customer_id": customer_id,
                    "tr_datetime": f"{event_idx} 00:00:00",
                    "mcc_code": [5411, 6011, 4829][event_idx % 3],
                    "tr_type": [1010, 2020][event_idx % 2],
                    "amount": -10.0 * (event_idx + 1),
                    "term_id": f"terminal_{customer_id}_{event_idx}",
                }
            )
    pd.DataFrame(rows).to_csv(raw_dir / "transactions.csv.gz", index=False)
    pd.DataFrame(labels).to_csv(raw_dir / "gender_train.csv", index=False)

    out_dir = tmp_path / "processed" / "gender"
    report = prepare_public_benchmark_dataset(
        "gender",
        raw_root=tmp_path / "raw",
        output_dir=out_dir,
        max_seq_len=16,
    )

    events = pd.read_parquet(out_dir / "events.parquet")
    canonical = pd.read_parquet(out_dir / "canonical_events.parquet")
    transformed = pd.read_parquet(out_dir / "transformed_events.parquet")

    assert report["events_are_raw_pretransform"] is True
    assert (out_dir / "transformed_events.parquet").exists()
    assert events.equals(canonical)
    assert set(events["mcc_code"]) == {"5411", "6011", "4829"}
    assert set(events["tr_type"]) == {"1010", "2020"}
    assert "term_id" in events.columns
    assert transformed["mcc_code"].max() > 4
    assert transformed["tr_type"].max() > 4
    assert transformed["amount"].dtype.kind == "f"

    with (out_dir / "data_report.json").open() as f:
        saved_report = json.load(f)
    assert saved_report["events_are_raw_pretransform"] is True
    assert saved_report["transformed_events_path"].endswith("transformed_events.parquet")

    with (out_dir / "preprocessor.pkl").open("rb") as f:
        preprocessor = pickle.load(f)
    with (out_dir / "splits.json").open() as f:
        splits = json.load(f)
    with (out_dir / "prepared_config.yaml").open() as f:
        config = yaml.safe_load(f)

    dataset = EventSequenceDataset(
        events,
        splits["train"],
        preprocessor,
        config,
        mode="pretrain",
    )
    sample = dataset[0]
    assert int(sample["event_type"].max()) > preprocessor.MASK_EVENT
    assert preprocessor.UNK not in set(sample["event_type"].tolist())


def test_transition_matrix_loads_raw_events_with_preprocessor_vocab(tmp_path) -> None:
    config = build_public_benchmark_config("gender", max_seq_len=16)
    config["data"]["min_vocab_count"] = 1

    events = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2],
            "event_timestamp": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "tr_datetime": ["0 00:00:00", "1 00:00:00", "0 00:00:00", "1 00:00:00"],
            "mcc_code": ["5411", "6011", "5411", "4829"],
            "amount": [-10.0, -20.0, -30.0, -40.0],
            "tr_type": ["1010", "2020", "1010", "3030"],
            "term_id": ["t1", "t2", "t3", "t4"],
            "gender": [0, 0, 1, 1],
        }
    )
    preprocessor = EventPreprocessor(config)
    preprocessor.fit(events, [1, 2])

    events_path = tmp_path / "events.parquet"
    splits_path = tmp_path / "splits.json"
    preprocessor_path = tmp_path / "preprocessor.pkl"
    events.to_parquet(events_path, index=False)
    splits_path.write_text(json.dumps({"train": [1, 2], "val": [], "test": []}))
    with preprocessor_path.open("wb") as f:
        pickle.dump(preprocessor, f, protocol=pickle.HIGHEST_PROTOCOL)

    sequences, vocab_size = _load_train_sequences(
        events_path=events_path,
        splits_path=splits_path,
        event_type_col="mcc_code",
        entity_col="customer_id",
        timestamp_col="event_timestamp",
        preprocessor_path=preprocessor_path,
    )

    vocab = preprocessor.vocab["mcc_code"]
    assert vocab_size == len(vocab)
    assert sequences == [
        [vocab["5411"], vocab["6011"]],
        [vocab["5411"], vocab["4829"]],
    ]
    assert all(event_id != preprocessor.UNK for seq in sequences for event_id in seq)


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
