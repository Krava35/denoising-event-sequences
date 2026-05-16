from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.corruption.pipeline import CorruptionPipeline
from src.data.collate import collate_fn
from src.data.dataset import EventSequenceDataset
from src.data.forecasting import (
    DERIVED_TIME_FEATURE_NAMES,
    build_forecast_stats,
    get_num_feature_dim,
    has_valid_forecast_cut,
    sample_forecast_cut,
    scenario_from_outputs,
)
from src.data.preprocessing import EventPreprocessor
from src.models.dme_encoder import DMEEncoder
from src.models.pooling import get_pooling
from src.models.tokenizer import MixedEventTokenizer
from src.training.forecast_pretrain import forecast_pretrain
from src.training.losses import compute_forecast_loss, compute_pretraining_loss
from src.utils.logging import MetricsLogger


def _make_df(n_entities: int = 8, events_per_entity: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    base_time = pd.Timestamp("2024-01-01")
    for entity_idx in range(n_entities):
        timestamp = base_time + pd.Timedelta(days=entity_idx)
        for event_idx in range(events_per_entity):
            timestamp += pd.Timedelta(hours=int(rng.integers(1, 8)))
            rows.append(
                {
                    "entity_id": f"e_{entity_idx}",
                    "timestamp": timestamp,
                    "event_type": int((event_idx + entity_idx) % 5),
                    "amount": float(rng.normal(100.0, 15.0)),
                    "cat_col": int(rng.integers(0, 4)),
                    "target": entity_idx % 2,
                }
            )
    return pd.DataFrame(rows)


def _config() -> dict:
    return {
        "data": {
            "event_type_col": "event_type",
            "timestamp_col": "timestamp",
            "numerical_cols": ["amount"],
            "categorical_cols": ["cat_col"],
            "group_col": "entity_id",
            "target_col": "target",
            "max_seq_len": 12,
            "min_vocab_count": 1,
            "amount_cols": ["amount"],
            "robust_scale_cols": ["amount"],
            "use_time_features": True,
            "time_feature_mode": "derived_numeric",
        },
        "model": {
            "event_type_emb_dim": 12,
            "cat_emb_dim": 8,
            "num_projection_dim": 12,
            "time_projection_dim": 12,
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 4,
            "dim_feedforward": 64,
            "dropout": 0.0,
            "activation": "gelu",
            "max_seq_len": 16,
            "use_profile_token": True,
            "client_embedding_dim": 24,
        },
        "pooling": {
            "type": "multi",
            "components": ["profile", "gated_attention", "mean", "max", "last"],
        },
        "forecasting": {
            "enabled": True,
            "cut_min_ratio": 0.50,
            "cut_max_ratio": 0.80,
            "min_future_events": 2,
            "count_num_buckets": 6,
            "gap_num_buckets": 6,
            "alpha_forecast": 0.2,
            "lambda_event_type_profile": 1.0,
            "lambda_count": 0.5,
            "lambda_amount": 0.5,
            "lambda_gap": 0.3,
            "lambda_cat_profile": 0.5,
        },
        "corruption": {
            "event_level_masking": {"prob": 0.0},
            "event_type": {
                "selected_prob": 0.40,
                "mask_prob": 0.20,
                "transition_replace_prob": 0.0,
                "random_replace_prob": 0.10,
                "keep_predict_prob": 0.10,
                "use_transition_aware_replacement": False,
            },
            "categorical_features": {"mask_prob": 0.10, "random_replace_prob": 0.0},
            "time_noise": {"corruption_prob": 0.0},
            "numerical_noise": {"corruption_prob": 0.0},
        },
        "loss": {
            "lambda_event_type": 1.0,
            "lambda_time": 1.0,
            "lambda_num": 0.5,
            "lambda_cat": 0.5,
            "lambda_exist": 0.1,
        },
    }


def _bundle() -> tuple[pd.DataFrame, list[str], EventPreprocessor, dict, dict]:
    cfg = _config()
    df = _make_df()
    entity_ids = df["entity_id"].unique().tolist()
    prep = EventPreprocessor(cfg)
    prep.fit(df, entity_ids)
    stats = build_forecast_stats(df, entity_ids, prep, cfg)
    return df, entity_ids, prep, cfg, stats


def _vocab_info(prep: EventPreprocessor, cfg: dict) -> dict:
    return {
        "event_type_vocab_size": len(prep.vocab[prep.event_type_col]),
        "cat_vocab_sizes": [len(prep.vocab[col]) for col in prep.categorical_cols],
        "num_num_features": get_num_feature_dim(prep, cfg),
        "num_classes": 2,
    }


def test_forecast_stats_train_only() -> None:
    cfg = _config()
    df = _make_df(n_entities=4, events_per_entity=8)
    df.loc[df["entity_id"] == "e_3", "event_type"] = 999
    train_ids = ["e_0", "e_1", "e_2"]

    prep = EventPreprocessor(cfg)
    prep.fit(df, train_ids)
    stats = build_forecast_stats(df, train_ids, prep, cfg)

    unk_id = prep.vocab[prep.event_type_col]["<UNK>"]
    assert stats["event_type_global_freq"][unk_id] < 1e-4
    assert "999" not in prep.vocab[prep.event_type_col]


def test_forecast_cut_validity() -> None:
    cfg = _config()
    n_events = 10
    assert has_valid_forecast_cut(n_events, cfg)
    for _ in range(50):
        cut = sample_forecast_cut(n_events, cfg)
        assert 1 <= cut < n_events
        assert n_events - cut >= cfg["forecasting"]["min_future_events"]


def test_tokenizer_profile_token_prepends_sequence() -> None:
    torch.manual_seed(0)
    tokenizer = MixedEventTokenizer(
        event_type_vocab_size=10,
        event_type_emb_dim=8,
        cat_vocab_sizes=[7],
        cat_emb_dim=4,
        num_num_features=3,
        num_projection_dim=8,
        time_projection_dim=8,
        hidden_dim=16,
        max_seq_len=8,
        dropout=0.0,
        use_profile_token=True,
    )
    out = tokenizer(
        event_type=torch.randint(1, 10, (2, 5)),
        time_delta=torch.randn(2, 5),
        num_features=torch.randn(2, 5, 3),
        cat_features=torch.randint(1, 7, (2, 5, 1)),
        attention_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    assert out.shape == (2, 6, 16)
    assert out.isfinite().all()


def test_multi_pooling_returns_client_embedding_dim() -> None:
    pool = get_pooling(
        "multi",
        16,
        client_embedding_dim=20,
        components=["profile", "gated_attention", "mean", "max", "last"],
        has_profile_token=True,
        dropout=0.0,
    )
    x = torch.randn(3, 7, 16)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[0, 5:] = False
    out = pool(x, mask)
    assert out.shape == (3, 20)
    assert out.isfinite().all()


def test_derived_time_features_are_appended_and_finite() -> None:
    df, entity_ids, prep, cfg, _ = _bundle()
    dataset = EventSequenceDataset(df, entity_ids, prep, cfg, mode="finetune")
    sample = dataset[0]
    expected_dim = 1 + len(DERIVED_TIME_FEATURE_NAMES)
    assert sample["num_features"].shape[-1] == expected_dim
    assert sample["num_features"].isfinite().all()


def test_forecast_dataset_targets_and_collate() -> None:
    df, entity_ids, prep, cfg, stats = _bundle()
    dataset = EventSequenceDataset(
        df, entity_ids, prep, cfg, mode="forecast", forecast_stats=stats
    )
    sample = dataset[0]
    targets = sample["forecast_targets"]

    assert sample["event_type"].shape[0] <= cfg["data"]["max_seq_len"]
    assert targets["future_event_type_profile"].shape == (len(prep.vocab[prep.event_type_col]),)
    assert targets["future_amount_stats"].shape == (4,)
    assert len(targets["future_cat_profiles"]) == 1
    assert targets["future_event_type_profile"].isfinite().all()
    assert targets["future_amount_stats"].isfinite().all()
    assert 0 <= int(targets["future_count_bucket"]) < cfg["forecasting"]["count_num_buckets"]
    assert 0 <= int(targets["future_gap_bucket"]) < cfg["forecasting"]["gap_num_buckets"]

    batch = collate_fn([dataset[0], dataset[1]])
    forecast_targets = batch["forecast_targets"]
    assert forecast_targets["future_event_type_profile"].ndim == 2
    assert forecast_targets["future_count_bucket"].shape == (2,)
    assert forecast_targets["future_gap_bucket"].shape == (2,)
    assert forecast_targets["future_cat_profiles"][0].ndim == 2


def test_forecast_heads_and_loss_backward() -> None:
    torch.manual_seed(0)
    df, entity_ids, prep, cfg, stats = _bundle()
    dataset = EventSequenceDataset(
        df, entity_ids, prep, cfg, mode="forecast", forecast_stats=stats
    )
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_fn)))
    model = DMEEncoder(cfg, _vocab_info(prep, cfg))

    pretrain_out = model(batch, mode="pretrain")
    B, L = batch["event_type"].shape
    assert pretrain_out["event_type_logits"].shape == (
        B,
        L,
        len(prep.vocab[prep.event_type_col]),
    )
    assert pretrain_out["hidden_states"].shape == (B, L, cfg["model"]["hidden_dim"])

    forecast_out = model(batch, mode="forecast")
    assert forecast_out["representation"].shape == (B, cfg["model"]["client_embedding_dim"])
    assert forecast_out["future_event_type_profile"].shape == (
        B,
        len(prep.vocab[prep.event_type_col]),
    )
    assert forecast_out["future_count_bucket_logits"].shape == (B, 6)
    assert forecast_out["future_amount_stats"].shape == (B, 4)
    assert forecast_out["future_gap_bucket_logits"].shape == (B, 6)
    assert len(forecast_out["future_cat_profiles"]) == 1

    loss = compute_forecast_loss(forecast_out, batch["forecast_targets"], cfg)
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(g.isfinite().all() for g in grads)


def test_scenario_constraints() -> None:
    torch.manual_seed(0)
    df, entity_ids, prep, cfg, stats = _bundle()
    model = DMEEncoder(cfg, _vocab_info(prep, cfg))
    dataset = EventSequenceDataset(
        df, entity_ids, prep, cfg, mode="forecast", forecast_stats=stats
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_fn)))
    outputs = model(batch, mode="forecast")

    scenario = scenario_from_outputs(outputs, stats, prep.vocab, top_k=3)
    valid_event_ids = set(prep.vocab[prep.event_type_col].values())
    valid_cat_ids = set(prep.vocab[prep.categorical_cols[0]].values())

    assert scenario["future_count_bucket"] in stats["count_bucket_labels"]
    assert scenario["gap_bucket"] in stats["gap_bucket_labels"]
    assert len(scenario["dominant_event_types"]) <= 3
    assert all(item["id"] in valid_event_ids for item in scenario["dominant_event_types"])
    assert scenario["expected_amount_stats"]["std"] >= 0.0
    assert 0.0 <= scenario["expected_amount_stats"]["positive_share"] <= 1.0
    cat_profile = scenario["categorical_profiles"][prep.categorical_cols[0]]
    assert all(item["id"] in valid_cat_ids for item in cat_profile)
    assert all(item["probability"] >= 0.0 for item in scenario["dominant_event_types"])


def test_hybrid_forecast_smoke_backward() -> None:
    torch.manual_seed(0)
    df, entity_ids, prep, cfg, stats = _bundle()
    dataset = EventSequenceDataset(
        df, entity_ids, prep, cfg, mode="forecast", forecast_stats=stats
    )
    batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate_fn)))
    vocab_info = _vocab_info(prep, cfg)
    model = DMEEncoder(cfg, vocab_info)
    pipe = CorruptionPipeline(
        cfg["corruption"],
        vocab_sizes={
            "event_type": vocab_info["event_type_vocab_size"],
            "cat_features": vocab_info["cat_vocab_sizes"],
        },
    )

    corrupted, targets, masks = pipe(batch)
    masks["attention_mask"] = corrupted["attention_mask"]
    denoise_out = model(corrupted, mode="pretrain")
    denoise_loss = compute_pretraining_loss(denoise_out, targets, masks, cfg)

    forecast_out = model(batch, mode="forecast")
    forecast_loss = compute_forecast_loss(forecast_out, batch["forecast_targets"], cfg)
    total = denoise_loss["total"] + 0.2 * forecast_loss["total"]
    assert torch.isfinite(total)
    total.backward()


def test_forecast_pretrain_one_epoch_saves_checkpoint_and_metrics(tmp_path) -> None:
    torch.manual_seed(0)
    df, entity_ids, prep, cfg, stats = _bundle()
    cfg["training"] = {
        "num_epochs_pretrain": 1,
        "batch_size": 4,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "gradient_clip_val": 1.0,
        "mixed_precision": False,
        "log_every_n_steps": 1,
    }
    train_ids = entity_ids[:6]
    val_ids = entity_ids[6:]

    train_loader = DataLoader(
        EventSequenceDataset(df, train_ids, prep, cfg, mode="forecast", forecast_stats=stats),
        batch_size=4,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        EventSequenceDataset(df, val_ids, prep, cfg, mode="forecast", forecast_stats=stats),
        batch_size=2,
        collate_fn=collate_fn,
    )
    vocab_info = _vocab_info(prep, cfg)
    model = DMEEncoder(cfg, vocab_info)
    pipe = CorruptionPipeline(
        cfg["corruption"],
        vocab_sizes={
            "event_type": vocab_info["event_type_vocab_size"],
            "cat_features": vocab_info["cat_vocab_sizes"],
        },
    )
    metrics_logger = MetricsLogger(str(tmp_path / "logs"), "forecast_smoke")

    ckpt = forecast_pretrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        corruption_pipeline=pipe,
        config=cfg,
        output_dir=str(tmp_path / "checkpoints"),
        device=torch.device("cpu"),
        logger=metrics_logger,
        vocab_info=vocab_info,
    )

    assert (tmp_path / "checkpoints" / "best_forecast_checkpoint.pt").exists()
    assert ckpt.endswith("best_forecast_checkpoint.pt")
    assert metrics_logger.metrics_path.exists()
    assert metrics_logger.metrics_path.read_text().strip()
