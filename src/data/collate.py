import logging

import torch

logger = logging.getLogger(__name__)


def _collate_forecast_targets(batch: list[dict]) -> dict:
    targets = [item["forecast_targets"] for item in batch]
    result: dict = {
        "future_event_type_profile": torch.stack(
            [target["future_event_type_profile"] for target in targets]
        ),
        "future_count_bucket": torch.stack(
            [target["future_count_bucket"] for target in targets]
        ).long(),
        "future_amount_stats": torch.stack(
            [target["future_amount_stats"] for target in targets]
        ),
        "future_gap_bucket": torch.stack(
            [target["future_gap_bucket"] for target in targets]
        ).long(),
    }

    first_cat_profiles = targets[0].get("future_cat_profiles", [])
    result["future_cat_profiles"] = [
        torch.stack([target["future_cat_profiles"][j] for target in targets])
        for j in range(len(first_cat_profiles))
    ]
    return result


def collate_fn(batch: list[dict]) -> dict:
    B = len(batch)
    max_L = max(item["event_type"].shape[0] for item in batch)
    n_num = batch[0]["num_features"].shape[-1]
    n_cat = batch[0]["cat_features"].shape[-1]

    event_type = torch.zeros(B, max_L, dtype=torch.long)
    time_delta = torch.zeros(B, max_L, dtype=torch.float)
    num_features = torch.zeros(B, max_L, n_num, dtype=torch.float)
    cat_features = torch.zeros(B, max_L, n_cat, dtype=torch.long)
    attention_mask = torch.zeros(B, max_L, dtype=torch.bool)

    for i, item in enumerate(batch):
        L = item["event_type"].shape[0]
        event_type[i, :L] = item["event_type"]
        time_delta[i, :L] = item["time_delta"]
        if n_num > 0:
            num_features[i, :L] = item["num_features"]
        if n_cat > 0:
            cat_features[i, :L] = item["cat_features"]
        attention_mask[i, :L] = item["attention_mask"]

    labels = torch.stack([item["label"] for item in batch])
    entity_ids = [item["entity_id"] for item in batch]

    if event_type.shape != (B, max_L):
        raise ValueError(f"event_type shape {event_type.shape} != ({B}, {max_L})")
    if time_delta.shape != (B, max_L):
        raise ValueError(f"time_delta shape {time_delta.shape} != ({B}, {max_L})")
    if num_features.shape != (B, max_L, n_num):
        raise ValueError(f"num_features shape {num_features.shape} != ({B}, {max_L}, {n_num})")
    if cat_features.shape != (B, max_L, n_cat):
        raise ValueError(f"cat_features shape {cat_features.shape} != ({B}, {max_L}, {n_cat})")
    if attention_mask.shape != (B, max_L):
        raise ValueError(f"attention_mask shape {attention_mask.shape} != ({B}, {max_L})")
    if labels.shape != (B,):
        raise ValueError(f"labels shape {labels.shape} != ({B},)")

    result = {
        "event_type": event_type,
        "time_delta": time_delta,
        "num_features": num_features,
        "cat_features": cat_features,
        "attention_mask": attention_mask,
        "label": labels,
        "entity_id": entity_ids,
    }

    if "forecast_targets" in batch[0]:
        result["forecast_targets"] = _collate_forecast_targets(batch)
    if "forecast_cut" in batch[0]:
        result["forecast_cut"] = torch.stack([item["forecast_cut"] for item in batch]).long()

    return result
