import logging

import torch

logger = logging.getLogger(__name__)


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

    assert event_type.shape == (B, max_L)
    assert time_delta.shape == (B, max_L)
    assert num_features.shape == (B, max_L, n_num)
    assert cat_features.shape == (B, max_L, n_cat)
    assert attention_mask.shape == (B, max_L)
    assert labels.shape == (B,)

    return {
        "event_type": event_type,
        "time_delta": time_delta,
        "num_features": num_features,
        "cat_features": cat_features,
        "attention_mask": attention_mask,
        "label": labels,
        "entity_id": entity_ids,
    }
