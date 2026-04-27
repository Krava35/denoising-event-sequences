from __future__ import annotations

import torch
import torch.nn as nn


class CLSPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:  # noqa: ARG002
        super().__init__()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H] → берём первый токен (CLS)
        return x[:, 0, :]  # [B, H]


class MeanPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:  # noqa: ARG002
        super().__init__()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H], attention_mask: [B, L] bool
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        sum_x = (x * mask).sum(dim=1)               # [B, H]
        count = mask.sum(dim=1).clamp(min=1.0)      # [B, 1]
        return sum_x / count                         # [B, H]


class MaxPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:  # noqa: ARG002
        super().__init__()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H], attention_mask: [B, L] bool
        # паддинг → -inf, чтобы исключить из max
        x = x.masked_fill(~attention_mask.unsqueeze(-1), float("-inf"))
        return x.max(dim=1).values  # [B, H]


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # a_i = softmax(W2 * tanh(W1 * h_i))
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H], attention_mask: [B, L] bool
        scores = self.attention(x).squeeze(-1)              # [B, L]
        scores = scores.masked_fill(~attention_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)              # [B, L]
        return (weights.unsqueeze(-1) * x).sum(dim=1)      # [B, H]


class GatedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score_linear = nn.Linear(hidden_dim, 1)
        self.gate_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H], attention_mask: [B, L] bool
        scores = self.score_linear(x).squeeze(-1)           # [B, L]
        scores = scores.masked_fill(~attention_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)              # [B, L]
        gates = torch.sigmoid(self.gate_linear(x))         # [B, L, H]
        return (weights.unsqueeze(-1) * gates * x).sum(dim=1)  # [B, H]


_REGISTRY: dict[str, type[nn.Module]] = {
    "cls": CLSPooling,
    "mean": MeanPooling,
    "max": MaxPooling,
    "attention": AttentionPooling,
    "gated_attention": GatedAttentionPooling,
}


def get_pooling(pooling_type: str, hidden_dim: int) -> nn.Module:
    if pooling_type not in _REGISTRY:
        raise ValueError(
            f"Unknown pooling type '{pooling_type}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[pooling_type](hidden_dim)
