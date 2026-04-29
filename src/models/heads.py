from __future__ import annotations

import torch
import torch.nn as nn


class EventTypeHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H] → [B, L, V]
        return self.linear(x)


class TimeDeltaHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H] → [B, L, 1]
        return self.mlp(x)


class NumericalHead(nn.Module):
    def __init__(self, hidden_dim: int, num_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H] → [B, L, N]
        return self.linear(x)


class CategoricalHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_sizes: list[int]) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, v) for v in vocab_sizes]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: [B, L, H] → list of [B, L, V_j] per categorical feature
        return [head(x) for head in self.heads]


class ExistenceHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H] → [B, L, 1]
        return self.linear(x)


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H] (после pooling) → [B, num_classes]
        return self.classifier(x)


class FutureEventTypeProfileHead(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FutureCountBucketHead(nn.Module):
    def __init__(self, input_dim: int, num_buckets: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_buckets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FutureAmountStatsHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FutureGapBucketHead(nn.Module):
    def __init__(self, input_dim: int, num_buckets: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_buckets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FutureCategoricalProfileHead(nn.Module):
    def __init__(self, input_dim: int, vocab_sizes: list[int]) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(input_dim, vocab_size) for vocab_size in vocab_sizes])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [head(x) for head in self.heads]
