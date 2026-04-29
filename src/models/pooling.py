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


class MultiPoolingProjection(nn.Module):
    _VALID_COMPONENTS = {"profile", "gated_attention", "mean", "max", "last"}

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        components: list[str] | None = None,
        dropout: float = 0.1,
        has_profile_token: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.components = components or ["profile", "gated_attention", "mean", "max", "last"]
        self.has_profile_token = has_profile_token

        unknown = sorted(set(self.components) - self._VALID_COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown multi-pooling components: {unknown}")
        if not self.components:
            raise ValueError("MultiPoolingProjection requires at least one component")

        self.gated_attention = (
            GatedAttentionPooling(hidden_dim) if "gated_attention" in self.components else None
        )
        in_dim = hidden_dim * len(self.components)
        mid_dim = max(hidden_dim, output_dim)
        self.projection = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _split_profile(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if self.has_profile_token:
            return x[:, 0, :], x[:, 1:, :], attention_mask[:, 1:]
        return None, x, attention_mask

    @staticmethod
    def _mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _max_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        masked = x.masked_fill(~attention_mask.unsqueeze(-1), float("-inf"))
        out = masked.max(dim=1).values
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

    @staticmethod
    def _last_valid_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        rows = torch.arange(x.shape[0], device=x.device)
        return x[rows, lengths]

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        profile, event_x, event_mask = self._split_profile(x, attention_mask)
        pooled: list[torch.Tensor] = []

        for component in self.components:
            if component == "profile":
                pooled.append(profile if profile is not None else event_x[:, 0, :])
            elif component == "gated_attention":
                pooled.append(self.gated_attention(event_x, event_mask))
            elif component == "mean":
                pooled.append(self._mean_pool(event_x, event_mask))
            elif component == "max":
                pooled.append(self._max_pool(event_x, event_mask))
            elif component == "last":
                pooled.append(self._last_valid_pool(event_x, event_mask))

        concat = torch.cat(pooled, dim=-1)
        return self.projection(concat)


_REGISTRY: dict[str, type[nn.Module]] = {
    "cls": CLSPooling,
    "mean": MeanPooling,
    "max": MaxPooling,
    "attention": AttentionPooling,
    "gated_attention": GatedAttentionPooling,
}


def get_pooling(
    pooling_type: str,
    hidden_dim: int,
    *,
    client_embedding_dim: int | None = None,
    components: list[str] | None = None,
    dropout: float = 0.1,
    has_profile_token: bool = False,
) -> nn.Module:
    if pooling_type == "multi":
        return MultiPoolingProjection(
            hidden_dim=hidden_dim,
            output_dim=client_embedding_dim or hidden_dim,
            components=components,
            dropout=dropout,
            has_profile_token=has_profile_token,
        )
    if pooling_type not in _REGISTRY:
        raise ValueError(
            f"Unknown pooling type '{pooling_type}'. Available: {list(_REGISTRY) + ['multi']}"
        )
    return _REGISTRY[pooling_type](hidden_dim)
