from __future__ import annotations

import torch
import torch.nn as nn


class TimeAwareTransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # pre-norm: стабильнее при смешанной точности
        )
        # enable_nested_tensor несовместим с norm_first=True
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(
        self,
        x: torch.Tensor,              # [B, L, H]
        attention_mask: torch.Tensor, # [B, L], True=реальный токен, False=паддинг
    ) -> torch.Tensor:               # [B, L, H]
        # PyTorch ожидает: src_key_padding_mask True=игнорировать → инвертируем
        # Булева маска безопасна для fp16: внутри PyTorch сам управляет dtype
        src_key_padding_mask = ~attention_mask  # [B, L]

        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
        return out
