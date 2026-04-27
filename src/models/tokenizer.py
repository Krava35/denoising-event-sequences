from __future__ import annotations

import torch
import torch.nn as nn


class MixedEventTokenizer(nn.Module):
    def __init__(
        self,
        event_type_vocab_size: int,
        event_type_emb_dim: int = 64,
        cat_vocab_sizes: list[int] | None = None,
        cat_emb_dim: int = 32,
        num_num_features: int = 0,
        num_projection_dim: int = 64,
        time_projection_dim: int = 64,
        hidden_dim: int = 256,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        cat_vocab_sizes = cat_vocab_sizes or []

        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id

        self.event_type_embedding = nn.Embedding(event_type_vocab_size, event_type_emb_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.time_projection = nn.Sequential(nn.Linear(1, time_projection_dim), nn.GELU())

        if num_num_features > 0:
            self.num_projection: nn.Module | None = nn.Sequential(
                nn.Linear(num_num_features, num_projection_dim), nn.GELU()
            )
        else:
            self.num_projection = None

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(v, cat_emb_dim) for v in cat_vocab_sizes]
        )

        total_dim = event_type_emb_dim + time_projection_dim
        if num_num_features > 0:
            total_dim += num_projection_dim
        total_dim += cat_emb_dim * len(cat_vocab_sizes)

        self.output_projection = nn.Linear(total_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        event_type: torch.LongTensor,               # [B, L]
        time_delta: torch.FloatTensor,              # [B, L]
        num_features: torch.FloatTensor | None = None,  # [B, L, N]
        cat_features: torch.LongTensor | None = None,   # [B, L, C]
        attention_mask: torch.BoolTensor | None = None, # [B, L]
    ) -> torch.FloatTensor:                         # [B, L, hidden_dim]
        B, L = event_type.shape

        # [B, L, event_type_emb_dim]
        event_type_emb = self.event_type_embedding(event_type)

        # [B, L, 1] → [B, L, time_projection_dim]
        log_time = torch.log1p(time_delta.clamp(min=0)).unsqueeze(-1)
        time_emb = self.time_projection(log_time)

        # [1, L, hidden_dim] — добавляется после проекции
        positions = torch.arange(L, device=event_type.device)
        pos_emb = self.position_embedding(positions).unsqueeze(0)

        components: list[torch.Tensor] = [event_type_emb, time_emb]

        if self.num_projection is not None and num_features is not None:
            # [B, L, num_projection_dim]
            components.append(self.num_projection(num_features))

        if self.cat_embeddings and cat_features is not None:
            for j, emb_layer in enumerate(self.cat_embeddings):
                # [B, L, cat_emb_dim]
                components.append(emb_layer(cat_features[..., j]))

        # [B, L, total_dim]
        x = torch.cat(components, dim=-1)

        # [B, L, hidden_dim]
        x = self.output_projection(x)
        x = x + pos_emb

        x = self.layer_norm(x)
        x = self.dropout(x)

        if attention_mask is not None:
            # [B, L, 1] → обнулить паддинг-позиции
            x = x * attention_mask.unsqueeze(-1).float()

        assert x.shape == (B, L, self.hidden_dim), (
            f"Expected ({B}, {L}, {self.hidden_dim}), got {x.shape}"
        )
        return x
