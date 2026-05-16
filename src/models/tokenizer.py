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
        use_profile_token: bool = False,
    ) -> None:
        super().__init__()
        cat_vocab_sizes = cat_vocab_sizes or []

        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.use_profile_token = use_profile_token

        self.event_type_embedding = nn.Embedding(event_type_vocab_size, event_type_emb_dim)
        self.position_embedding = nn.Embedding(
            max_seq_len + (1 if use_profile_token else 0), hidden_dim
        )
        self.profile_token = (
            nn.Parameter(torch.zeros(1, 1, hidden_dim)) if use_profile_token else None
        )
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
    ) -> torch.FloatTensor:                         # [B, L(+1), hidden_dim]
        B, L = event_type.shape

        # [B, L, event_type_emb_dim]
        event_type_emb = self.event_type_embedding(event_type)

        # time_delta is already transformed and scaled by EventPreprocessor.
        # Do not clamp/log it again: robust-scaled values can be negative.
        time_emb = self.time_projection(time_delta.unsqueeze(-1))

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

        if self.use_profile_token:
            event_positions = torch.arange(1, L + 1, device=event_type.device)
            event_pos_emb = self.position_embedding(event_positions).unsqueeze(0)
            profile_pos_emb = self.position_embedding(
                torch.zeros(1, dtype=torch.long, device=event_type.device)
            ).unsqueeze(0)
            profile = self.profile_token.expand(B, 1, self.hidden_dim) + profile_pos_emb
            x = torch.cat([profile, x + event_pos_emb], dim=1)
        else:
            positions = torch.arange(L, device=event_type.device)
            pos_emb = self.position_embedding(positions).unsqueeze(0)
            x = x + pos_emb

        x = self.layer_norm(x)
        x = self.dropout(x)

        if attention_mask is not None:
            if self.use_profile_token:
                profile_mask = torch.ones(
                    B, 1, dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([profile_mask, attention_mask], dim=1)
            x = x * attention_mask.unsqueeze(-1).float()

        expected_len = L + (1 if self.use_profile_token else 0)
        assert x.shape == (B, expected_len, self.hidden_dim), (
            f"Expected ({B}, {expected_len}, {self.hidden_dim}), got {x.shape}"
        )
        return x
