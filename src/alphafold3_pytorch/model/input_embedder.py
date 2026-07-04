from __future__ import annotations

import torch
from torch import nn

from alphafold3_pytorch.model.config import AlphaFold3Config


class InputEmbedder(nn.Module):
    def __init__(self, config: AlphaFold3Config) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.token_dim)
        self.pair_projection = nn.Linear(config.token_dim * 2, config.pair_dim)

    def forward(self, residue_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.token_embedding(residue_type)
        left = token[:, :, None, :].expand(-1, -1, token.shape[1], -1)
        right = token[:, None, :, :].expand(-1, token.shape[1], -1, -1)
        pair = self.pair_projection(torch.cat([left, right], dim=-1))
        return token, pair
