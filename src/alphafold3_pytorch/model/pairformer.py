from __future__ import annotations

import torch
from torch import nn

from alphafold3_pytorch.model.config import AlphaFold3Config


class PairformerBlock(nn.Module):
    def __init__(self, config: AlphaFold3Config) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(config.token_dim)
        self.pair_norm = nn.LayerNorm(config.pair_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(config.token_dim, config.token_dim * 4),
            nn.GELU(),
            nn.Linear(config.token_dim * 4, config.token_dim),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(config.pair_dim, config.pair_dim * 4),
            nn.GELU(),
            nn.Linear(config.pair_dim * 4, config.pair_dim),
        )
        self.pair_to_token = nn.Linear(config.pair_dim, config.token_dim)

    def forward(self, token: torch.Tensor, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair_update = self.pair_mlp(self.pair_norm(pair))
        pair = pair + pair_update
        pooled_pair = pair.mean(dim=2)
        token_update = self.token_mlp(self.token_norm(token)) + self.pair_to_token(pooled_pair)
        token = token + token_update
        return token, pair


class PairformerTrunk(nn.Module):
    def __init__(self, config: AlphaFold3Config) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(PairformerBlock(config) for _ in range(config.trunk_blocks))

    def forward(self, token: torch.Tensor, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            token, pair = block(token, pair)
        return token, pair
