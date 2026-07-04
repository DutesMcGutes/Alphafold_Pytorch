from __future__ import annotations

import torch
from torch import nn

from alphafold3_pytorch.model.config import AlphaFold3Config


class AtomHead(nn.Module):
    def __init__(self, config: AlphaFold3Config) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(config.token_dim),
            nn.Linear(config.token_dim, config.atom_dim),
            nn.GELU(),
            nn.Linear(config.atom_dim, 4 * 3),
        )

    def forward(self, token: torch.Tensor, token_coords: torch.Tensor) -> torch.Tensor:
        offsets = self.proj(token).reshape(token.shape[0], token.shape[1], 4, 3)
        return token_coords[:, :, None, :] + offsets
