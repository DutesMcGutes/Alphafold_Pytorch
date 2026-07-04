from __future__ import annotations

import torch
from torch import nn

from alphafold3_pytorch.model.config import AlphaFold3Config


class DiffusionCoordinateHead(nn.Module):
    def __init__(self, config: AlphaFold3Config) -> None:
        super().__init__()
        self.steps = config.diffusion_steps
        self.denoiser = nn.Sequential(
            nn.LayerNorm(config.token_dim),
            nn.Linear(config.token_dim, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, 3),
        )

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        coords = torch.zeros(token.shape[0], token.shape[1], 3, device=token.device, dtype=token.dtype)
        update = self.denoiser(token)
        for _ in range(self.steps):
            coords = coords + update / max(self.steps, 1)
        return coords
