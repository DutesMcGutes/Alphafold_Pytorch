from __future__ import annotations

import torch
from torch import nn

from alphafold3_pytorch.model.atom_head import AtomHead
from alphafold3_pytorch.model.config import AlphaFold3Config
from alphafold3_pytorch.model.diffusion import DiffusionCoordinateHead
from alphafold3_pytorch.model.input_embedder import InputEmbedder
from alphafold3_pytorch.model.pairformer import PairformerTrunk


class AlphaFold3Model(nn.Module):
    def __init__(self, config: AlphaFold3Config | None = None) -> None:
        super().__init__()
        self.config = config or AlphaFold3Config()
        self.embedder = InputEmbedder(self.config)
        self.trunk = PairformerTrunk(self.config)
        self.diffusion = DiffusionCoordinateHead(self.config)
        self.atom_head = AtomHead(self.config)

    def forward(self, residue_type: torch.Tensor) -> dict[str, torch.Tensor]:
        token, pair = self.embedder(residue_type)
        token, pair = self.trunk(token, pair)
        token_coords = self.diffusion(token)
        atom_coords = self.atom_head(token, token_coords)
        return {"token": token, "pair": pair, "token_coords": token_coords, "atom_coords": atom_coords}
