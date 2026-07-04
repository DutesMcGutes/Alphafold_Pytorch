from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaFold3Config:
    token_dim: int = 384
    pair_dim: int = 128
    atom_dim: int = 128
    trunk_blocks: int = 4
    diffusion_steps: int = 20
    vocab_size: int = 32
