from __future__ import annotations

import torch

from alphafold3_pytorch import AlphaFold3Config, AlphaFold3Model


def test_forward_shapes() -> None:
    model = AlphaFold3Model(AlphaFold3Config(token_dim=32, pair_dim=16, atom_dim=16, trunk_blocks=1))
    residue_type = torch.randint(0, 20, (2, 5))
    out = model(residue_type)
    assert out["token"].shape == (2, 5, 32)
    assert out["pair"].shape == (2, 5, 5, 16)
    assert out["token_coords"].shape == (2, 5, 3)
    assert out["atom_coords"].shape == (2, 5, 4, 3)
