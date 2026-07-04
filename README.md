# AlphaFold3 PyTorch

A pure PyTorch implementation scaffold for AlphaFold 3-style inference, designed to become compatible with official AlphaFold 3 weights obtained directly from Google DeepMind.

This repository is intentionally separate from the teaching repo `Alphafold3_Repro`:

- `Alphafold3_Repro`: tutorials, guidance, readable exercises, and runnable educational demos.
- `Alphafold3_PyTorch`: model files only, focused on PyTorch modules and weight compatibility.

## Status

This is a model-only scaffold. It defines the package boundaries for the implementation:

- input embedding
- Pairformer-style trunk
- diffusion-style coordinate module
- atom prediction head
- full model wrapper
- future DeepMind weight conversion adapter

It does not include AlphaFold 3 weights.

## Weight Policy

Official AlphaFold 3 weights must be obtained directly from Google DeepMind and used only under their terms. This repository must never commit, mirror, or redistribute model weights or converted checkpoints.

## Install

```bash
pip install -e .
```

## Reference Projects

- https://github.com/kilianmandon/alphafold3-pytorch
- https://github.com/kilianmandon/alphafold3-decoded
- https://github.com/google-deepmind/alphafold3
