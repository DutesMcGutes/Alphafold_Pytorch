# AlphaFold3 PyTorch

A pure PyTorch scaffold for AlphaFold 3-style inference, built with an eye toward eventually loading the real AlphaFold 3 weights from Google DeepMind.

This lives separately from `Alphafold3_Repro` on purpose:

- `Alphafold3_Repro` is the teaching repo: tutorials, exercises, runnable demos.
- `Alphafold3_PyTorch` is just the model: PyTorch modules and, eventually, weight compatibility, with none of the tutorial scaffolding.

## Where things stand

This is a model-only scaffold right now. It lays out the package boundaries, not a finished model:

- input embedding
- a Pairformer-style trunk
- a diffusion-style coordinate module
- an atom prediction head
- a full model wrapper
- a DeepMind weight-conversion adapter (not built yet)

No AlphaFold 3 weights ship with this repo.

## Weight policy

Official AlphaFold 3 weights have to come directly from Google DeepMind, under their terms. This repo will never commit, mirror, or redistribute weights or converted checkpoints. That's a hard line, not a TODO.

## Install

```bash
pip install -e .
```

## Reference projects

- https://github.com/google-deepmind/alphafold3
