# bs-mamba2-infer maintainer guide

This is a standalone, inference-only port of Euiyeon Kim and Yong-Hoon Choi's
BSMamba2 vocal separator, based on `EuiYeonKim/BSMamba2` revision
`42eb8c84a1bf0d388a994b4c29edd0d9d6a0a2b5` (MIT). The associated paper is
“Mamba2 Meets Silence: Robust Vocal Source Separation for Sparse Regions,”
arXiv:2508.14556. The maintained scope is only the BSMamba2/Mamba2 graph and
vocal inference; Mamba v1, BS-RoFormer, TS-BSmamba2, training, datasets,
evaluation, Lightning, Hydra, W&B, TensorBoard, and bundled weights do not ship.

## Layout and conventions

- `src/bs_mamba2_infer/model.py` owns the graph and pure-PyTorch fallback.
- `audio.py` owns waveform normalization, STFT/iSTFT, and overlap-add geometry.
- `checkpoint.py` owns the TOML registry, resolver, cache, download, and hash.
- `api.py` owns the public lifecycle; `load()` is load-once, `infer()` is
  ready-only, `release()` permits reload, and `close()` is terminal/idempotent.
- Every load-bearing source file has a nav header. Do not add training/eval
  surfaces or core compiled dependencies.

## Backends (Torch / MLX)

Optional, additive `backend=` axis on `BSMamba2Session`/`separate`: `None`/
`"torch"` (default, unchanged), `"mlx"` (Apple Silicon, needs the `[mlx]`
extra), or `"auto"`. `device` keeps its existing Torch-only meaning
regardless of `backend` — it is never overloaded to mean "Apple Silicon";
`backend="mlx"` only accepts `None`/`"auto"`/`"mps"` for `device` and raises
for anything else (see README's "Backends (Torch / MLX)" for the public
contract).

- `backends/` — the compute seam. `base.py` holds the `SeparationBackend`
  protocol (one whole waveform in, the vocals stem out — this package has
  no multi-stem dict, unlike sibling packages, since `audio.py` itself
  always extracts mask-estimator index 0 regardless of
  `BSMamba2Config.num_stems`) and `BackendUnavailable`. `torch_backend.py`
  is `BSMamba2Session.infer()`'s pre-existing body (device-aware AMP,
  `separate_waveform`) moved behind the seam verbatim. `mlx_backend.py` is
  the MLX path, built from an *already-loaded* Torch `BSMamba2`
  (`MLXBackend.from_torch_model`) — it never touches a checkpoint path or
  raw state dict; `api.py`'s `load()` keeps sole ownership of checkpoint
  resolution and the `"model."`-prefix/featurizer-key state-dict filtering,
  even for `backend="mlx"` sessions (the Torch model built there is
  disposable scaffolding, read only for its `state_dict()`).
- `mlx/` — **from-scratch port**, not vendored: there is no upstream MLX
  BSMamba2 to take. `model.py` mirrors `model.py`'s frequency-domain graph
  (real-valued internally; complex only at the input/output edges, handled
  as an explicit real/imag pair rather than MLX complex dtype) and
  deliberately vectorizes the Mamba2 recurrence across its 16 heads (one
  batched einsum per timestep instead of Torch's nested per-head Python
  loop) — verified numerically against Torch, not merely assumed safe to
  change. `convert.py` maps Torch's state-dict keys to this module tree and
  `load_converted_weights()` raises on any unmatched or dropped tensor
  rather than silently partial-loading (`model.load_weights(strict=False)`
  would). `rfft_guard.py`'s `exact_zero_safe_rfft()` is applied
  unconditionally per org policy but was measured **inert** for this
  package's own STFT call shape (unlike `bs-roformer-infer`, where it was
  ~250,000x load-bearing) — see its module docstring for the full
  measurement and how it was obtained.
- Chunking here has **no overlap and no fade window** — `audio.py`'s
  `chunk_step` always equals `CHUNK_SAMPLES` exactly, so chunks tile the
  waveform back to back. `backends/mlx_backend.py`'s `_chunk_geometry`
  mirrors that arithmetic exactly; do not port the sibling packages'
  overlap-add/fade-window machinery in by habit — it does not apply here.
- STFT/iSTFT depend on `mlx-spectro` (`SpectralTransform`, `center=True`,
  `center_pad_mode="reflect"`, `istft(..., torch_like=True)`), measured
  directly against `torch.stft`/`torch.istft` before being wired in (a
  stft→istft round trip agreed to rel_L2 ~9e-7 on a hop-aligned window).
  MLX quirk worth remembering if this ever needs re-deriving: `mx.split`'s
  list argument is numpy-style cut *points*, not torch.split's per-part
  *sizes*; and a list of *tuples* of `nn.Module`s is invisible to
  `mlx.utils.tree_flatten` (no error — the params just never appear), while
  a list of *lists* is traversed correctly. The latter was caught by
  `load_converted_weights`'s audit during this port, not by inspection.

## Checkpoint facts

`official-vocals` is the author-linked Drive checkpoint; its local audited SHA-256
is `b1fcf93fdd6f7bc79e5410b1330b3d0fd5a6ca6ea23a94730c684629c71cd5ca`.
`msst-vocals` is the MSST v1.0.19 release asset with audited SHA-256
`fceb733b41742abb8df563092f4cb5bbc9243c6c03e7f34dc44c14e60732c72f`.
Both have weight license `NOASSERTION`; code license and checkpoint terms must
stay separate in docs and metadata.

## CUDA extra contract

The core package supports Python 3.10+ without native Mamba dependencies. The
`cuda` extra is deliberately empty: mamba-ssm omits Torch from its PEP 517
build metadata, so putting it in published dependency metadata breaks standard
pip installs and universal locks. Native acceleration remains optional and is
installed separately after compatible Torch with `pip install --no-build-isolation
'mamba-ssm==2.2.2'`. The `cuda` extra therefore remains a successful
compatibility command on Python 3.13 while the pure-Torch runtime stays the
supported default.

## Verification

```bash
uv run pytest -q
python -m build
uv venv /tmp/bs-mamba2-wheel-venv
uv pip install --python /tmp/bs-mamba2-wheel-venv/bin/python dist/*.whl
/tmp/bs-mamba2-wheel-venv/bin/python -c 'import bs_mamba2_infer; print(bs_mamba2_infer.BSMamba2Session)'
rg -n -i 'lightning|hydra|wandb|tensorboard|training_step|validation_step|dataset|evaluate' src pyproject.toml
```

The CUDA golden command uses the checked local author fixture:

```bash
PYTHONPATH=src /path/to/python -c 'from bs_mamba2_infer import BSMamba2Session; from pathlib import Path; r=Path("../.dev-cache/bs-mamba-bandit-probe"); s=BSMamba2Session(checkpoint=r/"weights/official-bsmamba2/vocals/2025-03-14_05-40/weights/sota_model.ckpt", device="cuda").load(); s.infer(r/"listening/00_mixture.wav", output_path="/tmp/bsmamba2-vocals.wav")'
```

Recorded fixture environment: Linux, Python 3.11, torch 2.7.1+cu126, CUDA,
`mamba_ssm` optional accelerator, checkpoint SHA above, 44.1 kHz two-channel
input. The current output differs from the untouched upstream reconstruction by
at most one 16-bit PCM least-significant bit (`6.103515625e-05`), RMS
`1.5819565e-05`; this is a documented tolerance closure, not bit identity.
