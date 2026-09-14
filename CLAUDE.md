# bs-mamba2-infer maintainer guide

**`docs/`** is local-only by policy (2026-09-14): kept on disk, gitignored,
never pushed to GitHub.

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
- **Torch-vs-MLX parity is measured, on the real checkpoint, end to end.**
  `tests/test_mlx_parity.py` (marked `realweights`) run against real
  `msst-vocals`: max abs difference 2.0191e-10 / 6.9122e-10 / 1.6007e-10 for
  the signal, zero-padded-tail and near-silent-tail cases, i.e. 4.5095e-06 /
  2.2695e-06 / 4.0820e-06 relative to a reference peak that sits near
  -80 dBFS. Quote the relative figures, never the bare max-abs — against an
  output that quiet, 2e-10 sounds impressive and says nothing.
  Two traps if you re-run it, both paid for the hard way:
  **(a)** the test reports through `print()`, and pytest discards a *passing*
  test's stdout, so a run without `-s` measures correctly and throws the
  reading away — one 87-minute run returned a verdict and no numbers. Use
  `-v -s`, set `PYTHONUNBUFFERED=1` so each case lands on disk as it
  finishes, and redirect to a file rather than piping through `tail`.
  **(b)** it resolves the checkpoint once in a module-scoped fixture but
  re-resolves on every session construction, so a cache cleared mid-run fails
  a *later* case an hour in, not the first one. Do not clean `~/.cache` while
  it is running. Budget 2 h 27 min for the three cases.
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

`official-vocals` is the author-linked Drive checkpoint, now marked
`status = "unavailable"`: the Drive URL serves a genuine 0-byte file (verified
2026-07-31). Its audited SHA-256
`b1fcf93fdd6f7bc79e5410b1330b3d0fd5a6ca6ea23a94730c684629c71cd5ca` is retained
so a locally supplied copy still verifies via `checkpoint=`. Re-hosting is
blocked on a licence decision (NOASSERTION weights; constitution articles 4/8).
The session/CLI default is now `msst-vocals`.
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

# MLX backend, on an Apple Silicon Mac with the [mlx] extra installed and the
# default checkpoint already cached (this test never downloads):
uv sync --extra dev --extra mlx
uv run pytest -m realweights tests/test_mlx_parity.py -v
```

`MLX_ENABLE_AMP=0` does not need to be set manually for the command above --
`backends/mlx_backend.py` sets it itself (`os.environ.setdefault`) before
constructing any MLX model. The realweights test needs an arm64 interpreter:
it skips silently under x86_64 (including Rosetta), so a green run on the
wrong arch exercises no MLX code and proves nothing about that path. Confirm
`python -c "import platform; print(platform.machine())"` says `arm64` first.

The CUDA golden command uses the checked local author fixture:

```bash
PYTHONPATH=src /path/to/python -c 'from bs_mamba2_infer import BSMamba2Session; from pathlib import Path; r=Path("../.dev-cache/bs-mamba-bandit-probe"); s=BSMamba2Session(checkpoint=r/"weights/official-bsmamba2/vocals/2025-03-14_05-40/weights/sota_model.ckpt", device="cuda").load(); s.infer(r/"listening/00_mixture.wav", output_path="/tmp/bsmamba2-vocals.wav")'
```

Recorded fixture environment: Linux, Python 3.11, torch 2.7.1+cu126, CUDA,
`mamba_ssm` optional accelerator, checkpoint SHA above, 44.1 kHz two-channel
input. The current output differs from the untouched upstream reconstruction by
at most one 16-bit PCM least-significant bit (`6.103515625e-05`), RMS
`1.5819565e-05`; this is a documented tolerance closure, not bit identity.
