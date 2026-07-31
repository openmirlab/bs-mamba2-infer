# bs-mamba2-infer

Inference-only vocal separation with the BSMamba2 architecture.

## Why this exists

[BSMamba2](https://github.com/EuiYeonKim/BSMamba2) is Euiyeon Kim's official
PyTorch implementation of a BSMamba2 vocal separator.  Its upstream entry
point is tied to training-era Hydra/Lightning configuration and CUDA-only
`mamba_ssm` installation.  This package reprovides only the runnable inference
path as a standalone library and CLI, with a pure-PyTorch fallback and optional
CUDA acceleration.

## Acknowledgments

- [Euiyeon Kim](https://github.com/EuiYeonKim) and Yong-Hoon Choi authored
  BSMamba2 and the original [source repository](https://github.com/EuiYeonKim/BSMamba2).
- The author-provided checkpoint is linked from the upstream
  [pretrained-weights table](https://github.com/EuiYeonKim/BSMamba2#pretrained-weights).
- [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
  provides the separately listed compatible MSST v1.0.19 checkpoint.

## Citation

```bibtex
@article{kim2025mamba2,
  title={Mamba2 Meets Silence: Robust Vocal Source Separation for Sparse Regions},
  author={Kim, Euiyeon and Choi, Yong-Hoon},
  journal={arXiv preprint arXiv:2508.14556},
  year={2025}
}
```

This citation is from the [primary arXiv record](https://arxiv.org/abs/2508.14556).

## Features

- One task: stereo vocal extraction at 44.1 kHz.
- `separate(...)` one-shot API and load-once `BSMamba2Session` API.
- Checksum-verified cache/download registry and manual checkpoint paths.
- `auto`, `cpu`, `cuda`, `cuda:N`, and `mps` device selection. `auto` stays
  cuda-else-cpu deliberately (MPS is opt-in, never auto-promoted — see
  CHANGELOG for the measured CPU-vs-MPS parity that lifted the earlier
  MPS refusal).

## Scope

Included: BSMamba2's model graph, STFT/iSTFT, chunked preprocessing,
checkpoint/cache handling, device lifecycle, API, and CLI.

Out of scope forever: Mamba v1, BS-RoFormer, TS-BSmamba2, model training,
evaluation metrics, datasets, Lightning, Hydra, W&B, TensorBoard, and bundled
weights.

## Install

```bash
pip install bs-mamba2-infer
```

The default installation has no compiled Mamba dependency. `pip install
'bs-mamba2-infer[cuda]'` remains a successful compatibility command, but the
extra is deliberately empty: `mamba-ssm` omits Torch from its PEP 517 build
metadata, so declaring it as a normal dependency makes pip and universal locks
attempt a broken isolated build.

For the optional upstream CUDA accelerator, use Linux with a compatible
PyTorch/CUDA toolchain, install Torch first, then follow Mamba's required
non-isolated build step:

```bash
pip install torch  # choose the wheel matching your CUDA runtime
pip install --no-build-isolation 'mamba-ssm==2.2.2'
pip install bs-mamba2-infer
```

Python 3.13 remains a successful pure-PyTorch install, including with the
`[cuda]` compatibility extra. Use an environment supported by `mamba-ssm` for
the optional accelerated path; this package never makes it a core dependency.

## Quick start

```python
from bs_mamba2_infer import BSMamba2Session

with BSMamba2Session(device="auto") as session:
    result = session.infer("mixture.wav", output_path="vocals.wav")
print(result.vocals.shape, result.sample_rate)
```

`separate("mixture.wav")` is the equivalent one-shot convenience function. It
creates and discards a session per call; use `BSMamba2Session` when repeated
calls should retain the loaded model. Sessions are not safe for concurrent
`infer()` calls.

Arrays are channel-first and need `sample_rate=44100`; file paths supply their
own sample rate. Mono is duplicated to stereo. This release rejects resampling
instead of silently changing audio.

```bash
bs-mamba2-infer mixture.wav vocals.wav --device cuda:0
```

## Backends and devices

Two independent choices:

| Argument | Values | Meaning |
|---|---|---|
| `backend` | `torch` (default), `mlx`, `auto` | which framework computes |
| `device` | `auto`, `cpu`, `cuda`, `cuda:N`, `mps` | where Torch computes (backend="torch" only) |

`backend` defaults to `torch`, so nothing changes unless you ask. `auto` picks
`mlx` only when it is genuinely importable on this machine, falling back to
`torch` otherwise. Requesting a backend that cannot run here raises
immediately — before any checkpoint is downloaded — rather than quietly
using a different one. `backend="mlx"` owns its own Apple Silicon execution
and accepts only `device` of `auto`/`mps` (or none), refusing anything else
rather than ignoring it.

### The MLX backend

Native Apple Silicon execution through [MLX](https://github.com/ml-explore/mlx)
and [mlx-spectro](https://github.com/ssmall256/mlx-spectro) for STFT/iSTFT —
a from-scratch port of this package's own Mamba2 graph (there is no upstream
MLX BSMamba2 to vendor). Install it with the extra, never part of the core
install:

```bash
pip install "bs-mamba2-infer[mlx]"
```

```python
BSMamba2Session(backend="mlx").load()
```

```bash
bs-mamba2-infer mixture.wav vocals.wav --backend mlx
```

`backend="mlx"` accepts only `device` of `None`, `"auto"`, or `"mps"` and
raises for anything else (`"cuda"`, `"cpu"`, ...) rather than reinterpreting
it — MLX owns its own execution target, Torch device strings mean nothing to
it.

MPS and MLX both need an **arm64 Python interpreter**. Under Rosetta/x86_64
they report as unavailable rather than failing loudly — an x86_64
interpreter makes `torch.backends.mps.is_available()` return `False`, and
MLX publishes no macOS x86_64 wheel at all, so the `[mlx]` extra cannot even
install there. An accelerated path just looks absent rather than
misconfigured. This is easy to hit without noticing: an x86_64 `uv`
resolves x86_64 interpreters, so `uv sync` can silently produce an
environment where the accelerated paths structurally cannot exist. Check
with `python -c "import platform; print(platform.machine())"` — it must
print `arm64`.

Torch-vs-MLX parity, measured on the real `msst-vocals` checkpoint through
this public API. The fixture is synthetic — a 3 s stereo harmonic tone at
44.1 kHz, **not real music** — in three tail conditions. Each pads to exactly
one 8 s chunk, so the tail always lands inside the zero-padded region that
every real track's final chunk also has:

| tail | reference peak | max abs diff | rel to peak | rel L2 |
|---|---|---|---|---|
| signal | 4.477348e-05 | 2.0191e-10 | 4.5095e-06 | 4.2311e-06 |
| zero-padded | 3.045707e-04 | 6.9122e-10 | 2.2695e-06 | 2.3398e-06 |
| near-silent | 3.921362e-05 | 1.6007e-10 | 4.0820e-06 | 4.6030e-06 |

The reference peaks are small because this model's output on that fixture
sits near -80 dBFS; that is exactly why the relative columns are reported and
a bare max-abs figure would mean nothing here. Reproduce with `pytest -m
realweights tests/test_mlx_parity.py -v -s` — 2 h 27 min for the three cases,
since the Mamba2 recurrence is a per-timestep Python loop. Keep the `-s`:
without it pytest discards a passing test's printed measurements and the run
yields a verdict but no numbers.

Measured separately and reported here because it is a different claim: the
Metal rfft-zero guard is **inert** for this package — `max_abs` 3.96e-09 with
the guard vs. 5.82e-09 without on a zero-padded fixture, both already at
ordinary float32 noise floor (see `tests/test_mlx_parity.py`'s module
docstring). It is applied anyway.

Speed: not yet benchmarked. The Mamba2 recurrence is a per-timestep Python
loop (no fused CUDA/Triton kernel exists for MLX), so this is inherently
loop-bound on both frameworks; see `mlx/model.py`'s module docstring for the
one deliberate, verified-safe optimization taken (vectorizing the recurrence
across attention heads).

## Checkpoints and cache

The default `msst-vocals` checkpoint auto-downloads to
`~/.cache/bs-mamba2-infer/msst-vocals.ckpt` and is verified against the
SHA-256 in `config/checkpoints.toml`. It comes from MSST
[v1.0.19](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.19).
Set `BS_MAMBA2_INFER_CACHE=/path/to/cache` or pass `cache_dir=` to move it.

The author-trained `official-vocals` entry is currently **unavailable**: its
Google Drive link now serves an empty file (verified 2026-07-31), and this
project does not re-host weights whose licence is unstated. The entry is kept
in the registry with its audited SHA-256, so if you obtain
`sota_model.ckpt` yourself (for example from a copy the author republishes),
passing `checkpoint=/path/to/sota_model.ckpt` still verifies it byte-for-byte.

Both checkpoint licenses are `NOASSERTION`: the code is MIT, but no separate
weight license was verified. Review the source terms before downstream use.

## What this project will NEVER bundle

No author or community checkpoint is committed to this repository, wheel, or
sdist. They remain user-cache downloads or explicitly supplied local files.

## Development

```bash
uv run pytest -q
python -m build
```

The real golden gate needs the approved local fixtures and a CUDA environment
with the optional `mamba_ssm` accelerator. See `CLAUDE.md` for its exact command
and recorded environment/tolerance evidence.

## License

MIT; see [LICENSE](LICENSE). The implementation credits and preserves the
upstream MIT notice. Checkpoint terms are separate and currently `NOASSERTION`.

## Support

Report reproducible issues with the checkpoint ID, SHA-256, Torch version,
device, and exact input sample rate.
