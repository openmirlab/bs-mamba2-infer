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
- `auto`, `cpu`, `cuda`, and `cuda:N` device selection. MPS is deliberately
  unsupported (`device="mps"` raises) -- Apple Silicon/MLX/MPS support is
  out of scope for this package.

## Scope

Included: BSMamba2's model graph, STFT/iSTFT, chunked preprocessing,
checkpoint/cache handling, device lifecycle, API, and CLI.

Out of scope forever: Mamba v1, BS-RoFormer, TS-BSmamba2, model training,
evaluation metrics, datasets, Lightning, Hydra, W&B, TensorBoard, and bundled
weights.

## Install

`bs-mamba2-infer` is not published on PyPI. Install from the repository:

```bash
git clone https://github.com/openmirlab/bs-mamba2-infer.git
cd bs-mamba2-infer
python -m pip install .
```

The default installation has no compiled Mamba dependency. The `[cuda]`
extra is deliberately empty: `mamba-ssm` omits Torch from its PEP 517 build
metadata, so declaring it as a normal dependency makes pip and universal locks
attempt a broken isolated build.

For the optional upstream CUDA accelerator, use Linux with a compatible
PyTorch/CUDA toolchain, install Torch first, then follow Mamba's required
non-isolated build step:

```bash
pip install torch  # choose the wheel matching your CUDA runtime
pip install --no-build-isolation 'mamba-ssm==2.2.2'
python -m pip install .  # from the cloned repository above
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
