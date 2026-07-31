# Changelog

## Unreleased

### Default checkpoint switched to msst-vocals; official-vocals marked unavailable

- **Behavior change, called out deliberately** (allowed by article 7: this
  package has never been published, so its surface owes no compatibility):
  the session/CLI default `checkpoint_id` is now `msst-vocals`. The previous
  default, `official-vocals`, has a dead upstream: its author-linked Google
  Drive URL serves a genuine 0-byte file (verified 2026-07-31 -- the download's
  sha256 is the empty-string hash).
- `official-vocals` is marked `status = "unavailable"` rather than deleted:
  its audited SHA-256 is retained, so a locally obtained copy passed via
  `checkpoint=` still verifies byte-for-byte. `obtain()` refuses the download
  path loudly, naming the cause and the working alternative.
- Re-hosting the weights ourselves is deliberately NOT done: they carry
  `NOASSERTION` (no upstream licence statement), and redistributing unlicensed
  weights is not this package's call. Asking the author for a grant is the
  recorded follow-up.

- **CONTRACT CHANGE**, called out deliberately per article 8: `device="mps"`
  previously raised `ValueError` and `tests/test_contract.py` asserted that
  it must (`test_mps_is_not_claimed`). That negative contract is reversed on
  measured evidence. Real-checkpoint (`msst-vocals`) CPU-vs-MPS parity,
  through the public API, in-memory float comparison (never a written WAV --
  this model's output sits near -80 dBFS, about three 16-bit quantization
  levels, so a file-based comparison measures the container, not the
  backend), 3 s stereo harmonic fixture (220 Hz fundamental + 11 harmonics +
  5 Hz vibrato + light noise -- Gaussian noise alone produces near-silent
  output on both devices and measures nothing), seed 20260730, torch 2.13.0,
  Apple M2:
  ```
  peak amplitude : 8.861387e-05
  max_abs        : 3.6380e-10
  rel_to_peak    : 4.1054e-06
  rel_L2         : 4.7284e-06
  ```
  In line with `bs-roformer-infer`'s own CPU-vs-MPS measurement
  (`rel_to_peak` ~4.5e-06 on its checkpoint). `_resolve_device` now accepts
  `"mps"`, raising `RuntimeError` (never a silent CPU downgrade) when
  explicitly requested but unavailable. **`device="auto"` is deliberately
  unchanged** -- still cuda-else-cpu; a Mac caller's outputs do not move
  under them just because this release added MPS support. Also fixed a
  latent bug this change surfaced: `BSMamba2Session.load()` picked the
  optional native `mamba_ssm` backend (CUDA/Triton-only) for any non-CPU
  Torch device, which would have been wrong on MPS; it is now `"auto"` only
  for `device.type == "cuda"`, forcing the portable graph on both `cpu` and
  `mps`.
- Added an optional MLX backend (`backend="mlx"`, `[mlx]` extra) for native
  Apple Silicon execution: a from-scratch port of the Mamba2 recurrence, band
  split, and mask estimator (there is no upstream MLX BSMamba2 to vendor),
  using `mlx-spectro` for STFT/iSTFT. `backend` is an additive, opt-in axis
  (`torch` default, `mlx`, `auto`); `device` keeps its existing Torch-only
  meaning and is never overloaded to mean "Apple Silicon" -- `backend="mlx"`
  owns its own execution target and accepts only `None`/`"auto"`/`"mps"`,
  raising for anything else rather than reinterpreting it. Torch-vs-MLX
  parity measured on the real `msst-vocals` checkpoint through the public
  API, including a zero-padded-tail and a near-silent-tail fixture:
  TODO_PARITY_NUMBERS.
- Added `backends/` (the compute seam: `SeparationBackend` protocol,
  `TorchBackend` wrapping today's `separate_waveform` unchanged,
  `MLXBackend`) and `mlx/` (the vendored-from-scratch MLX model, weight
  converter with an auditing strict-load gate, and the Metal rfft-artifact
  guard). `import bs_mamba2_infer` stays MLX-free (subprocess-verified).
- Measured (not assumed) that MLX 0.31.2's Metal rfft-zero artifact, load-
  bearing in `bs-roformer-infer`, is **inert** for this package's own STFT
  call shape -- see `mlx/rfft_guard.py`'s module docstring for the numbers.
  Applied the guard anyway, per standing org policy.

## 0.1.0 - 2026-07-22

- Initial inference-only BSMamba2 vocal separation package.
- Added verified official and MSST v1.0.19 checkpoint metadata, explicit device
  selection, cache integrity checks, a reusable session, one-shot API, and CLI.
- Made the `cuda` compatibility extra explicitly empty because mamba-ssm's
  undeclared Torch build dependency breaks standard isolated installation;
  documented its separate upstream non-isolated accelerator installation.
