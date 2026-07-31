"""Torch-vs-MLX output parity on the real checkpoint, including silence.

Marked ``realweights`` and deselected by default: needs the MLX extra, an
Apple Silicon Mac, and the default checkpoint already on disk. It never
downloads. Run explicitly:

    pytest -m realweights tests/test_mlx_parity.py -v

The silence case is the point of this file. Every track's final chunk pads
with zeros (`audio.py`'s `separate_waveform`), and this architecture has no
overlap-add fade window to smooth a boundary the way the sibling roformer
packages do -- a fixture without silence would not exercise the actual
shipped path.

rfft-artifact guard (`mlx/rfft_guard.py`): measured **inert** for this
package, directly, through this same full STFT -> model -> iSTFT chain, on a
synthetic zero-padded-tail fixture, before this file was written (small
random-weight model, not the real checkpoint, to keep the measurement fast --
the effect being measured is a property of the STFT call, not the weights):

    without guard   max_abs 5.82e-09   rel_L2 1.09e-06
    with guard      max_abs 3.96e-09   rel_L2 7.22e-07

Both already at ordinary float32 noise floor -- nothing resembling
bs-roformer-infer's ~250,000x gap. Applied unconditionally anyway per org
policy; see `mlx/rfft_guard.py`'s module docstring for the full measurement,
including why (`mlx_spectro`'s own STFT returns exact 0.0 for an all-zero
frame on this MLX build even without the guard).

Regression-test validation (the actual "remove the fix, confirm it fails"
check for this package -- the rfft guard above has no such story since it is
inert here): this port's real near-miss was a freq-major-vs-channel-major
reshape order bug in the final complex mask multiply (`mlx/model.py`'s
`BSMamba2MLX.__call__`) -- reshaping the ORIGINAL (b, c, f, t) real/imag
directly instead of reusing the freq-major-merged (b, f*c, t) tensor torch
itself multiplies in. Reproducing that exact bug on a small random-weight
model and comparing against the same Torch reference: max_abs went from
1.03e-08 (fixed) to 1.79e-02 (bug reintroduced, rel_L2 1.42) -- a ~1.7
million-fold jump, comfortably caught by this file's `MAX_ABS_TOLERANCE`.
Fixed version restored; the buggy path was never committed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.realweights

MODEL = "msst-vocals"
SEED = 20260731
SAMPLE_RATE = 44_100
SIGNAL_SECONDS = 3
MAX_ABS_TOLERANCE = 1e-5


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        import mlx_spectro  # noqa: F401
    except ImportError:
        return False
    return True


def _checkpoint_path() -> Path | None:
    from bs_mamba2_infer.checkpoint import checkpoint_specs, resolved_path

    spec = checkpoint_specs()[MODEL]
    return resolved_path(spec)


@pytest.fixture(scope="module")
def checkpoint_ready():
    if not _mlx_available():
        pytest.skip("MLX extra not installed: pip install 'bs-mamba2-infer[mlx]'")
    path = _checkpoint_path()
    if path is None:
        pytest.skip(f"{MODEL} is not cached; obtain it via the package's own resolver first")
    return path


def _fixture(tail: str) -> np.ndarray:
    """One short stereo waveform: harmonically structured content, `tail`
    behaviour applied to its second half. Always shorter than CHUNK_SAMPLES
    (352800), so it always pads to exactly one chunk -- the tail always lands
    inside that chunk's zero-padded region regardless of which `tail` case is
    under test.

    Gaussian noise was tried first and measures nothing: a vocal separator
    correctly finds ~no vocals in noise, so Torch and MLX both emit near-zero
    output and "parity" is really "two near-silent arrays agree" (max_abs
    exactly 0.0 was observed). This must be harmonically structured -- a
    fundamental with several harmonics and vibrato, so the model actually has
    something to separate and a real divergence has something to show up in.
    """
    rng = np.random.default_rng(SEED)
    samples = SIGNAL_SECONDS * SAMPLE_RATE
    t = np.arange(samples) / SAMPLE_RATE
    f0 = 220.0 * (1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t))  # 5 Hz vibrato
    voiced = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 12))  # 11 harmonics
    env = 0.5 * (1 - np.cos(2 * np.pi * np.clip(t / SIGNAL_SECONDS, 0, 1)))  # raised-cosine fade
    mono = 0.3 * voiced * env + 0.05 * rng.standard_normal(t.size)
    signal = np.stack([mono, np.roll(mono, 17)], axis=0).astype(np.float32)  # (channels, samples)
    if tail == "signal":
        return signal
    half = samples // 2
    if tail == "zeros":
        signal[:, half:] = 0.0
        return signal
    if tail == "near_silent":
        signal[:, half:] *= 1e-6
        return signal
    raise ValueError(tail)


@pytest.mark.parametrize("tail", ["signal", "zeros", "near_silent"])
def test_mlx_matches_torch_end_to_end(checkpoint_ready, tail):
    """Through the public API on both backends -- not a bare module forward.

    Compares the in-memory float `.vocals` array, never a written WAV: this
    model's output sits near -80 dBFS (about three 16-bit quantization
    levels), so a file-based comparison measures the container, not the
    backend -- `output_path=` is deliberately not used here.
    """
    from bs_mamba2_infer import BSMamba2Session

    mix = _fixture(tail)

    with BSMamba2Session(MODEL, backend="torch", device="cpu") as torch_session:
        reference = torch_session.infer(mix, sample_rate=SAMPLE_RATE).vocals

    with BSMamba2Session(MODEL, backend="mlx") as mlx_session:
        candidate = mlx_session.infer(mix, sample_rate=SAMPLE_RATE).vocals

    assert reference.shape == candidate.shape
    diff = np.abs(reference - candidate)
    worst = float(diff.max())
    peak = float(np.abs(reference).max())
    ref_norm = float(np.linalg.norm(reference))
    rel_to_peak = worst / peak if peak > 0 else float("nan")
    rel_l2 = float(np.linalg.norm(diff)) / ref_norm if ref_norm > 0 else float("nan")
    print(
        f"\n[tail={tail}] peak={peak:.6e} max_abs={worst:.4e} "
        f"rel_to_peak={rel_to_peak:.4e} rel_L2={rel_l2:.4e}"
    )
    assert worst < MAX_ABS_TOLERANCE, (
        f"tail={tail}: Torch-vs-MLX max abs {worst:.3e} (peak {peak:.3e}, rel_to_peak "
        f"{rel_to_peak:.3e}, rel_L2 {rel_l2:.3e}) exceeds {MAX_ABS_TOLERANCE:.0e}. If this "
        f"fired only for a silent tail, suspect exact_zero_safe_rfft in "
        f"bs_mamba2_infer/mlx/rfft_guard.py -- investigate rather than widen the tolerance"
    )
