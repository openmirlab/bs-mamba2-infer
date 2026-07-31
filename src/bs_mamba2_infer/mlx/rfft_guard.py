# /nav: Metal rfft-zero-artifact workaround -- measured INERT for this package.
# Applied unconditionally anyway per org policy; see docstring for the numbers.
# Reads: mlx.core
"""`exact_zero_safe_rfft` -- routes `mx.fft.rfft` through the CPU stream.

MLX 0.31.2's Metal rfft kernel packs two real FFTs into one complex FFT; in
float32 that cancellation is not bit-exact, so a frame whose true value is
exactly zero comes back as roughly 4.5e-07 instead of 0.

Whether this matters for a given model depends on what happens to that
corrupted frame downstream. In `bs-roformer-infer` it was severely
load-bearing (~250,000x worse without it) because that model's `L2Norm` uses
eps=1e-12 (five orders below the artifact, so its clamp never engages) and
then spreads the corrupted frame across every time position via attention.
`mdxnet-infer` measured the same guard **inert** for its architecture (no
eps that small, no attention).

Measured **inert** for this package, directly, on its own model, with a
genuine zero-padded-tail fixture through the full STFT -> model -> iSTFT
chain (not assumed by analogy to either sibling): max abs Torch-vs-MLX
divergence was 5.82e-09 without the guard and 3.96e-09 with it -- both
already at ordinary float32 noise floor, nothing resembling bs-roformer's
~250,000x gap. Isolated further: `mlx_spectro`'s own `SpectralTransform.stft`
returns an exact `0.0` for a purely all-zero frame on this MLX build (0.31.2)
even without the guard, so the guard has nothing to fix at the point this
package actually calls into `mx.fft.rfft` -- unlike `bs-roformer-infer`'s
vendored model, which called `mx.fft.rfft` directly on a real Metal stream
where the packed-two-reals-into-one-complex-FFT artifact does reproduce. This
does NOT establish that `mlx_spectro` is unconditionally safe in general (a
different call shape or MLX version could differ); it establishes that it is
safe for the specific call this package makes. `RMSNormMLX`'s eps (1e-12,
matching torch's `F.normalize` default -- the same order that let the
roformer bug through) and the Mamba2 recurrence's cross-sequence state
propagation were both plausible reasons this package might have matched
bs-roformer's result rather than mdxnet's; they didn't. This is why it was
measured rather than inferred from either sibling's architecture by analogy.
See `tests/test_mlx_parity.py`'s module docstring for the fixture and number.

Applied unconditionally regardless of the measurement, per the org's standing
policy of cheap insurance once a real bug in this class has already been
found elsewhere.

Caveat, stated rather than hidden: this swaps a module-level attribute, so it
is not thread-safe. Inference here is single-threaded per session.

Delete this once MLX's rfft kernel is fixed upstream.

Reads: mlx.core
"""

from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx


@contextmanager
def exact_zero_safe_rfft():
    """Context manager: `mx.fft.rfft` runs on the CPU stream while active."""
    original = mx.fft.rfft

    def cpu_stream_rfft(*args, **kwargs):
        with mx.stream(mx.cpu):
            result = original(*args, **kwargs)
            mx.eval(result)
        return result

    mx.fft.rfft = cpu_stream_rfft
    try:
        yield
    finally:
        mx.fft.rfft = original
