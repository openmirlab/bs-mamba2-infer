# /nav: MLX compute path -- builds from an already-loaded Torch model, owns
# chunking (no overlap/fade -- see _chunk_geometry) and STFT/iSTFT (mlx-spectro).
# Reads: .base, ..mlx (lazily), mlx.core (lazily), mlx_spectro (lazily), numpy
"""MLX backend -- native Apple Silicon execution behind the same seam.

Builds the from-scratch MLX `BSMamba2MLX` from an already-loaded, already
state-dict-verified Torch `BSMamba2` -- `BSMamba2Session.load()` keeps sole
ownership of checkpoint resolution, sha256 verification, and the
"model."-prefix / featurizer-key state-dict filtering (`api.py` lines
~104-109); this module never touches a checkpoint path or raw state dict
directly, only an already-constructed `torch.nn.Module`. That is a
deliberate, measured tradeoff shared with the org's other from-scratch MLX
ports (`demucs-infer`): building the (disposable, for backend="mlx") Torch
model first costs some load time, but keeps checkpoint-format knowledge in
exactly one place rather than duplicating the "model."-prefix logic here.

Chunking here is simpler than the sibling packages' overlap-add backends --
see `backends/base.py`'s module docstring: this package's chunk_step always
equals its chunk size exactly, so there is no overlap and no fade window,
just non-overlapping tiles placed back to back. `_chunk_geometry` mirrors
`audio.py`'s `separate_waveform` integer arithmetic exactly (padding_add's
formula in particular): any divergence there is a length bug, not a
numerical one, so it is reproduced verbatim rather than re-derived.

STFT/iSTFT use `mlx_spectro.SpectralTransform` (`center=True,
center_pad_mode="reflect", window_fn="hann", periodic=True`, `istft(...,
torch_like=True)`) -- measured directly against `torch.stft`/`torch.istft`
before this file was written (not assumed from a sibling package): a
stft-then-istft round trip through mlx_spectro agreed with the same round
trip through Torch to rel_L2 ~9e-7 on a hop-aligned window. mlx_spectro
already implements STFT's reflect-mode center padding internally -- MLX's
own `mx.pad` has no reflect mode at all (a real, separately-encountered
limitation), but this file never needs to work around that itself because it
never calls `mx.pad` with reflect semantics; mlx_spectro owns that internally.

`exact_zero_safe_rfft` is applied unconditionally around every STFT call, per
org policy, regardless of what was measured for this package -- see
`mlx/rfft_guard.py` and `tests/test_mlx_parity.py`'s module docstring for the
measurement and its result.

Reads: .base (BackendUnavailable), ..mlx (BSMamba2MLX, BSMamba2ConfigMLX,
convert_torch_to_mlx_weights, load_converted_weights, exact_zero_safe_rfft),
numpy
"""

from __future__ import annotations

import numpy as np

from .base import BackendUnavailable

SAMPLE_RATE = 44_100
N_FFT = WIN_LENGTH = 2_048
HOP_LENGTH = 441
CHUNK_SAMPLES = 8 * SAMPLE_RATE


def _chunk_geometry(duration: int) -> tuple[int, int]:
    """Mirrors `audio.py`'s `separate_waveform` padding arithmetic exactly.

    Returns `(window_size, padding_add)`. For this package's actual
    constants, `CHUNK_SAMPLES` is already an exact multiple of `HOP_LENGTH`
    (352800 = 800 * 441), so `pad_chunk` is always 0 and `window_size ==
    hop_size == CHUNK_SAMPLES` -- i.e. plain non-overlapping tiling, not the
    overlap-add fade-window geometry the sibling MLX backends use. Computed
    generally anyway (not hardcoded to 0) so a future constant change is not
    a silent divergence from `audio.py`.
    """
    pad_chunk = (HOP_LENGTH - (CHUNK_SAMPLES % HOP_LENGTH)) % HOP_LENGTH
    window_size = CHUNK_SAMPLES + pad_chunk
    hop_size = CHUNK_SAMPLES + pad_chunk
    padding_whole = CHUNK_SAMPLES - CHUNK_SAMPLES  # always 0; kept symbolic, see audio.py
    padding_add = hop_size - (duration + 2 * padding_whole - window_size) % hop_size
    return window_size, padding_add


class MLXBackend:
    """Runs BSMamba2 natively on Apple Silicon through MLX."""

    name = "mlx"

    def __init__(self, model, device: str = "mps") -> None:
        self._model = model
        self._device = device
        self._transform = None  # built lazily on first separate() call

    # ------------------------------------------------------------- availability

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mlx.core  # noqa: F401
            import mlx_spectro  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def _require(cls) -> None:
        if not cls.is_available():
            raise BackendUnavailable(
                "the MLX backend needs the optional extra: pip install 'bs-mamba2-infer[mlx]' (Apple Silicon)"
            )

    # ------------------------------------------------------------- construction

    @classmethod
    def from_torch_model(cls, torch_model, device: str | None = "mps") -> MLXBackend:
        """Convert an already-loaded, already-verified Torch `BSMamba2` in place.

        Never touches a checkpoint path or raw state dict -- see this
        module's docstring for why.
        """
        import os

        cls._require()
        device = cls._select_device(device)
        os.environ.setdefault("MLX_ENABLE_AMP", "0")  # accuracy-affecting optimizations are opt-in, default off

        from ..mlx import (
            BSMamba2ConfigMLX,
            BSMamba2MLX,
            convert_torch_to_mlx_weights,
            load_converted_weights,
        )

        config = BSMamba2ConfigMLX(
            dim=torch_model.dim,
            depth=len(torch_model.layers),
            stereo=torch_model.stereo,
            num_stems=torch_model.num_stems,
            # Not introspectable from the resident torch model's public
            # attributes; every checkpoint this package ships uses the
            # BSMamba2Config() default (api.py never overrides it). If that
            # ever changes, load_converted_weights below will raise loudly
            # on a shape/key mismatch rather than silently building the
            # wrong-depth MLP.
            mask_estimator_depth=2,
        )
        mlx_model = BSMamba2MLX(config)
        weights = convert_torch_to_mlx_weights(torch_model.state_dict())
        load_converted_weights(mlx_model, weights)
        mlx_model.eval()
        return cls(mlx_model, device=device)

    @staticmethod
    def _select_device(device: str | None) -> str:
        """MLX owns its own execution target; a Torch device string is refused."""
        if device in (None, "auto", "mps"):
            return "mps"
        raise BackendUnavailable(
            f"backend 'mlx' cannot honour device {device!r}; it executes on Apple "
            f"Silicon and accepts None, 'auto', or 'mps'. Use backend='torch' to select a Torch device."
        )

    # ----------------------------------------------------------------- protocol

    @property
    def resolved_device(self) -> str:
        return self._device

    @property
    def model(self):
        """The resident MLX model, for callers composing the advanced path directly."""
        return self._model

    def release(self) -> None:
        self._model = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass

    def separate(self, mix: np.ndarray) -> np.ndarray:
        """Chunked STFT -> model -> iSTFT, mirroring `separate_waveform`'s
        exact non-overlapping tiling (see `_chunk_geometry`)."""
        import mlx.core as mx
        import mlx_spectro as ms

        from ..mlx import exact_zero_safe_rfft

        if self._transform is None:
            self._transform = ms.SpectralTransform(
                N_FFT, HOP_LENGTH, WIN_LENGTH,
                window_fn="hann", periodic=True,
                center=True, center_pad_mode="reflect",
            )
        transform = self._transform

        mix = np.ascontiguousarray(mix, dtype=np.float32)
        duration = mix.shape[-1]
        window_size, padding_add = _chunk_geometry(duration)
        padded = np.pad(mix, ((0, 0), (0, padding_add)))
        num_chunks = padded.shape[-1] // window_size

        reconstructed_chunks = []
        for index in range(num_chunks):
            part = padded[:, index * window_size:(index + 1) * window_size]
            chunk = mx.array(part)
            with exact_zero_safe_rfft():
                spectrum = transform.stft(chunk, output_layout="bfn")  # (channels, freq, time) complex
                mx.eval(spectrum)
            real, imag = spectrum.real[None], spectrum.imag[None]  # add batch axis
            out_real, out_imag = self._model(real, imag)
            mx.eval(out_real, out_imag)
            out_complex = out_real[0] + 1j * out_imag[0]  # (channels, freq, time)
            recon = transform.istft(out_complex, length=window_size, torch_like=True, input_layout="bfn")
            mx.eval(recon)
            reconstructed_chunks.append(np.array(recon, dtype=np.float32))

        # No overlap between chunks (see _chunk_geometry / base.py docstring),
        # so placing them back to back is exact -- no fade window, no
        # accumulator, and no exposure to the measured mx.array.at[...].add()
        # large-scatter corruption other org packages worked around.
        stitched = np.concatenate(reconstructed_chunks, axis=-1)
        return stitched[:, :duration]
