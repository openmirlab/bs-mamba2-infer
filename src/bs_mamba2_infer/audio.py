# /nav: audio normalization and overlap-add separation path.
# Owns file/array conversion, 44.1 kHz validation, STFT/iSTFT, and the exact
# chunk geometry from the upstream Separator.  No dataset or evaluation code
# crosses this boundary.
# Reads: model.py for the complex mask estimator; api.py for device/model.
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.nn import functional as F

if TYPE_CHECKING:
    from .model import BSMamba2

SAMPLE_RATE = 44_100
N_FFT = WIN_LENGTH = 2_048
HOP_LENGTH = 441
CHUNK_SAMPLES = 8 * SAMPLE_RATE


def load_audio(audio: str | Path | np.ndarray | Tensor, sample_rate: int | None) -> tuple[Tensor, int]:
    """Return a contiguous float32 tensor shaped (channels, samples)."""
    if isinstance(audio, (str, Path)):
        samples, rate = sf.read(str(audio), always_2d=True, dtype="float32")
        return torch.from_numpy(samples.T.copy()), int(rate)
    if sample_rate is None:
        raise ValueError("sample_rate is required when audio is an array or tensor.")
    value = torch.as_tensor(audio, dtype=torch.float32)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError("audio must have shape (samples,), (channels, samples), or be a file path.")
    return value.contiguous(), int(sample_rate)


def require_supported_audio(value: Tensor, sample_rate: int) -> Tensor:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"BSMamba2 checkpoints require {SAMPLE_RATE} Hz audio; received {sample_rate} Hz.")
    if value.shape[0] == 1:
        return value.repeat(2, 1)
    if value.shape[0] != 2:
        raise ValueError(f"expected mono or stereo audio, received {value.shape[0]} channels.")
    return value


def _stft(chunk: Tensor) -> Tensor:
    # Upstream creates the torchaudio Hann buffer on CPU then moves it with the
    # module.  CUDA-side window generation differs by a few ULP, so preserve
    # that construction order for exact-checkpoint parity.
    window = torch.hann_window(WIN_LENGTH, dtype=chunk.dtype).to(chunk.device)
    batch, channels, samples = chunk.shape
    output = torch.stft(chunk.reshape(batch * channels, samples), N_FFT, HOP_LENGTH, WIN_LENGTH, window=window, center=True, return_complex=True)
    return output.reshape(batch, channels, output.shape[-2], output.shape[-1])


def _istft(spectrogram: Tensor, length: int) -> Tensor:
    window = torch.hann_window(WIN_LENGTH, dtype=spectrogram.real.dtype).to(spectrogram.device)
    channels = spectrogram.shape[0]
    output = torch.istft(spectrogram.reshape(channels, spectrogram.shape[-2], spectrogram.shape[-1]), N_FFT, HOP_LENGTH, WIN_LENGTH, window=window, center=True)
    if output.shape[-1] != length:
        raise RuntimeError(f"Unexpected iSTFT length {output.shape[-1]}; expected {length}.")
    return output.reshape(channels, length)


@torch.no_grad()
def separate_waveform(model: "BSMamba2", waveform: Tensor, *, batch_size: int = 4) -> Tensor:
    """Match upstream padding, chunking, STFT masking, iSTFT and overlap-add."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    chunk_step = CHUNK_SAMPLES
    pad_chunk = (HOP_LENGTH - (CHUNK_SAMPLES % HOP_LENGTH)) % HOP_LENGTH
    window_size, hop_size = CHUNK_SAMPLES + pad_chunk, chunk_step + pad_chunk
    padding_whole = CHUNK_SAMPLES - chunk_step
    duration = waveform.shape[-1]
    padding_add = hop_size - (duration + 2 * padding_whole - window_size) % hop_size
    padded = F.pad(waveform, (padding_whole, padding_whole + padding_add))
    padded_duration = padded.shape[-1]
    chunks = padded.unfold(-1, window_size, hop_size).permute(1, 0, 2).contiguous()
    results: list[Tensor] = []
    for start in range(0, chunks.shape[0], batch_size):
        current = chunks[start : start + batch_size]
        estimate = model(_stft(current))
        reconstructed = torch.stack([_istft(estimate[index, 0], window_size) for index in range(estimate.shape[0])])
        results.append(reconstructed)
    stacked = torch.cat(results)
    output = torch.zeros((1, waveform.shape[0], padded_duration), dtype=waveform.dtype, device=waveform.device)
    for index, chunk in enumerate(stacked):
        start = index * hop_size
        output[..., start : start + window_size] += chunk
    return output[..., padding_whole : -(padding_whole + padding_add)]
