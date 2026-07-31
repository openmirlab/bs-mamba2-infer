# /nav: today's Torch inference path, wrapped behind the backend seam verbatim.
# Reads: ..audio (separate_waveform), .base, torch, numpy
"""PyTorch backend -- the shipped path, moved behind the seam unchanged.

`separate()` is `BSMamba2Session.infer()`'s pre-existing body (device-aware
AMP context, `audio.separate_waveform`, numpy conversion) verbatim -- a
wrapper, never a second implementation, so the Torch path cannot drift from
`separate_waveform`/`BSMamba2` that callers still import directly.

Reads: ..audio (separate_waveform), .base (SeparationBackend), torch, numpy
"""

from __future__ import annotations

import numpy as np
import torch

from ..audio import separate_waveform


class TorchBackend:
    """Wraps an already-loaded BSMamba2 and its device as a SeparationBackend."""

    name = "torch"

    def __init__(self, model, device: torch.device, batch_size: int = 4) -> None:
        self._model = model.eval()
        self._device = device
        self._batch_size = batch_size

    @classmethod
    def is_available(cls) -> bool:
        return True

    @property
    def resolved_device(self) -> str:
        return str(self._device)

    @property
    def model(self):
        """The resident model, for callers composing the advanced path directly."""
        return self._model

    def separate(self, mix: np.ndarray) -> np.ndarray:
        waveform = torch.from_numpy(np.ascontiguousarray(mix, dtype=np.float32)).to(self._device)
        # Preserved exactly as BSMamba2Session.infer() has always scoped it:
        # the upstream entry point wraps its whole Separator call in CUDA AMP;
        # CPU (and, once supported, any non-CUDA device) keeps float32.
        amp = torch.autocast(device_type="cuda") if self._device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
        with amp:
            estimate = separate_waveform(self._model, waveform, batch_size=self._batch_size)[0]
        return estimate.detach().cpu().numpy().astype(np.float32, copy=False)

    def release(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            torch.mps.empty_cache()
