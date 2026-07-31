# /nav: SeparationBackend protocol -- one whole waveform in, vocals out.
# Seam sits above chunking (accumulates on-device); no overlap/fade here.
# Reads: numpy (boundary array type only)
"""The backend seam -- one narrow protocol every compute backend implements.

A backend owns everything framework-specific: model construction, checkpoint
weights, tensor layout, device placement, and the whole chunked STFT/model/
iSTFT pipeline `audio.py`'s `separate_waveform` already does for Torch. The
seam sits at a *whole waveform*, not a single chunk or a single forward pass:
`separate_waveform` accumulates its output tensor on-device across chunks, so
a lower seam (per-chunk forward, arrays crossing every chunk) would drag that
accumulator back to the host on every step and hand the performance win
straight back.

Above the seam nothing knows a tensor layout, a dtype, or which chip is busy:
checkpoint resolution/verification (`checkpoint.py`) and session lifecycle
(`api.py`) are decided once and stay unchanged regardless of which backend
runs.

Chunking geometry note, specific to this package (does not transfer from the
sibling packages' overlap-add backends): `audio.py`'s chunk_step always equals
CHUNK_SAMPLES exactly, so there is no overlap and no fade window -- chunks
tile the waveform back to back. A backend here is simpler than the
roformer/melband/demucs shift-overlap-fade machinery; do not port that
machinery in by habit.

Reads: numpy (boundary array type only)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SeparationBackend(Protocol):
    """Turns one loaded waveform into the vocals stem, hiding how and where
    it computed it. This package only ever produces one stem ("vocals",
    hardcoded to mask-estimator index 0 by `audio.py` itself regardless of
    `BSMamba2Config.num_stems`), so unlike sibling packages the seam returns
    a single array, not a dict of stems."""

    #: Stable identifier, matching the `backend=` argument that selects it.
    name: str

    @classmethod
    def is_available(cls) -> bool:
        """True when this backend can actually run on this machine right now."""

    @property
    def resolved_device(self) -> str:
        """The concrete target chosen, after any `auto`/`None` sentinel was resolved."""

    def separate(self, mix: np.ndarray) -> np.ndarray:
        """Separate one `(channels, samples)` float32 mixture into vocals,
        same shape as `mix`."""

    def release(self) -> None:
        """Drop resident model and device memory. Disk checkpoints stay."""


class BackendUnavailable(RuntimeError):
    """Raised when a backend is requested by name but cannot run here.

    Always raised, never swallowed into a fallback: silently substituting a
    different backend discards what the caller explicitly asked for, and
    would only be discovered by noticing the wrong hardware was busy.
    """
