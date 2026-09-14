# /nav: public BSMamba2 facade and lifecycle.
# A session is a reusable loaded instrument: load once, infer many times,
# release memory without deleting weights, then close permanently.  The one-shot
# facade is intentionally fresh per call and does not claim cross-call caching.
# Reads: audio.py, checkpoint.py, and model.py; exported by package __init__.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch

from .audio import SAMPLE_RATE, load_audio, require_supported_audio, separate_waveform
from .checkpoint import CheckpointSpec, checkpoint_specs, load_payload, obtain, resolved_path
from .model import BSMamba2, BSMamba2Config


@dataclass(frozen=True)
class SeparationResult:
    """The vocal stem, always channel-first float32 samples at 44.1 kHz."""

    vocals: np.ndarray
    sample_rate: int
    checkpoint_id: str

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, self.vocals.T, self.sample_rate)
        return destination


def _resolve_device(request: str) -> torch.device:
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request == "mps" or request.startswith("mps:"):
        raise ValueError("MPS is not supported: BSMamba2 parity has not been verified there.")
    try:
        device = torch.device(request)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device {request!r}; use auto, cpu, cuda, or cuda:N.") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {request!r}, but CUDA is unavailable.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {request!r}, but only {torch.cuda.device_count()} CUDA devices exist.")
    elif device.type != "cpu":
        raise ValueError(f"Unsupported device {request!r}; use auto, cpu, cuda, or cuda:N.")
    return device


class BSMamba2Session:
    """Reusable BSMamba2 inference session; not safe for concurrent infer calls."""

    def __init__(
        self,
        checkpoint_id: str = "msst-vocals",
        *,
        checkpoint: str | Path | None = None,
        cache_dir: str | Path | None = None,
        device: str = "auto",
        batch_size: int = 4,
    ) -> None:
        specs = checkpoint_specs()
        if checkpoint_id not in specs:
            raise ValueError(f"Unknown BSMamba2 checkpoint {checkpoint_id!r}; choose one of {', '.join(specs)}.")
        self._spec: CheckpointSpec = specs[checkpoint_id]
        self._checkpoint, self._cache_dir = checkpoint, cache_dir
        self._device = _resolve_device(device)
        self._batch_size = batch_size
        self._model: BSMamba2 | None = None
        self._status: Literal["unloaded", "loading", "ready", "released", "closed", "failed"] = "unloaded"
        self._load_count = 0

    @property
    def status(self) -> str:
        return self._status

    @property
    def device(self) -> torch.device:
        return self._device

    def cache_info(self) -> dict[str, object]:
        path = resolved_path(self._spec, cache_dir=self._cache_dir, checkpoint=self._checkpoint)
        return {"checkpoint_id": self._spec.identifier, "path": str(path) if path else None, "cached": path is not None, "cache_dir": str(self._cache_dir) if self._cache_dir else None}

    def load(self) -> "BSMamba2Session":
        if self._status == "closed":
            raise RuntimeError("This BSMamba2Session is closed and cannot be reloaded.")
        if self._status == "ready":
            return self
        if self._status == "failed":
            raise RuntimeError("The prior load failed; call release() before retrying.")
        self._status = "loading"
        try:
            path = obtain(self._spec, cache_dir=self._cache_dir, checkpoint=self._checkpoint)
            raw = load_payload(path)
            state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
            if not isinstance(state, dict):
                raise RuntimeError("Checkpoint does not contain a state dictionary.")
            state = {
                key.removeprefix("model."): value
                for key, value in state.items()
                if key.startswith("model.")
                or not any(key.startswith(prefix) for prefix in ("featurizer.", "inverse_featurizer.", "multi_featurizer."))
            }
            # `auto` selects native mamba_ssm only when that optional package is
            # installed; otherwise it keeps the default install on the pure
            # PyTorch path.  Native mamba_ssm kernels cannot run on CPU, so an
            # explicit CPU session must force the portable backend even when the
            # CUDA extra happens to be installed in the same environment.
            backend: Literal["auto", "torch"] = "torch" if self._device.type == "cpu" else "auto"
            model = BSMamba2(BSMamba2Config(), backend=backend)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Checkpoint graph is not BSMamba2-compatible (missing={missing}, unexpected={unexpected}).")
            self._model = model.eval().to(self._device)
            self._load_count += 1
            self._status = "ready"
        except BaseException:
            self._model = None
            self._status = "failed"
            raise
        return self

    def infer(self, audio: str | Path | np.ndarray | torch.Tensor, *, sample_rate: int | None = None, output_path: str | Path | None = None) -> SeparationResult:
        if self._status != "ready" or self._model is None:
            raise RuntimeError("Call session.load() successfully before infer().")
        waveform, rate = load_audio(audio, sample_rate)
        waveform = require_supported_audio(waveform, rate).to(self._device)
        # The upstream entry point wraps its whole Separator call in CUDA AMP;
        # preserve that default for checkpoint parity.  CPU keeps float32.
        amp = torch.autocast(device_type="cuda") if self._device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
        with amp:
            estimate = separate_waveform(self._model, waveform, batch_size=self._batch_size)[0].detach().cpu().numpy().astype(np.float32, copy=False)
        result = SeparationResult(estimate, SAMPLE_RATE, self._spec.identifier)
        if output_path is not None:
            result.write(output_path)
        return result

    def release(self) -> None:
        if self._status == "closed":
            return
        self._model = None
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        self._status = "released"

    def close(self) -> None:
        if self._status == "closed":
            return
        self.release()
        self._status = "closed"

    def __enter__(self) -> "BSMamba2Session":
        return self.load()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def separate(
    audio: str | Path | np.ndarray | torch.Tensor,
    *,
    sample_rate: int | None = None,
    checkpoint_id: str = "msst-vocals",
    checkpoint: str | Path | None = None,
    cache_dir: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 4,
    output_path: str | Path | None = None,
) -> SeparationResult:
    """Separate vocals in one call; a new session is intentionally created each time."""
    with BSMamba2Session(checkpoint_id, checkpoint=checkpoint, cache_dir=cache_dir, device=device, batch_size=batch_size) as session:
        return session.infer(audio, sample_rate=sample_rate, output_path=output_path)
