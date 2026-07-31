# /nav: public BSMamba2 facade and lifecycle.
# A session is a reusable loaded instrument: load once, infer many times,
# release memory without deleting weights, then close permanently.  The one-shot
# facade is intentionally fresh per call and does not claim cross-call caching.
# `backend=` selects the compute seam (torch/mlx/auto, see backends/base.py);
# `device` keeps its existing Torch-only meaning and is only ever resolved by
# _resolve_device() for backend="torch" -- the MLX backend owns its own device
# resolution independently (backends/mlx_backend.py's _select_device), so it
# is never routed through _resolve_device and never affected by that
# function's own (separately owned) MPS contract.
# Reads: audio.py, checkpoint.py, model.py, backends/; exported by package __init__.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch

from .audio import SAMPLE_RATE, load_audio, require_supported_audio
from .backends import get_backend, resolve_backend_name
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


def mps_available() -> bool:
    """True when this torch build exposes a usable Apple Silicon MPS backend."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _resolve_device(request: str) -> torch.device:
    if request == "auto":
        # Legacy auto-selection, deliberately unchanged (D8): MPS is opt-in,
        # never promoted silently, so a Mac caller's outputs do not move
        # under them just because this release added MPS support.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request == "mps":
        # An explicit accelerator request is honoured or the call fails --
        # never silently downgraded to CPU, matching CUDA's own contract below.
        if not mps_available():
            raise RuntimeError(f"Requested {request!r}, but this build has no usable Apple Silicon MPS backend.")
        return torch.device("mps")
    if request.startswith("mps:"):
        raise ValueError(f"Unsupported device {request!r}; MPS has no indexed devices, use 'mps'.")
    try:
        device = torch.device(request)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device {request!r}; use auto, cpu, cuda, cuda:N, or mps.") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {request!r}, but CUDA is unavailable.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {request!r}, but only {torch.cuda.device_count()} CUDA devices exist.")
    elif device.type != "cpu":
        raise ValueError(f"Unsupported device {request!r}; use auto, cpu, cuda, cuda:N, or mps.")
    return device


class BSMamba2Session:
    """Reusable BSMamba2 inference session; not safe for concurrent infer calls."""

    def __init__(
        self,
        checkpoint_id: str = "official-vocals",
        *,
        checkpoint: str | Path | None = None,
        cache_dir: str | Path | None = None,
        device: str = "auto",
        backend: str | None = None,
        batch_size: int = 4,
    ) -> None:
        specs = checkpoint_specs()
        if checkpoint_id not in specs:
            raise ValueError(f"Unknown BSMamba2 checkpoint {checkpoint_id!r}; choose one of {', '.join(specs)}.")
        self._spec: CheckpointSpec = specs[checkpoint_id]
        self._checkpoint, self._cache_dir = checkpoint, cache_dir
        self._backend_name = resolve_backend_name(backend)
        if self._backend_name == "torch":
            self._torch_device: torch.device | None = _resolve_device(device)
            self._resolved_device_str = str(self._torch_device)
        else:
            # MLX owns its own execution target; a Torch device string is
            # refused rather than reinterpreted (see backends/mlx_backend.py).
            # This never touches _resolve_device -- that function's MPS
            # contract is Torch's own and is not overloaded here.
            from .backends.mlx_backend import MLXBackend

            self._torch_device = None
            self._resolved_device_str = MLXBackend._select_device(device)
        self._batch_size = batch_size
        self._backend = None
        self._status: Literal["unloaded", "loading", "ready", "released", "closed", "failed"] = "unloaded"
        self._load_count = 0

    @property
    def status(self) -> str:
        return self._status

    @property
    def backend(self) -> str:
        """The resolved backend name ('torch' or 'mlx')."""
        return self._backend_name

    @property
    def device(self) -> torch.device | str:
        """A `torch.device` for backend='torch' sessions; the resolved device
        string (e.g. 'mps') for backend='mlx' sessions -- MLX does not
        construct a `torch.device` for its own execution target."""
        return self._torch_device if self._backend_name == "torch" else self._resolved_device_str

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
            if self._backend_name == "torch":
                # `auto` selects native mamba_ssm only when that optional package is
                # installed; otherwise it keeps the default install on the pure
                # PyTorch path.  Native mamba_ssm kernels cannot run on CPU, so an
                # explicit CPU session must force the portable backend even when the
                # CUDA extra happens to be installed in the same environment.
                native_backend: Literal["auto", "torch"] = "torch" if self._torch_device.type == "cpu" else "auto"
            else:
                # The Torch model built here for backend="mlx" is disposable
                # scaffolding: only its state_dict is read (by MLXBackend.
                # from_torch_model, see backends/mlx_backend.py), never its
                # forward(). Always the portable graph, both because it is
                # never executed and to avoid depending on mamba_ssm/CUDA
                # machinery on a machine that may have neither.
                native_backend = "torch"
            model = BSMamba2(BSMamba2Config(), backend=native_backend)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Checkpoint graph is not BSMamba2-compatible (missing={missing}, unexpected={unexpected}).")
            model = model.eval()
            backend_class = get_backend(self._backend_name)
            if self._backend_name == "torch":
                self._backend = backend_class(model.to(self._torch_device), self._torch_device, batch_size=self._batch_size)
            else:
                self._backend = backend_class.from_torch_model(model, device=self._resolved_device_str)
            self._load_count += 1
            self._status = "ready"
        except BaseException:
            self._backend = None
            self._status = "failed"
            raise
        return self

    def infer(self, audio: str | Path | np.ndarray | torch.Tensor, *, sample_rate: int | None = None, output_path: str | Path | None = None) -> SeparationResult:
        if self._status != "ready" or self._backend is None:
            raise RuntimeError("Call session.load() successfully before infer().")
        waveform, rate = load_audio(audio, sample_rate)
        waveform = require_supported_audio(waveform, rate)
        # numpy is the seam's boundary currency (backends/base.py) -- nothing
        # above this line knows a tensor layout, a dtype, or which chip is busy.
        mix = waveform.numpy().astype(np.float32, copy=False)
        estimate = self._backend.separate(mix)
        result = SeparationResult(estimate, SAMPLE_RATE, self._spec.identifier)
        if output_path is not None:
            result.write(output_path)
        return result

    def release(self) -> None:
        if self._status == "closed":
            return
        if self._backend is not None:
            self._backend.release()
        self._backend = None
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
    checkpoint_id: str = "official-vocals",
    checkpoint: str | Path | None = None,
    cache_dir: str | Path | None = None,
    device: str = "auto",
    backend: str | None = None,
    batch_size: int = 4,
    output_path: str | Path | None = None,
) -> SeparationResult:
    """Separate vocals in one call; a new session is intentionally created each time."""
    with BSMamba2Session(checkpoint_id, checkpoint=checkpoint, cache_dir=cache_dir, device=device, backend=backend, batch_size=batch_size) as session:
        return session.infer(audio, sample_rate=sample_rate, output_path=output_path)
