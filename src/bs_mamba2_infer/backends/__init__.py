# /nav: backend-name resolution -- torch/mlx/auto, honoured or refused, never swapped.
# Reads: .base, .torch_backend (lazily), .mlx_backend (lazily)
"""Backend registry -- the only name callers use to pick a compute path.

Resolution is by name and nothing more: asking for a backend that cannot run
here raises rather than quietly substituting another -- a silent substitution
is discovered only by noticing the wrong hardware was busy. Backend modules
are imported lazily so `import bs_mamba2_infer` never drags MLX into the
default import path.

Reads: .base (SeparationBackend, BackendUnavailable), .torch_backend (lazily),
.mlx_backend (lazily)
"""

from __future__ import annotations

from .base import BackendUnavailable, SeparationBackend

#: Every selectable backend name, in the order `auto` prefers them.
BACKEND_NAMES = ("mlx", "torch")
DEFAULT_BACKEND = "torch"


def _load(name: str):
    """Import a backend module on demand. Importing must not require its framework."""
    if name == "torch":
        from .torch_backend import TorchBackend

        return TorchBackend
    if name == "mlx":
        from .mlx_backend import MLXBackend

        return MLXBackend
    raise ValueError(f"backend must be None, 'auto', or one of {BACKEND_NAMES}; got {name!r}")


def resolve_backend_name(requested: str | None) -> str:
    """Resolve a requested backend name, honouring it exactly or raising.

    `None` and `"torch"` both mean the shipped Torch path -- the default
    never moves on its own. `"auto"` prefers MLX when it is genuinely
    importable on this machine, falling back to Torch otherwise; that is the
    one place a fallback is what the caller asked for. An *explicit* backend
    still raises when unavailable -- that request is honoured or refused,
    never downgraded.
    """
    if requested is None or requested == DEFAULT_BACKEND:
        return DEFAULT_BACKEND
    if requested == "auto":
        for name in BACKEND_NAMES:
            if _load(name).is_available():
                return name
        return DEFAULT_BACKEND
    if requested not in BACKEND_NAMES:
        raise ValueError(f"backend must be None, 'auto', or one of {BACKEND_NAMES}; got {requested!r}")
    backend = _load(requested)
    if not backend.is_available():
        raise BackendUnavailable(
            f"backend {requested!r} is unavailable on this machine; "
            f"it may need an optional extra (pip install 'bs-mamba2-infer[{requested}]')"
        )
    return requested


def get_backend(name: str):
    """Return the backend class registered under `name`."""
    return _load(name)


__all__ = [
    "BACKEND_NAMES",
    "DEFAULT_BACKEND",
    "BackendUnavailable",
    "SeparationBackend",
    "get_backend",
    "resolve_backend_name",
]
