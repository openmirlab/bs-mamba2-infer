# /nav: lazy barrel for the MLX subpackage -- keeps `import bs_mamba2_infer` MLX-free.
# Reads: .model, .convert, .rfft_guard (all lazily, via __getattr__)
"""Thin barrel for the from-scratch MLX BSMamba2 port -- lazy on purpose.

`bs_mamba2_infer.mlx` is only ever imported on demand by the (caller-owned)
MLX compute backend (`backends/mlx_backend.py`), never by the package's
default import path, so `import bs_mamba2_infer` stays MLX-free even with
this subpackage present. Attribute access is deferred via `__getattr__` so
merely importing this package doesn't eagerly import `mlx.core`/`mlx.nn`.

Reads: .model (BSMamba2MLX, BSMamba2ConfigMLX, lazily), .convert
(convert_torch_to_mlx_weights, load_converted_weights, lazily), .rfft_guard
(exact_zero_safe_rfft, lazily)
"""

from __future__ import annotations

__all__ = [
    "BSMamba2ConfigMLX",
    "BSMamba2MLX",
    "convert_torch_to_mlx_weights",
    "exact_zero_safe_rfft",
    "load_converted_weights",
]


def __getattr__(name: str):
    if name in ("BSMamba2MLX", "BSMamba2ConfigMLX"):
        from . import model

        return getattr(model, name)
    if name in ("convert_torch_to_mlx_weights", "load_converted_weights"):
        from . import convert

        return getattr(convert, name)
    if name == "exact_zero_safe_rfft":
        from .rfft_guard import exact_zero_safe_rfft

        return exact_zero_safe_rfft
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
