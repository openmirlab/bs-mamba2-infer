# /nav: torch state_dict -> BSMamba2MLX weights, plus an auditing load gate.
# load_converted_weights() raises on any unmatched/dropped tensor instead of
# silently partial-loading. One architecture, no variant-head ambiguity.
# Reads: mlx.core, mlx.utils (tree_flatten), numpy
"""PyTorch -> MLX weight conversion for BSMamba2MLX, plus a strict load gate.

This is a from-scratch port (no upstream MLX BSMamba2 exists), so the mapping
here is derived directly against this package's own `model.py`, not adapted
from a vendored conversion script. Every torch key has exactly one MLX
destination -- there is no variant-head ambiguity the way roformer siblings
have, since BSMamba2 has one architecture.

`load_converted_weights()` diffs the model's own parameter keys (via
`mlx.utils.tree_flatten`) against the converted weight keys and raises a
`ValueError` naming the mismatch *before* calling `load_weights` -- callers
must use this instead of calling `model.load_weights(strict=False)` directly,
which silently drops any key that doesn't match and can leave whole layers at
random initialization with no error. This exact check caught a real bug
during this port: `BSMamba2MLX.layers` was originally a list of *tuples* of
submodules, which `mlx.utils.tree_flatten` silently does not traverse (no
error at model-construction time) -- every `layers.N.*` converted tensor
showed up as "dropped" here and the loader raised naming all of them. Fixed
by switching to a list of lists (see `model.py`).

Layout differences from torch (verified empirically, not assumed from any
sibling package -- this package's Conv1d/ConvTranspose1d/Linear shapes were
checked directly against a fresh MLX module instance before writing this):
  - `nn.Linear` weight: (out, in) both frameworks -- no reshape.
  - `nn.Conv1d` weight: torch (out, in/groups, kernel) -> mlx (out, kernel,
    in/groups), i.e. `.transpose(0, 2, 1)`.
  - `nn.ConvTranspose1d` weight: torch (in, out, kernel) -> mlx (out, kernel,
    in), i.e. `.transpose(1, 2, 0)`.
  - 1-D parameters (biases, `A_log`, `D`, `dt_bias`, norm `gamma`/`weight`):
    identical shape, no reshape.

Reads: mlx.core, mlx.nn (via mlx.utils.tree_flatten), numpy
"""

from __future__ import annotations

import re
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten


def _to_numpy(value: Any) -> np.ndarray:
    try:
        return value.detach().cpu().numpy()
    except AttributeError:
        return np.array(value)


# --- key renames -------------------------------------------------------
# Applied in order; each is a (pattern, replacement) regex pair. Most of the
# tree (layers.N.{0,1}.{forward_blocks,backward_blocks}.0.{mixer,norm}.*,
# layers.N.{0,1}.linear.*, final_norm.gamma) needs NO renaming at all --
# this MLX port's submodule names were deliberately chosen to match torch's
# 1:1 there. Only band_split and mask_estimators, whose torch side uses
# nn.Sequential (which has no name of its own, only positional indices),
# need an explicit rename.
_KEY_RULES = [
    (re.compile(r"^band_split\.to_features\.(\d+)\.0\.gamma$"), r"band_split.norms.\1.gamma"),
    (re.compile(r"^band_split\.to_features\.(\d+)\.1\.weight$"), r"band_split.linears.\1.weight"),
    (re.compile(r"^band_split\.to_features\.(\d+)\.1\.bias$"), r"band_split.linears.\1.bias"),
]

# torch: mask_estimators.N.to_freqs.M.0.{2k}.{weight,bias}
#   to_freqs.M is a ModuleList entry; its `.0` is the wrapping Sequential's
#   first child (the _mlp Sequential itself; `.1` is the parameter-free GLU).
#   Inside the _mlp Sequential, Linear/Tanh alternate, so weight-bearing
#   indices are even: 0, 2, 4, ... -> mlp linear index k = torch_index // 2.
_MASK_MLP = re.compile(r"^mask_estimators\.(\d+)\.to_freqs\.(\d+)\.0\.(\d+)\.(weight|bias)$")


def _rename(key: str) -> str | None:
    """Return the MLX key for `key`, or None if this tensor is dropped
    entirely (there are none in this model, but keeping the hook symmetric
    with the sibling packages' convert.py shape)."""
    match = _MASK_MLP.match(key)
    if match:
        estimator, band, torch_index, kind = match.groups()
        linear_index = int(torch_index) // 2
        return f"mask_estimators.{estimator}.mlps.{band}.linears.{linear_index}.{kind}"
    for pattern, replacement in _KEY_RULES:
        if pattern.match(key):
            return pattern.sub(replacement, key)
    return key  # unchanged: layers.*, final_norm.gamma


def convert_torch_to_mlx_weights(state_dict: dict[str, Any]) -> dict[str, mx.array]:
    """Convert a BSMamba2 torch state dict to BSMamba2MLX's weight dict."""
    weights: dict[str, mx.array] = {}
    for key, value in state_dict.items():
        mlx_key = _rename(key)
        if mlx_key is None:
            continue
        array = _to_numpy(value)
        if key.endswith("conv1d.weight"):
            array = array.transpose(0, 2, 1)  # torch (out, in/groups, k) -> mlx (out, k, in/groups)
        elif key.endswith("linear.weight") and array.ndim == 3:
            array = array.transpose(1, 2, 0)  # ConvTranspose1d: torch (in, out, k) -> mlx (out, k, in)
        weights[mlx_key] = mx.array(array)
    return weights


def load_converted_weights(model, mlx_weights: dict[str, mx.array]) -> None:
    """Load `mlx_weights` into `model`, refusing a silent partial load.

    `model.load_weights(..., strict=False)` on its own accepts any degree of
    mismatch, dropping whatever doesn't line up without a warning. This
    checks first: every one of the model's own parameter keys (from
    `mlx.utils.tree_flatten(model.parameters())`) must be present in
    `mlx_weights`, and every key in `mlx_weights` must be consumed by the
    model -- otherwise a `ValueError` is raised naming counts and up to 5
    example keys on each side.
    """
    model_keys = {key for key, _ in tree_flatten(model.parameters())}
    weight_keys = set(mlx_weights.keys())

    unmatched_model = sorted(model_keys - weight_keys)
    dropped_weights = sorted(weight_keys - model_keys)

    if unmatched_model or dropped_weights:
        parts = []
        if unmatched_model:
            example = ", ".join(unmatched_model[:5])
            parts.append(f"{len(unmatched_model)} model parameters unmatched (e.g. {example})")
        if dropped_weights:
            example = ", ".join(dropped_weights[:5])
            parts.append(f"{len(dropped_weights)} converted tensors dropped (e.g. {example})")
        raise ValueError("MLX weight conversion incomplete: " + ", ".join(parts))

    mismatched_shapes = []
    model_params = dict(tree_flatten(model.parameters()))
    for key, array in mlx_weights.items():
        expected = model_params[key].shape
        if tuple(array.shape) != tuple(expected):
            mismatched_shapes.append(f"{key}: model={expected} converted={tuple(array.shape)}")
    if mismatched_shapes:
        raise ValueError("MLX weight conversion shape mismatch: " + "; ".join(mismatched_shapes[:5]))

    model.load_weights(list(mlx_weights.items()), strict=False)
