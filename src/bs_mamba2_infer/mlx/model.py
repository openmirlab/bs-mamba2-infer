# /nav: from-scratch MLX port of model.py's frequency-domain graph.
# No upstream MLX BSMamba2 exists to vendor. Real-valued internally; complex
# only at the input/output edges. STFT/iSTFT/chunking are NOT here -- see
# backends/mlx_backend.py, mirroring model.py/audio.py's own boundary.
# Reads: mlx.core, mlx.nn
"""From-scratch MLX port of BSMamba2's frequency-domain graph (model.py).

There is no upstream MLX BSMamba2 to vendor -- this is derived directly from
this package's own `bs_mamba2_infer.model`, not copied from any other
package. Only the *pure-PyTorch fallback* graph (`_PureMamba2`/`_PureBlock`)
is ported; the optional CUDA `mamba_ssm` native backend has no MLX
equivalent and is irrelevant here -- MLX always runs the einsum recurrence.

Module-boundary note (does NOT transfer from the sibling MLX ports): unlike
bs-roformer/melband/demucs, whose Torch models do their own STFT internally,
this package's `BSMamba2.forward()` takes an already-computed **complex**
STFT tensor and returns a complex masked spectrogram -- the STFT/iSTFT and
all chunking geometry live entirely in `audio.py`, outside the model. This
MLX port preserves that exact boundary: this module is the frequency-domain
graph only (real-valued internally, complex only at its input/output edges,
handled with a plain real/imag pair rather than MLX complex dtype so the
whole trunk needs zero complex-number reasoning);
`backends/mlx_backend.py` owns STFT/iSTFT/chunking, mirroring `audio.py`.

Reading `BSMamba2.forward()` closely matters: `torch.view_as_real` converts
the complex input to a real tensor **immediately**, and the entire
band-split/Mamba/mask-estimator stack runs in real-valued arithmetic --
complex dtype only reappears at the final masking multiply. `BSMamba2MLX.
__call__`'s `stacked`/`merged_real`/`merged_imag` setup and `_complex_mul` at
the bottom of this file are the only places that touch a real/imag pair as
such; everything between them is ordinary real-valued arithmetic.

Deliberate speed optimization, verified not to be an accuracy trade: the
Mamba2 recurrence is vectorized across the 16 heads (one batched einsum per
timestep instead of Torch's nested per-head Python loop) -- this changes
loop *structure*, not the math, and is checked against Torch numerically in
tests/test_mlx_parity.py, not merely assumed safe. The per-timestep Python
loop itself (~800 steps per 8s chunk) is not removable without reproducing
mamba_ssm's fused CUDA/Triton kernel, which does not exist for MLX.

`mx.split`'s list argument is numpy-style cut *points*, not torch.split's
per-part *sizes* -- `_split_sizes` below converts once so every call site
reads like the Torch original instead of silently taking the wrong slices.

Reads: mlx.core, mlx.nn
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import mlx.core as mx
from mlx import nn

# Invariant: sums to exactly N_FFT // 2 + 1 (1025, from audio.py's N_FFT=2048) --
# the STFT frequency-bin count. A change to N_FFT without a matching change here
# is a shape error deep in band-split with no pointer back to this line.
DEFAULT_FREQS_PER_BANDS = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)


def _split_sizes(value: mx.array, sizes: list[int], axis: int = -1) -> list[mx.array]:
    """torch.split(value, sizes, dim)-compatible: `sizes` are part sizes, not
    mx.split's native cut-point indices."""
    cuts = []
    total = 0
    for size in sizes[:-1]:
        total += size
        cuts.append(total)
    return mx.split(value, cuts, axis=axis)


class RMSNormMLX(nn.Module):
    """Matches torch's `F.normalize(value, dim=-1) * scale * gamma`.

    `F.normalize` divides by `max(||value||_2, eps=1e-12)` -- replicated
    exactly rather than a mean-square RMS formula, which is a different (if
    similar) normalization.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = mx.ones((dim,))

    def __call__(self, value: mx.array) -> mx.array:
        norm = mx.sqrt(mx.sum(value.astype(mx.float32) ** 2, axis=-1, keepdims=True))
        normalized = value.astype(mx.float32) / mx.maximum(norm, 1e-12)
        return normalized.astype(value.dtype) * self.scale * self.gamma


class MambaRMSNormMLX(nn.Module):
    """Mamba2's gated RMSNorm, matching `_MambaRMSNorm` (mean-square form, eps=1e-5)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, value: mx.array, gate: mx.array | None = None) -> mx.array:
        if gate is not None:
            value = value * nn.silu(gate)
        value = value * mx.rsqrt(mx.mean(value.astype(mx.float32) ** 2, axis=-1, keepdims=True) + self.eps)
        return value * self.weight


class PureMamba2MLX(nn.Module):
    """Mamba2's non-fused inference algorithm as ordinary MLX ops, head-vectorized.

    `state` carries a `heads` axis instead of being recomputed inside a
    per-head Python loop -- see the module docstring for why this is safe.
    """

    def __init__(self, d_model: int, *, expand: int = 4) -> None:
        super().__init__()
        self.d_model, self.d_state, self.d_conv, self.expand = d_model, 128, 4, expand
        self.d_inner, self.headdim, self.ngroups = expand * d_model, 64, 1
        self.d_ssm, self.nheads = self.d_inner, self.d_inner // self.headdim
        projected = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, projected, bias=False)
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, self.d_conv, groups=conv_dim, padding=self.d_conv - 1)
        self.dt_bias = mx.zeros((self.nheads,))
        self.A_log = mx.zeros((self.nheads,))
        self.D = mx.ones((self.nheads,))
        self.norm = MambaRMSNormMLX(self.d_ssm, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def __call__(self, value: mx.array) -> mx.array:
        batch, length, _ = value.shape
        zxbcdt = self.in_proj(value)
        z, xbc, dt = _split_sizes(zxbcdt, [self.d_ssm, self.d_ssm + 2 * self.d_state, self.nheads])

        # Depthwise "causal" conv1d: symmetric (d_conv-1) padding via MLX's
        # Conv1d, same as torch's nn.Conv1d(padding=d_conv-1), then keep only
        # the first `length` outputs -- discarding the future-looking tail is
        # what makes this causal despite the symmetric padding. MLX's Conv1d
        # is already NLC (batch, length, channels), unlike torch's NCL, so no
        # transpose is needed here (torch transposes in and out; that dance
        # is a layout-convention artifact, not part of the math).
        xbc = nn.silu(self.conv1d(xbc)[:, :length, :])
        x, b_coeff, c_coeff = _split_sizes(xbc, [self.d_ssm, self.d_state, self.d_state])
        x = x.reshape(batch, length, self.nheads, self.headdim)

        a = -mx.exp(self.A_log.astype(mx.float32))  # (nheads,)
        delta = nn.softplus(dt + self.dt_bias)  # (batch, length, nheads)

        state = mx.zeros((batch, self.nheads, self.headdim, self.d_state), dtype=value.dtype)
        head_values = []
        for index in range(length):
            delta_t = delta[:, index, :]  # (batch, nheads)
            d_a = mx.exp(delta_t.astype(mx.float32) * a).astype(value.dtype)  # (batch, nheads)
            injected = mx.einsum(
                "bh,bn,bhp->bhpn",
                delta_t.astype(value.dtype),
                b_coeff[:, index].astype(value.dtype),
                x[:, index],
            )
            state = state * d_a[:, :, None, None] + injected
            current = mx.einsum("bhpn,bn->bhp", state, c_coeff[:, index].astype(value.dtype))
            head_values.append(current + self.D.astype(value.dtype)[None, :, None] * x[:, index])
        y = mx.stack(head_values, axis=1)  # (batch, length, nheads, headdim)
        y = y.reshape(batch, length, self.d_ssm)
        return self.out_proj(self.norm(y, z))


class PureBlockMLX(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = MambaRMSNormMLX(dim, eps=1e-5)
        self.mixer = PureMamba2MLX(dim, expand=4)

    def __call__(self, hidden_states: mx.array, residual: mx.array | None = None) -> tuple[mx.array, mx.array]:
        residual = hidden_states + residual if residual is not None else hidden_states
        hidden_states = self.norm(residual)
        return self.mixer(hidden_states), residual


class MambaBlockMLX(nn.Module):
    """Bidirectional wrapper: forward_blocks/backward_blocks match Torch's
    single-element ModuleList naming so state-dict conversion is a rename,
    not a restructure -- MLX has no native `native`/`mamba_ssm` backend
    branch, so this is always the pure-torch-equivalent path."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.forward_blocks = [PureBlockMLX(dim)]
        self.backward_blocks = [PureBlockMLX(dim)]
        self.linear = nn.ConvTranspose1d(dim * 2, dim, kernel_size=1)

    def __call__(self, value: mx.array) -> mx.array:
        residual = None
        forward = value
        for block in self.forward_blocks:
            forward, residual = block(forward, residual)
        forward = forward + residual if residual is not None else forward

        residual = None
        backward = value[:, ::-1, :]
        for block in self.backward_blocks:
            backward, residual = block(backward, residual)
        backward = backward + residual if residual is not None else backward

        merged = mx.concatenate([forward, backward[:, ::-1, :]], axis=-1)
        # ConvTranspose1d(kernel_size=1) on NLC input needs no transpose
        # (torch transposes to NCL and back; MLX's conv modules are already
        # NLC, see PureMamba2MLX's conv1d comment above for the same point).
        return self.linear(merged) + value


class BandSplitMLX(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...]) -> None:
        super().__init__()
        self.dim_inputs = dim_inputs
        self.norms = [RMSNormMLX(size) for size in dim_inputs]
        self.linears = [nn.Linear(size, dim) for size in dim_inputs]

    def __call__(self, value: mx.array) -> mx.array:
        parts = _split_sizes(value, list(self.dim_inputs))
        return mx.stack([linear(norm(part)) for part, norm, linear in zip(parts, self.norms, self.linears)], axis=-2)


class _MLPMLX(nn.Module):
    """torch's `_mlp(dim_in, dim_out, hidden, depth)`: `depth` Linear layers
    with Tanh between (not after) every layer but the last -- (depth-1) Tanh
    activations for `depth` Linears. Ported as a plain list, not
    nn.Sequential, to keep this file's state-dict key convention explicit
    (see convert.py) rather than relying on MLX Sequential's `.layers.N`
    insertion rule.
    """

    def __init__(self, dim_in: int, dim_out: int, hidden: int, depth: int) -> None:
        super().__init__()
        dimensions = (dim_in,) + (hidden,) * (depth - 1) + (dim_out,)
        self.linears = [nn.Linear(left, right) for left, right in pairwise(dimensions)]

    def __call__(self, value: mx.array) -> mx.array:
        for index, linear in enumerate(self.linears):
            value = linear(value)
            if index != len(self.linears) - 1:
                value = mx.tanh(value)
        return value


class MaskEstimatorMLX(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...], depth: int) -> None:
        super().__init__()
        self.dim_inputs = dim_inputs
        self.mlps = [_MLPMLX(dim, size * 2, dim * 4, depth) for size in dim_inputs]
        self.glu = nn.GLU(axis=-1)

    def __call__(self, value: mx.array) -> mx.array:
        parts = [self.glu(mlp(value[..., index, :])) for index, mlp in enumerate(self.mlps)]
        return mx.concatenate(parts, axis=-1)


@dataclass(frozen=True)
class BSMamba2ConfigMLX:
    dim: int = 256
    depth: int = 6
    stereo: bool = True
    num_stems: int = 1
    mask_estimator_depth: int = 2


class BSMamba2MLX(nn.Module):
    """Frequency-domain graph only -- see module docstring for the boundary.

    `__call__` takes and returns a `(real, imag)` pair, each real-valued
    MLX arrays shaped like torch's `torch.view_as_real(value)` /
    `torch.view_as_complex(...)` boundary, so this module needs no complex
    dtype at all.
    """

    def __init__(self, config: BSMamba2ConfigMLX = BSMamba2ConfigMLX()) -> None:
        super().__init__()
        self.stereo, self.num_stems, self.dim = config.stereo, config.num_stems, config.dim
        channels = 2 if self.stereo else 1
        # NOTE: a list of *tuples* of submodules is invisible to MLX's parameter
        # tree (mlx.utils.tree_flatten silently skips it -- no error, its
        # params just never appear in model.parameters() at all). Caught here
        # before it shipped precisely because load_converted_weights (below)
        # audits both directions: with tuples, every "layers.N.*" converted
        # tensor would show up as a *dropped* weight (nowhere on the model
        # side to match it against) and the loader would raise naming all of
        # them -- loud, not a silent random-init model. Switched to a list of
        # *lists*, which MLX's tree_flatten does traverse; verified empirically
        # before writing this, not assumed.
        self.layers = [[MambaBlockMLX(config.dim), MambaBlockMLX(config.dim)] for _ in range(config.depth)]
        self.final_norm = RMSNormMLX(config.dim)
        self.freqs_per_bands = DEFAULT_FREQS_PER_BANDS
        inputs = tuple(2 * size * channels for size in self.freqs_per_bands)
        self.dim_inputs = inputs
        self.band_split = BandSplitMLX(config.dim, inputs)
        self.mask_estimators = [MaskEstimatorMLX(config.dim, inputs, config.mask_estimator_depth) for _ in range(config.num_stems)]

    def __call__(self, real: mx.array, imag: mx.array) -> tuple[mx.array, mx.array]:
        """`real`/`imag`: (batch, channels, freq, time). Returns the same
        shape, stem axis already reduced to stem 0 (see `mlx_backend.py`:
        this package always extracts stem 0, matching `audio.py`'s
        `_istft(estimate[index, 0], ...)`)."""
        batch, channels, freq, time = real.shape
        # torch: real = view_as_real(value)                     -> (b, c, f, t, 2)
        #        real = real.permute(0, 2, 1, 3, 4).reshape(...) -> (b, f*c, t, 2)   [freq-major merge]
        #        features_in = real.permute(0, 2, 1, 3).reshape(b, t, -1)
        # `real`/`imag` here are already torch's view_as_real split into a
        # pair, so `stacked` reproduces the SAME (b, f*c, t, 2) freq-major
        # merge -- kept around (not discarded after band_split's input) because
        # torch's own `real` variable is REASSIGNED to this exact merged shape
        # and is what the final complex multiply below actually uses, not the
        # original (b, c, f, t) layout. Reshaping (channels, freq) directly
        # without this permute would merge channel-major instead of freq-major
        # -- silently wrong, not just differently laid out.
        stacked = mx.stack([real, imag], axis=-1)  # (b, c, f, t, 2)
        stacked = stacked.transpose(0, 2, 1, 3, 4).reshape(batch, freq * channels, time, 2)  # (b, f*c, t, 2)
        merged_real, merged_imag = stacked[..., 0], stacked[..., 1]  # (b, f*c, t) each

        features_in = stacked.transpose(0, 2, 1, 3).reshape(batch, time, freq * channels * 2)
        features = self.band_split(features_in)

        for time_module, frequency_module in self.layers:
            b, t, bands, dim = features.shape
            features = time_module(features.transpose(0, 2, 1, 3).reshape(b * bands, t, dim))
            features = features.reshape(b, bands, t, dim).transpose(0, 2, 1, 3)
            features = frequency_module(features.reshape(b * t, bands, dim))
            features = features.reshape(b, t, bands, dim)

        normed = self.final_norm(features)
        masks = mx.stack([estimator(normed) for estimator in self.mask_estimators], axis=1)
        # torch: mask.reshape(b, num_stems, t, f*c, 2).permute(0,1,3,2,4) -> (b, num_stems, f*c, t, 2)
        masks = masks.reshape(batch, len(self.mask_estimators), time, freq * channels, 2)
        masks = masks.transpose(0, 1, 3, 2, 4)
        mask_real, mask_imag = masks[..., 0], masks[..., 1]

        # complex mask multiply in the SAME (b, f*c, t) freq-major-merged
        # domain torch multiplies in -- stem 0 only (this package always
        # extracts stem 0, see audio.py's `_istft(estimate[index, 0], ...)`).
        source_real, source_imag = _complex_mul(merged_real, merged_imag, mask_real[:, 0], mask_imag[:, 0])
        # torch: source.reshape(b, num_stems, f, c, t).permute(0,1,3,2,4) -- but
        # we already sliced to stem 0, so reshape (f*c, t) -> (f, c, t) directly.
        source_real = source_real.reshape(batch, freq, channels, time).transpose(0, 2, 1, 3)
        source_imag = source_imag.reshape(batch, freq, channels, time).transpose(0, 2, 1, 3)
        return source_real, source_imag


def _complex_mul(ar: mx.array, ai: mx.array, br: mx.array, bi: mx.array) -> tuple[mx.array, mx.array]:
    """(ar + i*ai) * (br + i*bi), done with real arithmetic only."""
    return ar * br - ai * bi, ar * bi + ai * br
