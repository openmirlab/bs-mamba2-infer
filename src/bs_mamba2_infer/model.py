# /nav: faithful BSMamba2 inference graph.
# Ports only the graph exercised by EuiYeonKim/BSMamba2's vocal checkpoint:
# band split, bidirectional Mamba2 blocks, and mask estimation.  Native
# mamba_ssm is an optional accelerator; the reference-compatible fallback uses
# only PyTorch and preserves upstream state-dict names.
# Reads: api.py for construction.
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# Invariant: sums to exactly N_FFT // 2 + 1 (1025, from audio.py's N_FFT=2048) --
# the STFT frequency-bin count. A change to N_FFT without a matching change here
# is a shape error deep in band-split with no pointer back to this line.
DEFAULT_FREQS_PER_BANDS = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, value: Tensor) -> Tensor:
        return F.normalize(value, dim=-1) * self.scale * self.gamma


class _MambaRMSNorm(nn.Module):
    """Pure-PyTorch equivalent of mamba_ssm's gated RMSNorm (inference only)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: Tensor, gate: Tensor | None = None) -> Tensor:
        # Upstream config uses norm_before_gate=False.
        if gate is not None:
            value = value * F.silu(gate)
        value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
        return value * self.weight


class _PureMamba2(nn.Module):
    """Mamba2's non-fused inference algorithm, expressed with ordinary torch ops."""

    def __init__(self, d_model: int, *, layer_idx: int | None = None, expand: int = 4) -> None:
        super().__init__()
        self.d_model, self.d_state, self.d_conv, self.expand = d_model, 128, 4, expand
        self.d_inner, self.headdim, self.ngroups = expand * d_model, 64, 1
        self.d_ssm, self.nheads = self.d_inner, self.d_inner // self.headdim
        self.layer_idx = layer_idx
        projected = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, projected, bias=False)
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, self.d_conv, groups=conv_dim, padding=self.d_conv - 1)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = _MambaRMSNorm(self.d_ssm, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, value: Tensor, inference_params: object | None = None) -> Tensor:
        del inference_params
        batch, length, _ = value.shape
        zxbcdt = self.in_proj(value)
        z, xbc, dt = torch.split(zxbcdt, [self.d_ssm, self.d_ssm + 2 * self.d_state, self.nheads], dim=-1)
        xbc = F.silu(self.conv1d(xbc.transpose(1, 2)).transpose(1, 2)[:, :length, :])
        x, b_coeff, c_coeff = torch.split(xbc, [self.d_ssm, self.d_state, self.d_state], dim=-1)
        x = x.reshape(batch, length, self.nheads, self.headdim)
        # The state is deliberately per head to keep the fallback's peak memory
        # bounded.  It is exact but slow; optional mamba_ssm is for production CUDA.
        outputs: list[Tensor] = []
        a = -torch.exp(self.A_log.float())
        delta = F.softplus(dt + self.dt_bias)
        for head in range(self.nheads):
            state = torch.zeros(batch, self.headdim, self.d_state, dtype=value.dtype, device=value.device)
            head_values: list[Tensor] = []
            for index in range(length):
                d_a = torch.exp(delta[:, index, head].float() * a[head]).to(value.dtype)
                injected = torch.einsum("b,bn,bp->bpn", delta[:, index, head].to(value.dtype), b_coeff[:, index], x[:, index, head])
                state = state * d_a[:, None, None] + injected
                current = torch.einsum("bpn,bn->bp", state, c_coeff[:, index])
                head_values.append(current + self.D[head].to(value.dtype) * x[:, index, head])
            outputs.append(torch.stack(head_values, dim=1))
        y = torch.cat(outputs, dim=-1)
        return self.out_proj(self.norm(y, z))


class _PureBlock(nn.Module):
    def __init__(self, dim: int, *, layer_idx: int) -> None:
        super().__init__()
        self.norm = _MambaRMSNorm(dim, eps=1e-5)
        self.mixer = _PureMamba2(dim, layer_idx=layer_idx, expand=4)
        self.mlp = None

    def forward(self, hidden_states: Tensor, residual: Tensor | None = None, inference_params: object | None = None) -> tuple[Tensor, Tensor]:
        residual = hidden_states + residual if residual is not None else hidden_states
        hidden_states = self.norm(residual)
        return self.mixer(hidden_states, inference_params=inference_params), residual


def _native_block(dim: int, layer_idx: int) -> nn.Module | None:
    try:
        from mamba_ssm.modules.block import Block
        from mamba_ssm.modules.mamba2 import Mamba2
        from mamba_ssm.ops.triton.layer_norm import RMSNorm as NativeRMSNorm
    except ImportError:
        return None
    return Block(
        dim,
        mixer_cls=partial(Mamba2, layer_idx=layer_idx, expand=4),
        norm_cls=partial(NativeRMSNorm, eps=1e-5),
        fused_add_norm=False,
        mlp_cls=nn.Identity,
    )


class MambaBlock(nn.Module):
    def __init__(self, dim: int, *, layer_idx: int, backend: Literal["auto", "native", "torch"] = "auto") -> None:
        super().__init__()
        native = _native_block(dim, layer_idx) if backend != "torch" else None
        if backend == "native" and native is None:
            raise RuntimeError(
                "The native backend needs mamba-ssm. Install compatible Torch, then run "
                "`pip install --no-build-isolation 'mamba-ssm==2.2.2'`."
            )
        self.forward_blocks = nn.ModuleList([native or _PureBlock(dim, layer_idx=layer_idx)])
        backward = _native_block(dim, layer_idx) if backend != "torch" else None
        self.backward_blocks = nn.ModuleList([backward or _PureBlock(dim, layer_idx=layer_idx)])
        self.linear = nn.ConvTranspose1d(dim * 2, dim, kernel_size=1)
        self.backend = "native" if native is not None else "torch"

    def forward(self, value: Tensor) -> Tensor:
        residual = None
        forward = value.clone()
        for block in self.forward_blocks:
            forward, residual = block(forward, residual, inference_params=None)
        forward = forward + residual if residual is not None else forward
        residual = None
        backward = torch.flip(value, [1])
        for block in self.backward_blocks:
            backward, residual = block(backward, residual, inference_params=None)
        backward = backward + residual if residual is not None else backward
        merged = torch.cat((forward, torch.flip(backward, [1])), dim=-1)
        return self.linear(merged.transpose(1, 2)).transpose(1, 2).contiguous() + value


class BandSplit(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...]) -> None:
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList([nn.Sequential(RMSNorm(size), nn.Linear(size, dim)) for size in dim_inputs])

    def forward(self, value: Tensor) -> Tensor:
        return torch.stack([layer(part) for part, layer in zip(value.split(self.dim_inputs, dim=-1), self.to_features)], dim=-2)


def _mlp(dim_in: int, dim_out: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dimensions = (dim_in,) + (hidden,) * (depth - 1) + (dim_out,)
    for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(nn.Linear(left, right))
        if index != len(dimensions) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class MaskEstimator(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...], depth: int) -> None:
        super().__init__()
        self.to_freqs = nn.ModuleList([nn.Sequential(_mlp(dim, size * 2, dim * 4, depth), nn.GLU(dim=-1)) for size in dim_inputs])

    def forward(self, value: Tensor) -> Tensor:
        return torch.cat([layer(part) for part, layer in zip(value.unbind(dim=-2), self.to_freqs)], dim=-1)


@dataclass(frozen=True)
class BSMamba2Config:
    dim: int = 256
    depth: int = 6
    stereo: bool = True
    num_stems: int = 1
    mask_estimator_depth: int = 2


class BSMamba2(nn.Module):
    """Compatible BSMamba2 vocal mask estimator; intentionally no Mamba v1 branch."""

    def __init__(self, config: BSMamba2Config = BSMamba2Config(), *, backend: Literal["auto", "native", "torch"] = "auto") -> None:
        super().__init__()
        self.stereo, self.num_stems, self.dim = config.stereo, config.num_stems, config.dim
        channels = 2 if self.stereo else 1
        self.layers = nn.ModuleList([nn.ModuleList((MambaBlock(config.dim, layer_idx=index, backend=backend), MambaBlock(config.dim, layer_idx=index, backend=backend))) for index in range(config.depth)])
        self.final_norm = RMSNorm(config.dim)
        inputs = tuple(2 * size * channels for size in DEFAULT_FREQS_PER_BANDS)
        self.band_split = BandSplit(config.dim, inputs)
        self.mask_estimators = nn.ModuleList([MaskEstimator(config.dim, inputs, config.mask_estimator_depth) for _ in range(config.num_stems)])

    @property
    def mamba_backend(self) -> str:
        return self.layers[0][0].backend

    def forward(self, value: Tensor) -> Tensor:
        real = torch.view_as_real(value)
        real = real.permute(0, 2, 1, 3, 4).reshape(value.shape[0], value.shape[2] * value.shape[1], value.shape[3], 2)
        features = self.band_split(real.permute(0, 2, 1, 3).reshape(value.shape[0], value.shape[3], -1))
        for time_module, frequency_module in self.layers:
            batch, time, bands, dim = features.shape
            features = time_module(features.permute(0, 2, 1, 3).reshape(batch * bands, time, dim)).reshape(batch, bands, time, dim).permute(0, 2, 1, 3)
            features = frequency_module(features.reshape(batch * time, bands, dim)).reshape(batch, time, bands, dim)
        mask = torch.stack([estimator(self.final_norm(features)) for estimator in self.mask_estimators], dim=1)
        mask = mask.reshape(value.shape[0], self.num_stems, value.shape[3], value.shape[2] * value.shape[1], 2).permute(0, 1, 3, 2, 4)
        source = torch.view_as_complex(real.unsqueeze(1).contiguous()) * torch.view_as_complex(mask.float().contiguous())
        source = source.reshape(value.shape[0], self.num_stems, value.shape[2], value.shape[1], value.shape[3])
        return source.permute(0, 1, 3, 2, 4).contiguous()
