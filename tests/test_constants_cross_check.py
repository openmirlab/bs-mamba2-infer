# /nav: offline duplication lock between audio.py and backends/mlx_backend.py.
# Owns nothing new -- it only asserts two already-existing declarations agree,
# so a future edit to one side cannot silently drift from the other unnoticed.
# Reads: audio.py, backends/mlx_backend.py, model.py, mlx/model.py
"""Locks the constants `backends/mlx_backend.py` re-declares as literals.

`audio.py` owns SAMPLE_RATE/N_FFT/WIN_LENGTH/HOP_LENGTH/CHUNK_SAMPLES as the
single source of truth; `backends/mlx_backend.py` re-declares the same values
as literals (MLX's STFT call needs plain Python ints, not a cross-module
import at that call site) -- see that module's docstring for why the
duplication exists. This test does NOT migrate the duplication away; it only
locks it so an edit to one side that is not mirrored on the other fails here
instead of only showing up as a silent numerical drift in production. Real
consolidation is deferred to post-merge.

It also locks an undocumented arithmetic coupling: `DEFAULT_FREQS_PER_BANDS`
(declared separately in `model.py` and `mlx/model.py`) must sum to exactly
`N_FFT // 2 + 1`, the STFT frequency-bin count that `audio.py`'s STFT and the
band-split stage both depend on. Nothing else in the codebase enforces this;
without it, a change to `N_FFT` fails as a deep shape error inside band-split
with no pointer back to this coupling.
"""
from __future__ import annotations

from bs_mamba2_infer.audio import (
    CHUNK_SAMPLES,
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    WIN_LENGTH,
)
from bs_mamba2_infer.backends import mlx_backend
from bs_mamba2_infer.mlx.model import DEFAULT_FREQS_PER_BANDS as MLX_DEFAULT_FREQS_PER_BANDS
from bs_mamba2_infer.model import DEFAULT_FREQS_PER_BANDS


def test_mlx_backend_constants_match_audio_py():
    assert mlx_backend.SAMPLE_RATE == SAMPLE_RATE
    assert mlx_backend.N_FFT == N_FFT
    assert mlx_backend.WIN_LENGTH == WIN_LENGTH
    assert mlx_backend.HOP_LENGTH == HOP_LENGTH
    assert mlx_backend.CHUNK_SAMPLES == CHUNK_SAMPLES


def test_default_freqs_per_bands_sums_to_stft_bin_count():
    expected = N_FFT // 2 + 1
    assert sum(DEFAULT_FREQS_PER_BANDS) == expected
    assert sum(MLX_DEFAULT_FREQS_PER_BANDS) == expected


def test_torch_and_mlx_freqs_per_bands_agree():
    assert DEFAULT_FREQS_PER_BANDS == MLX_DEFAULT_FREQS_PER_BANDS
