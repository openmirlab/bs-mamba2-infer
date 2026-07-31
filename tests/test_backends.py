"""Backend seam contract: resolution, import purity, and honest failure.

These are offline and hardware-independent. They guard the two properties the
seam exists to protect -- that a requested backend is honoured or refused,
never silently swapped, and that the default import path stays free of the
optional MLX framework.

`test_importing_the_package_does_not_pull_in_an_optional_framework` runs in a
subprocess deliberately: other tests in this file call `MLXBackend.
is_available()`, which really does `import mlx.core` on a machine that has it
installed (this dev environment does) -- that import lands in `sys.modules`
for the rest of the pytest process, and a same-process "does `import
bs_mamba2_infer` alone pull in mlx" check would then pass or fail depending on
test *order*, not on the actual import graph. A sibling package (mdxnet-infer)
hit exactly this flake.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from bs_mamba2_infer.backends import (
    BACKEND_NAMES,
    DEFAULT_BACKEND,
    BackendUnavailable,
    get_backend,
    resolve_backend_name,
)


def test_default_and_none_resolve_to_torch():
    assert resolve_backend_name(None) == DEFAULT_BACKEND == "torch"
    assert resolve_backend_name("torch") == "torch"


def test_auto_resolves_to_a_real_backend_name():
    assert resolve_backend_name("auto") in BACKEND_NAMES


def test_unknown_backend_name_raises_value_error():
    with pytest.raises(ValueError):
        resolve_backend_name("cuda")


def test_unavailable_mlx_raises_rather_than_substituting(monkeypatch):
    """An explicit request is honoured or fails loudly -- never downgraded.

    Forces unavailability via monkeypatch rather than relying on ambient
    hardware, so this test is correct whether or not MLX happens to be
    installed in the environment running it.
    """
    from bs_mamba2_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: False))
    with pytest.raises(BackendUnavailable):
        resolve_backend_name("mlx")


def test_auto_falls_back_to_torch_when_mlx_is_unavailable(monkeypatch):
    from bs_mamba2_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: False))
    assert resolve_backend_name("auto") == "torch"


def test_auto_prefers_mlx_when_available(monkeypatch):
    from bs_mamba2_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: True))
    assert resolve_backend_name("auto") == "mlx"


def test_torch_backend_satisfies_the_protocol_surface():
    backend = get_backend("torch")
    assert backend.name == "torch"
    assert backend.is_available() is True
    for method in ("separate", "release"):
        assert hasattr(backend, method), f"TorchBackend is missing {method}"


def test_mlx_backend_has_the_name_class_attribute():
    backend = get_backend("mlx")
    assert backend.name == "mlx"


@pytest.mark.parametrize("device", [None, "auto", "mps"])
def test_mlx_accepts_only_its_own_execution_target(device):
    from bs_mamba2_infer.backends.mlx_backend import MLXBackend

    assert MLXBackend._select_device(device) == "mps"


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cpu"])
def test_mlx_refuses_a_torch_device_rather_than_reinterpreting_it(device):
    """Treating device="cuda" as "the Apple GPU anyway" would discard what the
    caller explicitly asked for -- the same rule CUDA device selection already
    follows in api.py's _resolve_device."""
    from bs_mamba2_infer.backends.mlx_backend import MLXBackend

    with pytest.raises(BackendUnavailable):
        MLXBackend._select_device(device)


def test_session_mlx_device_is_never_silently_ignored(monkeypatch):
    """A sibling package once accepted backend='mlx' with any device string
    and silently ignored it (even advertising that in --help); another
    overloaded device='mlx' and had to revert. Neither is acceptable here:
    an invalid device for the MLX backend must raise at Session construction,
    before any checkpoint work happens."""
    from bs_mamba2_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: True))
    from bs_mamba2_infer import BSMamba2Session

    with pytest.raises(BackendUnavailable):
        BSMamba2Session(backend="mlx", device="cuda")


def test_importing_the_package_does_not_pull_in_an_optional_framework():
    """`pip install bs-mamba2-infer` must stay MLX-free and import-clean.

    Subprocess-isolated (see module docstring) so an earlier test's real
    `MLXBackend.is_available()` call -- which really does `import mlx.core`
    when MLX is installed -- cannot contaminate this check via `sys.modules`
    left over in the same process.
    """
    script = (
        "import sys\n"
        "import bs_mamba2_infer\n"
        "optional = {'mlx', 'mlx_spectro'}\n"
        "leaked = sorted({m.split('.')[0] for m in sys.modules} & optional)\n"
        "assert not leaked, f'import bs_mamba2_infer pulled in optional frameworks: {leaked}'\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


def test_check_stitched_length_accepts_the_expected_range():
    """`_check_stitched_length` mirrors `audio.py`'s `_istft` fail-loud
    contract for `MLXBackend.separate()`'s final stitch-then-crop step.
    Offline: this is pure arithmetic, no MLX import required."""
    from bs_mamba2_infer.backends.mlx_backend import _check_stitched_length

    _check_stitched_length(stitched_length=100, duration=100, padding_add=0)  # exact
    _check_stitched_length(stitched_length=105, duration=100, padding_add=5)  # padded to the limit


def test_check_stitched_length_rejects_shorter_than_duration():
    from bs_mamba2_infer.backends.mlx_backend import _check_stitched_length

    with pytest.raises(RuntimeError, match="Unexpected stitched MLX output length"):
        _check_stitched_length(stitched_length=99, duration=100, padding_add=5)


def test_check_stitched_length_rejects_longer_than_final_chunk_padding():
    from bs_mamba2_infer.backends.mlx_backend import _check_stitched_length

    with pytest.raises(RuntimeError, match="Unexpected stitched MLX output length"):
        _check_stitched_length(stitched_length=106, duration=100, padding_add=5)
