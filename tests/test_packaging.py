# /nav: published-metadata contracts.
# Ensures the optional accelerator cannot break the core-supported Python 3.13
# install because mamba_ssm's isolated build omits its Torch build requirement.
# Reads: pyproject.toml only; no package build or network access is required.
from __future__ import annotations

from pathlib import Path

import pytest

from bs_mamba2_infer.model import MambaBlock

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib


def test_cuda_extra_never_triggers_mambas_broken_isolated_build() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    cuda = project["project"]["optional-dependencies"]["cuda"]
    assert cuda == []
    assert all("mamba-ssm" not in dependency for dependency in project["project"]["dependencies"])


def test_missing_native_accelerator_has_an_actionable_install_hint() -> None:
    with pytest.raises(RuntimeError, match="--no-build-isolation.*mamba-ssm==2.2.2"):
        MambaBlock(256, layer_idx=0, backend="native")
