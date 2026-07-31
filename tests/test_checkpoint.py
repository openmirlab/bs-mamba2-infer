# /nav: offline registry and integrity checks.
# Exercises the package-owned checkpoint metadata and shared resolver without
# downloading a real weight file. Reads: checkpoint.py and package TOML data.
from __future__ import annotations

from pathlib import Path

import pytest

from bs_mamba2_infer.checkpoint import _parse_specs, checkpoint_specs, obtain, resolved_path, verify


def test_registry_contains_only_bsmamba2_vocals() -> None:
    specs = checkpoint_specs()
    assert set(specs) == {"official-vocals", "msst-vocals"}
    assert all(spec.architecture == "bsmamba2-vocals" for spec in specs.values())
    assert all(len(spec.sha256) == 64 for spec in specs.values())


def test_manual_missing_path_fails_before_download(tmp_path: Path) -> None:
    spec = checkpoint_specs()["official-vocals"]
    assert resolved_path(spec, checkpoint=tmp_path / "missing.ckpt") is None
    with pytest.raises(FileNotFoundError):
        obtain(spec, checkpoint=tmp_path / "missing.ckpt")


def test_checksum_failure(tmp_path: Path) -> None:
    spec = checkpoint_specs()["official-vocals"]
    path = tmp_path / "wrong.ckpt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="checksum"):
        verify(path, spec)


@pytest.mark.parametrize("text", ["not = [valid", "[checkpoints.bad]\narchitecture = 'x'\n"])
def test_malformed_registry_fails_clearly(text: str) -> None:
    with pytest.raises(RuntimeError, match="Malformed|malformed"):
        _parse_specs(text)


def test_unavailable_checkpoint_refuses_download_with_a_useful_message(tmp_path):
    """official-vocals' upstream Drive URL serves a genuine 0-byte file.

    The entry is marked status="unavailable" rather than deleted: the audited
    sha256 stays, so a locally supplied copy (checkpoint=...) still verifies,
    and re-hosting stays a licence decision (NOASSERTION weights) that is not
    this package's to make. obtain() must refuse the download path loudly and
    point at the working alternative.
    """
    import pytest as _pytest

    from bs_mamba2_infer.checkpoint import checkpoint_specs, obtain

    spec = checkpoint_specs()["official-vocals"]
    assert spec.status == "unavailable"

    with _pytest.raises(RuntimeError, match="msst-vocals"):
        obtain(spec, cache_dir=tmp_path)


def test_default_checkpoint_is_the_working_one():
    """The session/CLI default must be a checkpoint whose URL actually serves
    bytes -- msst-vocals (MSST release asset), not the dead official-vocals."""
    import inspect

    from bs_mamba2_infer.api import BSMamba2Session
    from bs_mamba2_infer.checkpoint import checkpoint_specs

    default = inspect.signature(BSMamba2Session.__init__).parameters["checkpoint_id"].default
    assert default == "msst-vocals"
    assert checkpoint_specs()[default].status == "available"
