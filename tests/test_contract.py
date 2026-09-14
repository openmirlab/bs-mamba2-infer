# /nav: offline public-contract and lifecycle tests.
# Uses small doubles rather than a checkpoint so these tests stay offline and
# prove session behavior independently of CUDA/golden hardware.
# Reads: public package API and api.py's model/load seams.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import bs_mamba2_infer.api as api
from bs_mamba2_infer import BSMamba2Session, SeparationResult, separate


class _FakeModel:
    created = 0
    backends: list[object] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        type(self).created += 1
        type(self).backends.append(kwargs.get("backend"))

    def load_state_dict(self, state: object, strict: bool = False) -> tuple[list[str], list[str]]:
        return [], []

    def eval(self) -> "_FakeModel":
        return self

    def to(self, device: object) -> "_FakeModel":
        return self


@pytest.fixture()
def loaded_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> BSMamba2Session:
    # Hermetic: cache_info() consults the real cache dir; without this, a
    # checkpoint genuinely present in ~/.cache flips its answers.
    monkeypatch.setenv("BS_MAMBA2_INFER_CACHE", str(tmp_path / "empty-cache"))
    path = tmp_path / "checkpoint.ckpt"
    torch.save({}, path)
    _FakeModel.created = 0
    _FakeModel.backends = []
    monkeypatch.setattr(api, "BSMamba2", _FakeModel)
    monkeypatch.setattr(api, "obtain", lambda *args, **kwargs: path)
    monkeypatch.setattr(api, "separate_waveform", lambda *args, **kwargs: torch.zeros((1, 2, 8)))
    return BSMamba2Session(device="cpu")


def test_public_symbols() -> None:
    assert BSMamba2Session.__name__ == "BSMamba2Session"
    assert callable(separate)
    assert SeparationResult(np.zeros((2, 1), np.float32), 44100, "official-vocals").sample_rate == 44100


def test_session_load_once_and_lifecycle(loaded_session: BSMamba2Session) -> None:
    with pytest.raises(RuntimeError, match="load"):
        loaded_session.infer(np.zeros((2, 8), np.float32), sample_rate=44100)
    loaded_session.load().load()
    first = loaded_session.infer(np.zeros((2, 8), np.float32), sample_rate=44100)
    second = loaded_session.infer(np.zeros((2, 8), np.float32), sample_rate=44100)
    assert first.vocals.shape == second.vocals.shape == (2, 8)
    assert _FakeModel.created == 1
    assert _FakeModel.backends == ["torch"]
    assert loaded_session.status == "ready"
    loaded_session.release()
    assert loaded_session.status == "released"
    loaded_session.load()
    assert _FakeModel.created == 2
    loaded_session.close()
    loaded_session.close()
    assert loaded_session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        loaded_session.load()


def test_cache_info_does_not_download(loaded_session: BSMamba2Session) -> None:
    info = loaded_session.cache_info()
    assert info["checkpoint_id"] == "msst-vocals"
    assert info["cached"] is False


@pytest.mark.parametrize("device", ["cuda", "cuda:999"])
def test_unavailable_cuda_is_not_silently_replaced(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        BSMamba2Session(device=device)


def test_mps_is_not_claimed() -> None:
    with pytest.raises(ValueError, match="MPS"):
        BSMamba2Session(device="mps")


def test_array_needs_rate(loaded_session: BSMamba2Session) -> None:
    loaded_session.load()
    with pytest.raises(ValueError, match="sample_rate"):
        loaded_session.infer(np.zeros((2, 8), np.float32))


def test_obtain_hashes_the_complete_download(tmp_path, monkeypatch):
    """A partial final write must not be hashed as if it were the whole file.

    obtain() writes through a buffered handle and then calls verify(), which
    re-opens the path and hashes what is on disk. Without an explicit flush,
    anything still in the write buffer is simply absent from those bytes, so the
    download fails its own checksum against a truncated file.

    The trigger is size-dependent, which is why it survived: shutil.copyfileobj
    writes 64 KiB at a time and writes that large pass straight through the
    buffer, so a payload whose size happens to align leaves no residue. This
    fixture ends with a small partial write on purpose.
    """
    import hashlib
    import io

    from bs_mamba2_infer import checkpoint as checkpoint_module

    payload = bytes(range(256)) * 4096 + b"x" * 100
    digest = hashlib.sha256(payload).hexdigest()

    class _Spec:
        identifier = "regression-probe"
        url = "http://example.invalid/probe.ckpt"
        sha256 = digest
        status = "available"

    monkeypatch.setattr(checkpoint_module, "urlopen", lambda url: io.BytesIO(payload))

    resolved = checkpoint_module.obtain(_Spec(), cache_dir=tmp_path)

    assert resolved.stat().st_size == len(payload)
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == digest
