# /nav: checkpoint registry, resolver, and integrity boundary.
# Reads the package-owned TOML once, resolves explicit/manual/cache locations by
# one shared function, and streams only verified files into the cache.  It owns
# no model code and never materializes a checkpoint while answering cache_info.
# Reads: config/checkpoints.toml; session.py calls the resolver and loader.
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
from pathlib import Path
import os
import pickle
import shutil
import tempfile
try:  # Python 3.10 has the compatible backport in core dependencies.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib
from typing import Mapping
from urllib.request import urlopen

import torch


@dataclass(frozen=True)
class CheckpointSpec:
    identifier: str
    architecture: str
    config: str
    url: str
    sha256: str
    size: int
    license: str
    provenance: str
    source_revision: str
    updated: str


class _IgnoredOmegaConf:
    """Compatibility sink for Lightning metadata never used by inference."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __setstate__(self, state: object) -> None:
        self.__dict__["state"] = state


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        # The official Lightning checkpoint pickles hparams as OmegaConf. The
        # inference loader needs only state_dict, so don't pull training config
        # machinery into the core installation merely to discard that metadata.
        if module.startswith("omegaconf"):
            return _IgnoredOmegaConf
        return super().find_class(module, name)


class _CheckpointPickleModule:
    Unpickler = _CheckpointUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)


def load_payload(path: str | Path) -> object:
    """Load a trusted checkpoint without importing upstream training deps."""
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_CheckpointPickleModule)


def _parse_specs(text: str) -> Mapping[str, CheckpointSpec]:
    try:
        raw = tomllib.loads(text)
        entries = raw["checkpoints"]
    except (KeyError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError("The packaged checkpoint registry is malformed.") from exc
    required = {"architecture", "config", "url", "sha256", "size", "license", "provenance", "source_revision", "updated"}
    result: dict[str, CheckpointSpec] = {}
    for identifier, entry in entries.items():
        if not isinstance(entry, dict) or required - entry.keys() or len(str(entry.get("sha256", ""))) != 64:
            raise RuntimeError(f"Malformed checkpoint entry: {identifier!r}.")
        result[identifier] = CheckpointSpec(identifier=identifier, **entry)
    return result


def checkpoint_specs() -> Mapping[str, CheckpointSpec]:
    path = resources.files("bs_mamba2_infer.config").joinpath("checkpoints.toml")
    try:
        return _parse_specs(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError("The packaged checkpoint registry is malformed.") from exc


def default_cache_dir() -> Path:
    override = os.environ.get("BS_MAMBA2_INFER_CACHE")
    return Path(override).expanduser() if override else Path.home() / ".cache" / "bs-mamba2-infer"


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def resolved_path(spec: CheckpointSpec, *, cache_dir: str | Path | None = None, checkpoint: str | Path | None = None) -> Path | None:
    """Answer where loading would find a file, without downloading it."""
    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
        return path if path.is_file() else None
    candidate = (Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()) / f"{spec.identifier}.ckpt"
    return candidate if candidate.is_file() else None


def verify(path: Path, spec: CheckpointSpec) -> Path:
    actual = digest(path)
    if actual != spec.sha256:
        raise RuntimeError(f"Checkpoint checksum mismatch for {spec.identifier}: expected {spec.sha256}, got {actual}.")
    return path


def obtain(spec: CheckpointSpec, *, cache_dir: str | Path | None = None, checkpoint: str | Path | None = None) -> Path:
    found = resolved_path(spec, cache_dir=cache_dir, checkpoint=checkpoint)
    if found is not None:
        return verify(found, spec)
    if checkpoint is not None:
        raise FileNotFoundError(f"Checkpoint does not exist: {Path(checkpoint).expanduser()}")
    if not spec.url:
        raise RuntimeError(f"{spec.identifier} has no verified download URL; provide checkpoint=PATH manually.")
    destination_dir = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{spec.identifier}.ckpt"
    with tempfile.NamedTemporaryFile(dir=destination_dir, delete=False, suffix=".part") as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urlopen(spec.url) as response:
                shutil.copyfileobj(response, temporary)
            verify(temporary_path, spec)
            temporary_path.replace(destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    return destination
