# /nav: CLI adapter contract.
# Verifies parsed CLI values reach the public one-shot API; model work remains
# mocked so this is independent of weights and devices. Reads: cli.py.
from __future__ import annotations

import bs_mamba2_infer.cli as cli


def test_cli_forwards_public_options(monkeypatch: object) -> None:
    received: dict[str, object] = {}
    monkeypatch.setattr(cli, "separate", lambda audio, **kwargs: received.update(audio=audio, **kwargs))
    assert cli.main(["mix.wav", "vocals.wav", "--device", "cpu", "--checkpoint-id", "msst-vocals"]) == 0
    assert received == {"audio": "mix.wav", "output_path": "vocals.wav", "device": "cpu", "checkpoint_id": "msst-vocals", "checkpoint": None, "cache_dir": None, "batch_size": 4}
