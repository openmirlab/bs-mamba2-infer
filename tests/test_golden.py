# /nav: opt-in real-checkpoint public-API regression gate.
# Compares every output sample against an untouched upstream reconstruction on
# the same fixture. It skips in ordinary offline CI because neither weights nor
# CUDA are bundled. Reads: tests/golden metadata and the public session API.
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from bs_mamba2_infer import BSMamba2Session


@pytest.mark.golden
def test_official_checkpoint_matches_upstream_reconstruction() -> None:
    root = os.environ.get("BS_MAMBA2_GOLDEN_ROOT")
    if not root or not torch.cuda.is_available():
        pytest.skip("set BS_MAMBA2_GOLDEN_ROOT and provide CUDA plus the optional mamba_ssm accelerator")
    fixture = json.loads((Path(__file__).parent / "golden" / "official_fixture.json").read_text())
    root_path = Path(root)
    mixture = root_path / "listening" / "00_mixture.wav"
    checkpoint = root_path / "weights" / "official-bsmamba2" / "vocals" / "2025-03-14_05-40" / "weights" / "sota_model.ckpt"
    upstream = Path(os.environ["BS_MAMBA2_UPSTREAM_OUTPUT"])
    # The fixture's checkpoint IS "official-vocals" (see tests/golden/official_fixture.json's
    # checkpoint_sha256). checkpoint_id defaults to "msst-vocals" -- omitting it here would make
    # obtain() verify this file's bytes against the WRONG spec's sha256 and fail before inference
    # ever runs, silently turning a passing golden test into a checksum-mismatch failure. Must be
    # explicit.
    with BSMamba2Session(checkpoint_id="official-vocals", checkpoint=checkpoint, device="cuda", batch_size=1) as session:
        actual = session.infer(mixture).vocals.T
    expected, rate = sf.read(upstream, dtype="float32", always_2d=True)
    difference = actual - expected
    assert rate == fixture["sample_rate"]
    assert actual.shape == expected.shape == (fixture["samples"], fixture["channels"])
    assert float(np.max(np.abs(difference))) <= fixture["max_abs_tolerance"]
    assert float(np.sqrt(np.mean(np.square(difference)))) <= fixture["rms_tolerance"]
