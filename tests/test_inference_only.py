# /nav: shipped-boundary regression check.
# Keeps training/evaluation orchestration out of source modules and core deps.
# Reads: src package and pyproject metadata as plain text.
from __future__ import annotations

from pathlib import Path


def test_shipped_runtime_has_no_training_framework_surface() -> None:
    root = Path(__file__).parents[1]
    text = "\n".join(path.read_text(encoding="utf-8") for path in (root / "src" / "bs_mamba2_infer").rglob("*.py"))
    for forbidden in ("pytorch_lightning", "hydra", "wandb", "tensorboard", "training_step", "validation_step"):
        assert forbidden not in text
