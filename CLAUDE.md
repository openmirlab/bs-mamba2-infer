# bs-mamba2-infer maintainer guide

This is a standalone, inference-only port of Euiyeon Kim and Yong-Hoon Choi's
BSMamba2 vocal separator, based on `EuiYeonKim/BSMamba2` revision
`42eb8c84a1bf0d388a994b4c29edd0d9d6a0a2b5` (MIT). The associated paper is
“Mamba2 Meets Silence: Robust Vocal Source Separation for Sparse Regions,”
arXiv:2508.14556. The maintained scope is only the BSMamba2/Mamba2 graph and
vocal inference; Mamba v1, BS-RoFormer, TS-BSmamba2, training, datasets,
evaluation, Lightning, Hydra, W&B, TensorBoard, and bundled weights do not ship.

## Layout and conventions

- `src/bs_mamba2_infer/model.py` owns the graph and pure-PyTorch fallback.
- `audio.py` owns waveform normalization, STFT/iSTFT, and overlap-add geometry.
- `checkpoint.py` owns the TOML registry, resolver, cache, download, and hash.
- `api.py` owns the public lifecycle; `load()` is load-once, `infer()` is
  ready-only, `release()` permits reload, and `close()` is terminal/idempotent.
- Every load-bearing source file has a nav header. Do not add training/eval
  surfaces or core compiled dependencies.

## Checkpoint facts

`official-vocals` is the author-linked Drive checkpoint; its local audited SHA-256
is `b1fcf93fdd6f7bc79e5410b1330b3d0fd5a6ca6ea23a94730c684629c71cd5ca`.
`msst-vocals` is the MSST v1.0.19 release asset with audited SHA-256
`fceb733b41742abb8df563092f4cb5bbc9243c6c03e7f34dc44c14e60732c72f`.
Both have weight license `NOASSERTION`; code license and checkpoint terms must
stay separate in docs and metadata.

## Verification

```bash
uv run pytest -q
python -m build
python -m venv /tmp/bs-mamba2-wheel-venv
/tmp/bs-mamba2-wheel-venv/bin/pip install dist/*.whl
/tmp/bs-mamba2-wheel-venv/bin/python -c 'import bs_mamba2_infer; print(bs_mamba2_infer.BSMamba2Session)'
rg -n -i 'lightning|hydra|wandb|tensorboard|training_step|validation_step|dataset|evaluate' src pyproject.toml
```

The CUDA golden command uses the checked local author fixture:

```bash
PYTHONPATH=src /path/to/python -c 'from bs_mamba2_infer import BSMamba2Session; from pathlib import Path; r=Path("../.dev-cache/bs-mamba-bandit-probe"); s=BSMamba2Session(checkpoint=r/"weights/official-bsmamba2/vocals/2025-03-14_05-40/weights/sota_model.ckpt", device="cuda").load(); s.infer(r/"listening/00_mixture.wav", output_path="/tmp/bsmamba2-vocals.wav")'
```

Recorded fixture environment: Linux, Python 3.11, torch 2.7.1+cu126, CUDA,
`mamba_ssm` optional accelerator, checkpoint SHA above, 44.1 kHz two-channel
input. The current output differs from the untouched upstream reconstruction by
at most one 16-bit PCM least-significant bit (`6.103515625e-05`), RMS
`1.5819565e-05`; this is a documented tolerance closure, not bit identity.
