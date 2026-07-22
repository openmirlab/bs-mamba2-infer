# BSMamba2 inference implementation plan

## Ratified scope

`bs-mamba2-infer` is a standalone, inference-only package for the BSMamba2
graph published by EuiYeonKim/BSMamba2.  It supports only compatible BSMamba2
vocal checkpoints.  It permanently excludes Mamba v1, BS-RoFormer,
TS-BSmamba2, training, evaluation, dataset code, Lightning, Hydra, W&B, and
TensorBoard.

## Grounding evidence

- Upstream reference (read-only): `../.dev-cache/bs-mamba-bandit-probe/BSMamba2`
  at `42eb8c84a1bf0d388a994b4c29edd0d9d6a0a2b5`.
- Its runnable inference chain is `src/separator.py` plus the BSMamba2 branch
  of `src/model/bs_model.py`, STFT transforms, and checkpoint state extraction
  in `src/utils/utils_inference.py`.  Training-oriented modules are not ported.
- Golden input: `../.dev-cache/bs-mamba-bandit-probe/listening/00_mixture.wav`.
  Upstream result: `../.dev-cache/bs-mamba-bandit-probe/weights/official-bsmamba2/vocals/2025-03-14_05-40/probe-sparse-entry/case/infer.wav`.
- Official checkpoint/config: `weights/official-bsmamba2/.../sota_model.ckpt`
  and sibling `tb_logs/hparams.yaml`; community MSST candidate/config:
  `weights/bs_mamba2_vocals.ckpt` and `weights/config_bs_mamba2_vocals.yaml`.

## Stages and gates

1. Record source/license/author/paper/checkpoint provenance and capture an
   untouched-upstream golden output plus environment metadata.  Commit only
   evidence and package skeleton.
2. Port the narrow graph, STFT/iSTFT and overlap-add path.  The core path uses
   a pure-PyTorch Mamba2/SSD implementation; an installed `mamba_ssm` may be
   selected as an optional CUDA acceleration, never a core dependency.
3. Add checkpoint registry, SHA-256 downloader/cache resolver, explicit device
   handling, package-qualified session lifecycle, one-shot facade and CLI.
4. Add offline contracts, lifecycle/cache/download/config/CLI/package tests and
   the real fixture regression.  Test official and compatible MSST checkpoints
   through the public API, then record every-sample A/B evidence.  If a
   bit-exact result cannot be reached, record the measured tolerance and its
   hardware/kernel cause; do not claim completion without it.
5. Build from sdist, clean-venv import smoke, inference-only scan, and full CI
   matrix.  Keep source, code, lifecycle, and verification changes in separate
   commits.

## Current risk

The upstream relies on compiled `mamba_ssm` kernels.  Its pure PyTorch fallback
must match its non-fused algorithm before it can be considered a replacement;
the full official checkpoint may require the optional CUDA accelerator for
practical runtime.  Numerical status remains unproven until stage 4.
