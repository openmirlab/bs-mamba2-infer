# Changelog

## Unreleased

### Default checkpoint switched to msst-vocals; official-vocals marked unavailable

- **Behavior change, called out deliberately** (allowed by article 7: this
  package has never been published, so its surface owes no compatibility):
  the session/CLI default `checkpoint_id` is now `msst-vocals`. The previous
  default, `official-vocals`, has a dead upstream: its author-linked Google
  Drive URL serves a genuine 0-byte file (verified 2026-07-31 -- the download's
  sha256 is the empty-string hash).
- `official-vocals` is marked `status = "unavailable"` rather than deleted:
  its audited SHA-256 is retained, so a locally obtained copy passed via
  `checkpoint=` still verifies byte-for-byte. `obtain()` refuses the download
  path loudly, naming the cause and the working alternative.
- Re-hosting the weights ourselves is deliberately NOT done: they carry
  `NOASSERTION` (no upstream licence statement), and redistributing unlicensed
  weights is not this package's call. Asking the author for a grant is the
  recorded follow-up.

- Removed Apple MLX/MPS backend support (out of scope per org canon,
  openmirlab-dev art. 4b): `device="mps"` raises `ValueError` again
  (`test_mps_is_not_claimed`), reversing the 2026-07-31 contract lift. The
  `backend=` argument, the `SeparationBackend` dispatch layer, the `mlx/`
  package, and the `[mlx]` packaging extra are removed entirely.

## 0.1.0 - 2026-07-22

- Initial inference-only BSMamba2 vocal separation package.
- Added verified official and MSST v1.0.19 checkpoint metadata, explicit device
  selection, cache integrity checks, a reusable session, one-shot API, and CLI.
- Made the `cuda` compatibility extra explicitly empty because mamba-ssm's
  undeclared Torch build dependency breaks standard isolated installation;
  documented its separate upstream non-isolated accelerator installation.
