# Changelog

## 0.1.0 - 2026-07-22

- Initial inference-only BSMamba2 vocal separation package.
- Added verified official and MSST v1.0.19 checkpoint metadata, explicit device
  selection, cache integrity checks, a reusable session, one-shot API, and CLI.
- Made the `cuda` compatibility extra explicitly empty because mamba-ssm's
  undeclared Torch build dependency breaks standard isolated installation;
  documented its separate upstream non-isolated accelerator installation.
