# /nav: command-line adapter for the public facade.
# Keeps argument parsing outside the API and maps every option directly to the
# one-shot function.  Reads: api.py only.
from __future__ import annotations

import argparse

from .api import separate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract vocals with a compatible BSMamba2 checkpoint.")
    parser.add_argument("input", help="44.1 kHz mono or stereo audio file")
    parser.add_argument("output", help="destination WAV file")
    parser.add_argument("--checkpoint-id", default="msst-vocals")
    parser.add_argument("--checkpoint", help="manual checkpoint path (including offline MSST checkpoint)")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N (backend=mlx only accepts auto/mps)")
    parser.add_argument("--backend", default=None, choices=["torch", "mlx", "auto"], help="torch (default), mlx (Apple Silicon, needs the [mlx] extra), or auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    separate(args.input, checkpoint_id=args.checkpoint_id, checkpoint=args.checkpoint, cache_dir=args.cache_dir, device=args.device, backend=args.backend, batch_size=args.batch_size, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
