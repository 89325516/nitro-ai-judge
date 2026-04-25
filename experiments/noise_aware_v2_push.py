from __future__ import annotations

import argparse
from pathlib import Path

try:
    from experiments.noise_aware_v2_process import generate
except ModuleNotFoundError:
    from noise_aware_v2_process import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate noise-aware V2 directional candidates.")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--train", type=Path, required=True)
    gen.add_argument("--test", type=Path, required=True)
    gen.add_argument("--anchor-output", type=Path, required=True)
    gen.add_argument("--best-output", type=Path, required=True)
    gen.add_argument("--candidate-start", type=int, required=True)
    gen.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate(Path.cwd(), args)


if __name__ == "__main__":
    main()
