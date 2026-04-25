from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.noise_aware_process import generate
except ModuleNotFoundError:
    from noise_aware_process import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate noise-aware denoising candidates after Candidate 159 improves.")
    sub = parser.add_subparsers(dest="mode", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--train", type=Path, required=True)
    gen.add_argument("--test", type=Path, required=True)
    gen.add_argument("--best-output", type=Path, required=True)
    gen.add_argument("--candidate-start", type=int, default=160)
    gen.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(Path(__file__).resolve().parents[1], args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
