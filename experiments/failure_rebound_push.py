from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.failure_rebound_process import generate
except ModuleNotFoundError:
    from failure_rebound_process import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate failure-rebound candidates after Candidate 129 underperforms.")
    sub = parser.add_subparsers(dest="mode", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--train", type=Path, required=True)
    gen.add_argument("--test", type=Path, required=True)
    gen.add_argument("--best-output", type=Path, required=True)
    gen.add_argument("--failed-output", type=Path, required=True)
    gen.add_argument("--anchor-output", type=Path, required=True)
    gen.add_argument("--high-risk-base-output", type=Path, required=True)
    gen.add_argument("--candidate-start", type=int, default=130)
    gen.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(Path(__file__).resolve().parents[1], args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
