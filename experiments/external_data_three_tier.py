from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.three_tier_external_emit import probe_report
    from experiments.three_tier_external_process import generate
except ModuleNotFoundError:
    from three_tier_external_emit import probe_report
    from three_tier_external_process import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fused three-tier external data candidates.")
    sub = parser.add_subparsers(dest="mode", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--train", type=Path, required=True)
    probe.add_argument("--test", type=Path, required=True)
    probe.add_argument("--report", type=Path)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--train", type=Path, required=True)
    generate_parser.add_argument("--test", type=Path, required=True)
    generate_parser.add_argument("--base-output", type=Path, required=True)
    generate_parser.add_argument("--high-risk-base-output", type=Path, required=True)
    generate_parser.add_argument("--candidate-start", type=int, default=46)
    generate_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = probe_report(args.train, args.test, args.report) if args.mode == "probe" else generate(root, args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
