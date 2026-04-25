from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FALLBACK_SCORE = 38.388844
PROMOTION_DELTA = 1.0
PROMOTION_SCORE = FALLBACK_SCORE + PROMOTION_DELTA


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_from_report(path: Path) -> dict:
    report = load_report(path)
    score = report.get("mean_score", report.get("score"))
    return {
        "name": path.stem,
        "path": str(path),
        "score": None if score is None else float(score),
        "score_type": report.get("score_type", "unknown"),
        "mode": report.get("mode", "unknown"),
        "row_count": report.get("row_count"),
    }


def compare_candidates(candidates: Sequence[dict]) -> dict:
    scored = [candidate for candidate in candidates if candidate["score"] is not None]
    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
    best = ranked[0] if ranked else None
    promote = bool(best and best["score_type"] == "local_estimate" and best["score"] >= PROMOTION_SCORE)
    return {
        "evidence_class": "local_estimate",
        "fallback_score": FALLBACK_SCORE,
        "promotion_score": PROMOTION_SCORE,
        "promoted": promote,
        "best_candidate": best,
        "candidates": ranked + [candidate for candidate in candidates if candidate["score"] is None],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local candidate reports against the promotion threshold.")
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/candidate_comparison_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_candidates([candidate_from_report(path) for path in args.reports])
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
