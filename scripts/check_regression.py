"""CI regression gate: compare a smoke run's mean baseline_score (from a
run directory's scores.jsonl) against a committed reference value.

One-sided with tolerance: fails ONLY when the mean drops more than
`tolerance` below the reference — improvements always pass (then refresh
the reference with --update-baseline). The smoke run is deterministic
(StubSystem on the frozen mini corpus), so any drop is a real change in
scoring/runner behavior, not noise.

Usage:
  python scripts/check_regression.py --scores RUN_DIR/scores.jsonl \
      --baseline tests/backtest/fixtures/smoke_baseline.json
  python scripts/check_regression.py --scores ... --baseline ... \
      --update-baseline --tolerance 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def mean_baseline_score(scores_path: Path) -> float:
    vals = []
    for line in Path(scores_path).read_text().splitlines():
        if not line.strip():
            continue
        v = json.loads(line).get("baseline_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    if not vals:
        raise SystemExit(f"no baseline_score values found in {scores_path}")
    return sum(vals) / len(vals)


def check(scores_path: Path, baseline_path: Path) -> tuple[bool, str]:
    ref = json.loads(Path(baseline_path).read_text())
    mean = mean_baseline_score(scores_path)
    floor = ref["mean_baseline_score"] - ref["tolerance"]
    ok = mean >= floor
    msg = (f"mean baseline_score={mean:.4f} "
           f"reference={ref['mean_baseline_score']:.4f} "
           f"tolerance={ref['tolerance']} floor={floor:.4f} "
           f"-> {'OK' if ok else 'REGRESSION'}")
    return ok, msg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, type=Path,
                    help="path to a run's scores.jsonl")
    ap.add_argument("--baseline", required=True, type=Path,
                    help="committed reference JSON "
                         "{mean_baseline_score, tolerance}")
    ap.add_argument("--update-baseline", action="store_true",
                    help="write the current mean as the new reference")
    ap.add_argument("--tolerance", type=float, default=1.0,
                    help="tolerance written on --update-baseline")
    args = ap.parse_args(argv)

    if args.update_baseline:
        mean = mean_baseline_score(args.scores)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(
            {"mean_baseline_score": mean, "tolerance": args.tolerance},
            indent=2) + "\n")
        print(f"baseline updated: mean_baseline_score={mean:.6f} "
              f"tolerance={args.tolerance}")
        return 0

    ok, msg = check(args.scores, args.baseline)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
