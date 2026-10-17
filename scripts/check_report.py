#!/usr/bin/env python
"""Check an AxisWorks accuracy report.

    python scripts/check_report.py reports/accuracy_2026-10-17.json
    python scripts/check_report.py reports/accuracy_2026-10-17.json --labels datasets/cadsy/samples

The first form recomputes every derived number from the details the report
carries. With --labels (the Cadsy part folders, from
scripts/fetch_cadsy_dataset.py) the truth behind each result is rebuilt from
the labels too. Exits 1 if anything does not check out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axisworks_eval.verify import check_report, check_truth  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("report", type=Path)
    ap.add_argument("--labels", type=Path, help="directory of Cadsy part_* folders")
    args = ap.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(
        f"{args.report.name}: commit {report.get('commit') or '?'}, generated {report.get('generated_at')}"
    )
    problems = check_report(report)
    print(
        f"  derived numbers: {'all follow from the details' if not problems else f'{len(problems)} do not'}"
    )
    if args.labels:
        truth = check_truth(report, args.labels)
        print(
            f"  truth from the labels: {'matches' if not truth else f'{len(truth)} counts differ'}"
        )
        problems += truth
    for problem in problems:
        print(f"    {problem}")
    print("  headline:")
    for name, value in (report.get("headline") or {}).items():
        print(f"    {name}: {value}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
