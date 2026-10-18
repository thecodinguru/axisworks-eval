#!/usr/bin/env python
"""Check an AxisWorks accuracy report.

    python scripts/check_report.py reports/accuracy_2026-10-18.json
    python scripts/check_report.py reports/accuracy_2026-10-18.json --labels datasets/cadsy/samples
    python scripts/check_report.py <report> --testset fixtures/drawing-testset-v1
    python scripts/check_report.py <report> --holdout fixtures/drawing-testset-v1

The first form recomputes every derived number from the details the report
carries. With --labels (the Cadsy part folders, from
scripts/fetch_cadsy_dataset.py) the truth behind each result is rebuilt from
the labels too. With --testset every score of the open-hardware test set is
rebuilt from its PDFs and the reads the report lists. With --holdout the
sealed split is checked against the manifest and every held-out score is
rebuilt from the sheet's PDF or DXF and the listed reads. Exits 1 if anything
does not check out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axisworks_eval.verify import (  # noqa: E402
    check_holdout,
    check_report,
    check_text_layer,
    check_truth,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("report", type=Path)
    ap.add_argument("--labels", type=Path, help="directory of Cadsy part_* folders")
    ap.add_argument("--testset", type=Path, help="the drawing-testset-v1 folder (its PDFs)")
    ap.add_argument(
        "--holdout", type=Path, help="the drawing-testset-v1 folder (holdout_v1.jsonl, PDFs, DXFs)"
    )
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
    if args.testset:
        rebuilt = check_text_layer(report, args.testset)
        print(
            f"  test set from its PDFs: {'matches' if not rebuilt else f'{len(rebuilt)} scores differ'}"
        )
        problems += rebuilt
    if args.holdout:
        held = check_holdout(report, args.holdout)
        print(f"  holdout from its vectors: {'matches' if not held else f'{len(held)} differ'}")
        problems += held
    for problem in problems:
        print(f"    {problem}")
    print("  headline:")
    for name, value in (report.get("headline") or {}).items():
        print(f"    {name}: {value}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
