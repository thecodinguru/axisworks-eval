#!/usr/bin/env python
"""Fetch the Cadsy CAD technical drawings sample from Hugging Face.

    python scripts/fetch_cadsy_dataset.py                 # all 65 parts -> datasets/cadsy
    python scripts/fetch_cadsy_dataset.py --parts 10      # first ten parts
    python scripts/fetch_cadsy_dataset.py --with-step     # also the STEP models

Plain urllib over ``resolve/main/<path>``; no Hugging Face client needed. The
output directory is gitignored; the committed four-part subset lives in
``fixtures/cadsy``. Dataset license: Apache-2.0 (LICENSE is fetched too).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = "cadsy/cad-technical-drawings"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
ROOT = Path(__file__).resolve().parents[1]


def _fetch(path: str, dest: Path, *, force: bool = False) -> bool:
    if dest.exists() and not force:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "axisworks-eval/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "cadsy")
    ap.add_argument("--parts", type=int, default=0, help="limit to the first N parts (0 = all)")
    ap.add_argument("--with-step", action="store_true", help="also download model.step per part")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.jsonl", "LICENSE", "README.md"):
        _fetch(name, args.out / name, force=args.force)
    lines = (args.out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if args.parts:
        rows = rows[: args.parts]

    fetched = skipped = 0
    for i, row in enumerate(rows, 1):
        keys = ["asme_image", "asme_labels", "iso_image", "iso_labels"]
        if args.with_step:
            keys.append("model_step")
        for key in keys:
            rel = row[key]
            try:
                if _fetch(rel, args.out / rel, force=args.force):
                    fetched += 1
                else:
                    skipped += 1
            except Exception as exc:
                print(f"  {rel}: {exc}", file=sys.stderr)
        print(f"[{i}/{len(rows)}] {row['part_id']}", flush=True)

    print(f"done: {fetched} files fetched, {skipped} already present -> {args.out}")
    print(f"evaluate with: python scripts/run_eval.py --root {args.out / 'samples'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
