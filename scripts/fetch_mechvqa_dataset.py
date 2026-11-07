#!/usr/bin/env python
"""Fetch the public MechVQA benchmark (ICML 2026, arXiv 2605.30794) from GitHub.

    python scripts/fetch_mechvqa_dataset.py                    # all 1,185 records -> datasets/mechvqa
    python scripts/fetch_mechvqa_dataset.py --limit 50         # the first 50 records and their images
    python scripts/fetch_mechvqa_dataset.py --check fixtures/mechvqa   # compare with the committed manifest

Plain urllib over ``raw.githubusercontent.com/<repo>/<commit>/<path>``, pinned to
one commit so the set cannot drift under the manifest. The benchmark JSONL
comes first, then every image it names (paths are relative to
``benchmark_data/``), then the licence, README, citation, the repo's own
evaluator and the inference prompt; a 404 on one of those extras is reported,
not fatal. Every file on disk afterwards is hashed into ``<out>/FETCHED.json``.

Licence facts, as found on 7 November 2026: the repository's code is
Apache-2.0 (its LICENSE file); the Hugging Face dataset card declares
Apache-2.0; the GitHub README is silent on the data's licence; the paper
(Appendix A) defers to the redistribution terms of the source materials the
drawings came from. The drawings therefore cannot be relicensed by us: the
images are fetched for evaluation only and never committed. The output
directory is gitignored (``datasets/``); what the repository commits is the
manifest of hashes under ``fixtures/mechvqa``.

With ``--check <fixture_dir>`` the fetched set is rebuilt into manifest rows
and compared with the committed manifest through
``axisworks_eval.mechvqa_metrics.check``; the import is lazy so this
script works before that module lands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO = "xiaofengShi/MechVQA"
COMMIT = "c8d61513aef05b4f5d117bb0c6d1a055451d6f33"
BASE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = "benchmark_data/vqa_benchmark/mechvqa_benchmark.jsonl"
#: Image paths inside the records are relative to this directory.
IMAGE_DIR = "benchmark_data"
#: Fetched beside the data so the licence, the citation and the scoring code
#: we replicate are on disk at the same pin; none of them is needed to score.
OPTIONAL_FILES = (
    "LICENSE",
    "README.md",
    "CITATION.cff",
    "evaluation/mechvqa_eval/evaluate_vqa.py",
    "prompts/mech_r1.jinja",
)
FETCHED_NAME = "FETCHED.json"


def _fetch(path: str, dest: Path, *, force: bool = False) -> bool:
    if dest.exists() and not force:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "axisworks-eval/1.0"})
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
    # A partial download must not look like a finished one on the next run.
    tmp.replace(dest)
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_paths(jsonl: Path, limit: int) -> list[str]:
    rows: list[dict] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    seen: dict[str, None] = {}
    for row in rows:
        for image in row.get("images", []):
            seen[f"{IMAGE_DIR}/{image}"] = None
    print(f"{len(rows)} records, {len(seen)} distinct images")
    return list(seen)


def _fetch_many(paths: list[str], out: Path, *, force: bool) -> tuple[int, int, list[str]]:
    """Fetch each path, continuing past errors; the failures come back for a retry."""
    fetched = skipped = 0
    failed: list[str] = []
    for i, rel in enumerate(paths, 1):
        try:
            if _fetch(rel, out / rel, force=force):
                fetched += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"  {rel}: {exc}", file=sys.stderr)
            failed.append(rel)
        if i % 50 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] images", flush=True)
    return fetched, skipped, failed


def _write_fetched(out: Path, paths: list[str]) -> dict:
    files = {rel: _sha256(out / rel) for rel in paths if (out / rel).is_file()}
    record = {
        "commit": COMMIT,
        "repo": REPO,
        "files": files,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / FETCHED_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _check(out: Path, fixture_dir: Path) -> int:
    sys.path.insert(0, str(ROOT))
    try:
        from axisworks_eval import mechvqa_metrics
    except ImportError as exc:
        print(
            f"--check needs axisworks_eval.mechvqa_metrics, which is not importable: {exc}"
        )
        return 1
    problems = mechvqa_metrics.check(out, fixture_dir)
    if problems:
        print(f"the fetched set does not match the manifest in {fixture_dir}:")
        for problem in problems[:20]:
            print(f"  {problem}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        return 1
    print(f"the fetched set matches the manifest in {fixture_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "mechvqa")
    ap.add_argument(
        "--limit", type=int, default=0, help="fetch images for the first N records only (0 = all)"
    )
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument(
        "--check",
        type=Path,
        metavar="FIXTURE_DIR",
        help="after fetching, compare the set with the committed manifest in this directory",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out / BENCHMARK_PATH
    try:
        _fetch(BENCHMARK_PATH, jsonl, force=args.force)
    except Exception as exc:
        print(f"could not fetch {BENCHMARK_PATH}: {exc}", file=sys.stderr)
        return 1

    missing: list[str] = []
    for rel in OPTIONAL_FILES:
        try:
            _fetch(rel, args.out / rel, force=args.force)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(f"  {rel}: not at this commit (404), skipped")
                missing.append(rel)
            else:
                print(f"  {rel}: {exc}", file=sys.stderr)
                missing.append(rel)
        except Exception as exc:
            print(f"  {rel}: {exc}", file=sys.stderr)
            missing.append(rel)

    images = _image_paths(jsonl, args.limit)
    fetched, skipped, failed = _fetch_many(images, args.out, force=args.force)
    if failed:
        print(f"retrying {len(failed)} failed images once")
        again, _, failed = _fetch_many(failed, args.out, force=True)
        fetched += again

    present = sum(1 for rel in images if (args.out / rel).is_file())
    record = _write_fetched(args.out, [BENCHMARK_PATH, *OPTIONAL_FILES, *images])
    print(
        f"done: {fetched} files fetched, {skipped} already present, "
        f"{present}/{len(images)} images on disk, {len(record['files'])} hashed in {FETCHED_NAME} -> {args.out}"
    )
    if missing:
        print(f"optional files not fetched: {', '.join(missing)}")
    if failed:
        print(f"images still missing after retry: {len(failed)}", file=sys.stderr)
        for rel in failed[:20]:
            print(f"  {rel}", file=sys.stderr)

    status = 1 if failed else 0
    if args.check is not None:
        status = max(status, _check(args.out, args.check))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
