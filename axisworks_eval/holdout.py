"""The held-out slice of the open-hardware test set, sealed before any tuning.

Cadsy stopped being a holdout on 17 October 2026: the frame-echo rule and the
revision matcher were fitted to its errors. The plan's gates are measured on
drawings nobody tuned against, so the vector sheets of ``drawing-testset-v1``
(public repositories, pinned commits, open licenses) are split once, here, and
the split is committed as ``holdout_v1.jsonl`` beside the manifest with its
SHA-256 in the README.

The split is by **publisher**, not by sheet: a publisher's drawings share a
title block, a template and habits, so a sheet-level split would let a rule
fitted on one Framework sheet be scored on the next. The dev slice is a fixed,
named list of publishers (3D-printer and small-robot parts, with the three
Voron DXFs that carry CAD dimension entities), so both sides keep CAD truth
and the holdout keeps the aerospace, enclosure and PCB publishers. Nothing is
random, so there is no seed to lose; ``check`` recomputes the split from the
manifest and refuses a file that differs.

Only the three classes a dimension benchmark can score are split:
``mechanical_drawing``, ``pcb_fab_assembly_drawing`` and ``cad_dxf``.
Schematics, block diagrams and the seven outline images carry no dimensions
in the sense of a characteristic table and stay out. Sheets of one revision
group, and the two formats of one drawing (``same_drawing_as``), always land
on the same side.

Rule for every change to prompts, extractor, matcher or rules: tune on Cadsy
and the ``dev`` slice; read the ``holdout`` slice only through the report run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

VERSION = "holdout_v1"
FILE_NAME = f"{VERSION}.jsonl"
CLASSES = ("mechanical_drawing", "pcb_fab_assembly_drawing", "cad_dxf")
#: Publishers whose sheets are the development slice; everyone else is held out.
DEV_PUBLISHERS = (
    "Voron Design",
    "Shenzhen Creality 3D Technology",
    "Prusa Research",
    "phenoptix / MeArm",
    "NASA Jet Propulsion Laboratory",
)
#: What a holdout row keeps from the manifest, in this order.
FIELDS = (
    "sheet_id",
    "split",
    "category",
    "domain",
    "publisher",
    "source_id",
    "license_spdx",
    "file_png",
    "file_vector",
    "alt_formats",
    "page",
    "same_drawing_as",
    "revision_group",
    "cad_educational_watermark",
    "proprietary_boilerplate_in_title_block",
    "text_layer_words",
    "dxf_dimension_entities",
    "sha256_png",
    "source_file_sha256",
)


def load_manifest(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root) / "manifest.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def split(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The holdout rows: every vector sheet of the three classes, with its side.

    A sheet's side is its publisher's; a revision group or a same-drawing pair
    that would straddle the two sides goes to the holdout whole, so no drawing
    is ever tuned on and scored.
    """
    kept = [r for r in rows if r.get("category") in CLASSES and r.get("file_vector")]
    side = {
        r["sheet_id"]: "dev" if r.get("publisher") in DEV_PUBLISHERS else "holdout" for r in kept
    }
    by_id = {r["sheet_id"]: r for r in kept}
    groups: dict[str, set[str]] = {}
    for r in kept:
        if r.get("revision_group"):
            groups.setdefault(f"group:{r['revision_group']}", set()).add(r["sheet_id"])
        twins = r.get("same_drawing_as") or []
        for twin in [twins] if isinstance(twins, str) else twins:
            if twin in by_id:
                groups.setdefault(f"twin:{min(r['sheet_id'], twin)}", set()).update(
                    {r["sheet_id"], twin}
                )
    for members in groups.values():
        if len({side[m] for m in members}) > 1:
            for m in members:
                side[m] = "holdout"
    out = []
    for r in sorted(kept, key=lambda r: r["sheet_id"]):
        row = {name: r.get(name) for name in FIELDS}
        row["split"] = side[r["sheet_id"]]
        row["alt_formats"] = [
            a["file"] if isinstance(a, dict) else a for a in (r.get("alt_formats") or [])
        ]
        out.append(row)
    return out


def dumps(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n" for row in rows)


def load(root: str | Path, name: str = FILE_NAME) -> list[dict[str, Any]]:
    path = Path(root) / name
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Counts a README states: sheets per side, per class, with CAD truth and text."""
    out: dict[str, Any] = {}
    for row in rows:
        block = out.setdefault(row["split"], {"sheets": 0, "publishers": set(), "by_class": {}})
        block["sheets"] += 1
        block["publishers"].add(row["publisher"])
        cls = block["by_class"].setdefault(
            row["category"], {"sheets": 0, "cad_truth": 0, "text_layer": 0}
        )
        cls["sheets"] += 1
        if (row.get("dxf_dimension_entities") or 0) > 0 or any(
            str(a).lower().endswith(".dxf") for a in row.get("alt_formats") or []
        ):
            cls["cad_truth"] += 1
        if (row.get("text_layer_words") or 0) > 0:
            cls["text_layer"] += 1
    for block in out.values():
        block["publishers"] = len(block["publishers"])
    return out


def check(root: str | Path, name: str = FILE_NAME) -> list[str]:
    """The committed split against one recomputed from the manifest; problems, if any."""
    expected = dumps(split(load_manifest(root)))
    path = Path(root) / name
    if not path.is_file():
        return [f"{name} is not under {root}"]
    actual = path.read_text(encoding="utf-8")
    if actual == expected:
        return []
    problems = []
    before = {json.loads(line)["sheet_id"]: line for line in actual.splitlines() if line.strip()}
    after = {json.loads(line)["sheet_id"]: line for line in expected.splitlines() if line.strip()}
    for sheet_id in sorted(set(before) | set(after)):
        if before.get(sheet_id) != after.get(sheet_id):
            problems.append(f"{name}: {sheet_id} differs from the manifest's split")
    return problems or [f"{name}: differs from the manifest's split"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="the drawing-testset-v1 folder")
    ap.add_argument("--write", action="store_true", help=f"write {FILE_NAME} beside the manifest")
    args = ap.parse_args(argv)
    rows = split(load_manifest(args.root))
    path = args.root / FILE_NAME
    if args.write:
        path.write_text(dumps(rows), encoding="utf-8")
        print(f"wrote {path}")
    for side, block in summary(rows).items():
        print(f"{side}: {block['sheets']} sheets from {block['publishers']} publishers")
        for cls, counts in sorted(block["by_class"].items()):
            print(
                f"  {cls:26} {counts['sheets']:4} sheets, CAD truth {counts['cad_truth']:3}, "
                f"text layer {counts['text_layer']:3}"
            )
    if path.is_file():
        problems = check(args.root)
        print(f"sha256 {sha256_of(path)}  {FILE_NAME}" + ("" if not problems else "  (DIFFERS)"))
        for p in problems[:10]:
            print(" ", p)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
