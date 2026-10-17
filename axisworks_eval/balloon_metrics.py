"""The arithmetic of the balloon-number scores, with nothing from the product in it.

Numbering a sheet and carrying numbers across a revision stay in the product
(``balloon_scorer`` runs them); everything here can be checked without them:
the truth pairs on twin sheets, the seeded revision's moved view, how one
numbering is scored against the truth and how the scores are pooled.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from axisworks_eval.cadsy_adapter import ExpectedCharacteristic

METHODS = ("none", "stable_id", "pairs")
SCENARIOS = ("twins", "seeded")


@dataclass
class MethodScore:
    truth: int = 0
    kept: int = 0
    misplaced: int = 0
    added: int | None = None
    fresh: int | None = None
    carried: int | None = None
    flagged: int | None = None
    misplaced_unflagged: int | None = None

    @property
    def kept_fraction(self) -> float:
        return self.kept / self.truth if self.truth else 0.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "truth": self.truth,
            "kept": self.kept,
            "kept_fraction": round(self.kept_fraction, 3),
            "misplaced": self.misplaced,
        }
        if self.added is not None:
            out["added"] = self.added
            out["fresh"] = self.fresh or 0
            out["fresh_fraction"] = round((self.fresh or 0) / self.added, 3) if self.added else 0.0
        if self.flagged is not None:
            out["carried"] = self.carried or 0
            out["flagged"] = self.flagged
            out["misplaced_unflagged"] = self.misplaced_unflagged or 0
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MethodScore:
        return cls(
            truth=data["truth"],
            kept=data["kept"],
            misplaced=data["misplaced"],
            added=data.get("added"),
            fresh=data.get("fresh"),
            carried=data.get("carried"),
            flagged=data.get("flagged"),
            misplaced_unflagged=data.get("misplaced_unflagged"),
        )


def score_numbers(
    old_numbers: Mapping[str, int],
    new_numbers: Mapping[str, int],
    truth: Mapping[str, str],
    *,
    added: Iterable[str] | None = None,
    flagged: Iterable[str] | None = None,
) -> MethodScore:
    """Score one numbering of the new sheet.

    ``truth`` maps a new row id to the old row id of the same feature. Rows in
    ``truth`` and ``added`` are judged; any other row of the new sheet is not,
    because nothing says which feature it is. ``flagged`` names the new rows a
    reviewer would be asked to check.
    """
    owner = {number: rid for rid, number in old_numbers.items()}
    added_ids = list(added) if added is not None else None
    score = MethodScore(truth=len(truth))
    for new_id, old_id in truth.items():
        number = new_numbers.get(new_id)
        if number is not None and number == old_numbers.get(old_id):
            score.kept += 1
    misplaced: list[str] = []
    for new_id in [*truth, *(added_ids or [])]:
        holder = owner.get(new_numbers.get(new_id))  # type: ignore[arg-type]
        if holder is not None and holder != truth.get(new_id):
            misplaced.append(new_id)
    score.misplaced = len(misplaced)
    if added_ids is not None:
        score.added = len(added_ids)
        score.fresh = sum(1 for rid in added_ids if new_numbers.get(rid) not in owner)
    if flagged is not None:
        flags = set(flagged)
        carried = {rid for rid, number in new_numbers.items() if number in owner}
        score.carried = len(carried)
        score.flagged = len(carried & flags)
        score.misplaced_unflagged = sum(1 for rid in misplaced if rid not in flags)
    return score


@dataclass
class BalloonResult:
    part_id: str
    scenario: str
    old_rows: int
    new_rows: int
    methods: dict[str, MethodScore] = field(default_factory=dict)
    moved_view: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "part_id": self.part_id,
            "scenario": self.scenario,
            "old_rows": self.old_rows,
            "new_rows": self.new_rows,
            "methods": {m: s.to_dict() for m, s in self.methods.items()},
        }
        if self.moved_view:
            out["moved_view"] = self.moved_view
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BalloonResult:
        return cls(
            part_id=data["part_id"],
            scenario=data["scenario"],
            old_rows=data["old_rows"],
            new_rows=data["new_rows"],
            methods={m: MethodScore.from_dict(s) for m, s in data["methods"].items()},
            moved_view=data.get("moved_view", ""),
        )


def twin_truth(
    old_rows: Sequence[ExpectedCharacteristic], new_rows: Sequence[ExpectedCharacteristic]
) -> dict[str, str]:
    """New id -> old id for rows whose (type, view, normalized text) is unique on both sheets."""

    def key(r: ExpectedCharacteristic) -> tuple[str, str, str]:
        return (r.char_type, r.view_name, r.normalized_text)

    old_count = Counter(key(r) for r in old_rows)
    new_count = Counter(key(r) for r in new_rows)
    old_by_key = {key(r): r.id for r in old_rows if old_count[key(r)] == 1}
    return {
        r.id: old_by_key[key(r)]
        for r in new_rows
        if new_count[key(r)] == 1 and key(r) in old_by_key
    }


def move_top_view(
    rows: Sequence[ExpectedCharacteristic],
) -> tuple[list[ExpectedCharacteristic], str]:
    """Move the view holding the top-most row below the others.

    Every other box is squeezed into the top half of the sheet and that view's
    boxes into the bottom half, so reading order changes and content does not.
    """
    if not rows:
        return [], ""
    top = min(rows, key=lambda r: (r.bbox["y0"], r.bbox["x0"]))
    view = top.view_name
    out: list[ExpectedCharacteristic] = []
    for r in rows:
        moved = copy.deepcopy(r)
        shift = 0.5 if r.view_name == view else 0.0
        moved.bbox = {
            "x0": r.bbox["x0"],
            "x1": r.bbox["x1"],
            "y0": shift + r.bbox["y0"] * 0.5,
            "y1": shift + r.bbox["y1"] * 0.5,
        }
        out.append(moved)
    return out, view


def summarize(results: Sequence[BalloonResult]) -> dict[str, dict[str, Any]]:
    """Pooled scores per method: kept over every truth pair, misplaced summed."""
    out: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        total = MethodScore()
        for r in results:
            s = r.methods.get(method)
            if s is None:
                continue
            total.truth += s.truth
            total.kept += s.kept
            total.misplaced += s.misplaced
            if s.added is not None:
                total.added = (total.added or 0) + s.added
                total.fresh = (total.fresh or 0) + (s.fresh or 0)
            if s.flagged is not None:
                total.carried = (total.carried or 0) + (s.carried or 0)
                total.flagged = (total.flagged or 0) + s.flagged
                total.misplaced_unflagged = (total.misplaced_unflagged or 0) + (
                    s.misplaced_unflagged or 0
                )
        out[method] = total.to_dict()
    return out
