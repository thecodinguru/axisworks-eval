"""Scores on real drawings: reads against CAD truth and against the text layer, by class.

The held-out slice of the open-hardware test set (``holdout.py``) has two kinds
of machine truth and no hand labels:

- **CAD truth** (``dxf_truth.py``): the dimensions a DXF states. A read is a
  *value match* when it states every number the printed dimension states
  (nominal and tolerance; ``full``), or at least its first number
  (``nominal``). Each printed dimension is matched to at most one read, in
  order, so three printed "10"s need three reads of 10. This gives how many
  of the printed dimensions were read right: the plan's "dimensional value
  match". It cannot give precision: a note or a hole table is not a
  dimension entity, so a read of one is neither right nor wrong here.
- **Text layer** (``text_layer_metrics.py``): every number a read states is
  printed near its box (``supported near``), the precision proxy; and the
  share of the text's toleranced and diameter callouts that were read.

Both are reported per drawing category (mechanical, PCB fabrication and
assembly, DXF-native) and per drawing class (clean, and the same sheet
scanned by ``degrade.py``), and never as "precision and recall".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from axisworks_eval.dxf_truth import CadDimension, to_details, truth_for_sheet
from axisworks_eval.text_layer_metrics import (
    Read,
    TextLayerScore,
    numbers,
    read_words,
    reads_to_details,
    score_sheet,
)

CLASSES = ("clean", "scanned")


@dataclass
class ValueScore:
    #: Printed dimensions with at least one number.
    truth: int = 0
    #: Dimension reads that state at least one number.
    reads: int = 0
    #: Printed dimensions a read states in full (every number).
    matched_full: int = 0
    #: Printed dimensions whose first number some read states first.
    matched_nominal: int = 0

    def add(self, other: ValueScore) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self)}
        out["value_match_full_rate"] = round(self.matched_full / self.truth, 3) if self.truth else 0.0
        out["value_match_nominal_rate"] = (
            round(self.matched_nominal / self.truth, 3) if self.truth else 0.0
        )
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValueScore:
        return cls(**{f.name: data[f.name] for f in fields(cls)})


def _stated(read: Read) -> list[float]:
    return numbers(read.text)


def score_values(
    truth: Sequence[CadDimension], reads: Iterable[Read], *, misses: int = 0
) -> tuple[ValueScore, list[str]]:
    """One sheet's dimension reads against its CAD truth; also up to ``misses`` unread examples."""
    candidates = [(_stated(r), r) for r in reads if r.kind == "dimension"]
    candidates = [(values, r) for values, r in candidates if values]
    score = ValueScore(reads=len(candidates))
    free_full = [True] * len(candidates)
    free_nominal = [True] * len(candidates)
    unread: list[str] = []
    for dim in truth:
        wanted = [round(v, 6) for v in dim.numbers]
        if not wanted:
            continue
        score.truth += 1
        full = next(
            (
                i
                for i, (values, _r) in enumerate(candidates)
                if free_full[i] and set(wanted) <= set(values)
            ),
            None,
        )
        if full is not None:
            free_full[full] = False
            score.matched_full += 1
        nominal = next(
            (
                i
                for i, (values, _r) in enumerate(candidates)
                if free_nominal[i] and values[0] == wanted[0]
            ),
            None,
        )
        if nominal is not None:
            free_nominal[nominal] = False
            score.matched_nominal += 1
        elif len(unread) < misses:
            unread.append(dim.text)
    return score, unread


@dataclass
class ClassScore:
    """One (category, drawing class) cell of the holdout table."""

    sheets: int = 0
    failed: int = 0
    values: ValueScore | None = None
    text: TextLayerScore | None = None

    def add_sheet(self, entry: dict[str, Any]) -> None:
        self.sheets += 1
        if entry.get("failed"):
            self.failed += 1
            return
        if "values" in entry:
            self.values = self.values or ValueScore()
            self.values.add(ValueScore.from_dict(entry["values"]))
        if "text" in entry:
            self.text = self.text or TextLayerScore()
            self.text.add(TextLayerScore.from_dict(entry["text"]))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sheets": self.sheets, "failed": self.failed}
        if self.values is not None:
            out["values"] = self.values.to_dict()
        if self.text is not None:
            out["text"] = self.text.to_dict()
        return out


def summarize(sheets: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """The holdout table from the per-sheet entries a report lists.

    An entry: ``{"sheet_id", "category", "class", "split", "failed"?, "values"?, "text"?}``.
    Returns ``by_class[drawing_class]`` → ``by_category[category]`` and ``overall``, plus
    an ``overall`` over the clean class only (the number a page shows first).
    """
    table: dict[str, dict[str, ClassScore]] = {}
    for entry in sheets:
        row = table.setdefault(entry.get("class", "clean"), {})
        row.setdefault(entry["category"], ClassScore()).add_sheet(entry)
        row.setdefault("overall", ClassScore()).add_sheet(entry)
    by_class = {
        drawing_class: {
            "by_category": {
                c: s.to_dict() for c, s in sorted(cells.items()) if c != "overall"
            },
            "overall": cells.get("overall", ClassScore()).to_dict(),
        }
        for drawing_class, cells in sorted(table.items())
    }
    return {"by_class": by_class, "overall": (by_class.get("clean") or {}).get("overall", {})}


def headline_lines(block: dict[str, Any]) -> dict[str, Any]:
    """What the report's headline shows for the holdout: value match and the text proxy, per class."""
    out: dict[str, Any] = {}
    for drawing_class, cells in (block.get("by_class") or {}).items():
        overall = cells.get("overall") or {}
        values = overall.get("values") or {}
        text = overall.get("text") or {}
        if values:
            out[f"holdout_values_matched_{drawing_class}"] = values.get("value_match_full_rate")
        if text:
            out[f"holdout_reads_supported_near_{drawing_class}"] = text.get("supported_near_rate")
            out[f"holdout_callouts_read_{drawing_class}"] = text.get("callouts_read_rate")
    return out


def score_entry(
    root: str | Path, row: dict[str, Any], reads: Sequence[Read], *, misses: int = 3
) -> dict[str, Any]:
    """What a report lists for one read of a sheet, and what a checker rebuilds from the vectors.

    ``row`` names the sheet's vector (``file`` or ``file_vector``) and its DXF
    exports (``alt_formats``). A PDF sheet gets its text-layer score; a sheet
    with CAD truth gets the truth it was scored against and its value score.
    """
    base = Path(root)
    vector = str(row.get("file") or row.get("file_vector") or "")
    out: dict[str, Any] = {"reads": reads_to_details(reads)}
    if vector.lower().endswith(".pdf"):
        words = read_words(base / vector)
        out["words"] = len(words)
        out["text"] = score_sheet(words, reads).to_dict()
    truth, source = truth_for_sheet(
        base, {"file_vector": vector, "alt_formats": row.get("alt_formats") or []}
    )
    if truth:
        score, unread = score_values(truth, reads, misses=misses)
        out["cad_truth"] = {"source": source, "dimensions": to_details(truth)}
        out["values"] = score.to_dict()
        if unread:
            out["unread_examples"] = unread
    return out
