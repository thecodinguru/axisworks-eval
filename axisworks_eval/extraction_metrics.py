"""Characteristic extraction against the Cadsy labels: the arithmetic, nothing from the product.

The truth is a sheet's labelled requirements: dimensions and callouts, and
feature control frames (one row per stacked frame). Datum feature symbols
name a datum and nothing is measured on them, so they are not scored. The
predictions are what the extractor read from the same sheet, written back in
the label format and parsed by the same code as the labels, so a difference
is a difference in reading, not in parsing.

The two sides are paired by the product's own revision matcher
(``diff_revisions``), the rules the twin and perturbation gates use. A pair
that comes out unchanged, or different only in presentation (``Ø`` against
``⌀``, a decimal comma), is a correct read. A pair whose values differ was
found but misread: it counts against precision and recall alike. An unpaired
truth row is a miss, an unpaired prediction a reading of something that is
not there.

The pairing itself is the product's matcher and runs in
``extraction_scorer``; this module holds what can be checked without it: the
truth rows, the reads in the label format, the counts and their pooling.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from axisworks_eval.bbox_scorer import box_of
from axisworks_eval.cadsy_adapter import (
    CadsyAnnotation,
    ExpectedCharacteristic,
    annotations_to_rows,
)

#: Label types that are inspected requirements, and the characteristic type each reads as.
SCORED_LABELS = {"dimension": "dimension", "callout": "dimension", "feature_control_frame": "gdt"}
CHAR_TYPES = ("dimension", "gdt")
_NO_BOX = (0.0, 0.0, 0.0, 0.0)
_DIGIT = re.compile(r"\d")


@dataclass
class TypeExtraction:
    truth: int = 0
    predicted: int = 0
    #: Paired and read right.
    correct: int = 0
    #: Paired at all, right or misread.
    located: int = 0

    def add(self, other: TypeExtraction) -> None:
        self.truth += other.truth
        self.predicted += other.predicted
        self.correct += other.correct
        self.located += other.located

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.truth if self.truth else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TypeExtraction:
        return cls(
            truth=data["truth"],
            predicted=data["predicted"],
            correct=data["correct"],
            located=data["located"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "truth": self.truth,
            "predicted": self.predicted,
            "correct": self.correct,
            "located": self.located,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
            "located_recall": round(self.located / self.truth, 3) if self.truth else 0.0,
        }


@dataclass
class ExtractionScore:
    by_type: dict[str, TypeExtraction] = field(
        default_factory=lambda: {t: TypeExtraction() for t in CHAR_TYPES}
    )
    #: "truth -> read" for pairs whose values differ, a few per sheet.
    misreads: list[str] = field(default_factory=list)

    @property
    def overall(self) -> TypeExtraction:
        total = TypeExtraction()
        for score in self.by_type.values():
            total.add(score)
        return total

    def to_dict(self, examples: int = 10) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "by_type": {t: s.to_dict() for t, s in sorted(self.by_type.items())},
            "misread_examples": self.misreads[:examples],
        }


def frame_text(frame: Any) -> str:
    """A read feature control frame in the label format: ``perpendicularity|⌀0.2(M)|B``."""
    symbol = (getattr(frame, "control_type", "") or "").strip().lower()
    tolerance = (getattr(frame, "tolerance_text", "") or "").strip()
    value = getattr(frame, "tolerance_value", None)
    if not _DIGIT.search(tolerance):
        tolerance = f"{value:g}" if value is not None else ""
    zone = (getattr(frame, "zone_shape", "") or "").strip().upper()
    if zone in {"CYLINDRICAL", "DIAMETER", "DIAMETRAL"} and not tolerance.startswith(
        ("⌀", "Ø", "ø")
    ):
        tolerance = f"⌀{tolerance}"
    modifier = {"MMC": "(M)", "LMC": "(L)"}.get(
        (getattr(frame, "material_condition", "") or "").upper(), ""
    )
    if modifier and not re.search(r"\((M|L)\)|Ⓜ|Ⓛ", tolerance):
        tolerance += modifier
    datums = [
        d.strip().upper() for d in (getattr(frame, "datum_refs", None) or []) if d and d.strip()
    ]
    return "|".join([symbol, tolerance, *datums])


def items_of(section: Any) -> list[Any]:
    """A result's section as a list: ``extract_page`` returns plain lists, a VLM response ``.items``."""
    if section is None:
        return []
    if isinstance(section, list):
        return section
    return list(getattr(section, "items", None) or [])


def predicted_annotations(result: Any) -> list[CadsyAnnotation]:
    """What an extraction result read, as label-format annotations (no view: views are not scored)."""
    out: list[CadsyAnnotation] = []
    for item in items_of(getattr(result, "dimensions", None)):
        text = (
            getattr(item, "raw_dimension_text", "")
            or getattr(item, "normalized_text", "")
            or getattr(item, "nominal_text", "")
            or ""
        ).strip()
        if text:
            out.append(
                CadsyAnnotation(
                    "dimension", "", text, box_of(getattr(item, "bbox", None)) or _NO_BOX
                )
            )
    for frame in items_of(getattr(result, "gdt_frames", None)):
        text = frame_text(frame)
        if text.strip("|"):
            out.append(
                CadsyAnnotation(
                    "feature_control_frame",
                    "",
                    text,
                    box_of(getattr(frame, "bbox", None)) or _NO_BOX,
                )
            )
    return out


def truth_annotations(labels: Iterable[CadsyAnnotation]) -> list[CadsyAnnotation]:
    """A sheet's scored labels, with the view cleared as on the predictions."""
    return [CadsyAnnotation(a.type, "", a.text, a.box) for a in labels if a.type in SCORED_LABELS]


def rows_of(annotations: list[CadsyAnnotation], prefix: str) -> list[ExpectedCharacteristic]:
    """Annotations as the rows the matcher reads, parsed by the same code for truth and reads."""
    return annotations_to_rows(annotations, prefix=prefix)


def merge_extraction(scores: Iterable[ExtractionScore], examples_each: int = 2) -> ExtractionScore:
    merged = ExtractionScore()
    for score in scores:
        for char_type, counts in score.by_type.items():
            merged.by_type.setdefault(char_type, TypeExtraction()).add(counts)
        merged.misreads.extend(score.misreads[:examples_each])
    return merged


def provenance_of_read(result: Any) -> dict[str, int]:
    """How many of a read's dimensions and frames carry a box, and name the stage that placed it.

    The plan's provenance gate asks that every published row says where it
    came from. On the harness side that is the read itself: ``with_box`` reads
    carry a box, ``with_source`` reads carry a box and its ``bbox_source``.
    """
    items = items_of(getattr(result, "dimensions", None)) + items_of(
        getattr(result, "gdt_frames", None)
    )
    boxed = [item for item in items if box_of(getattr(item, "bbox", None))]
    sourced = [item for item in boxed if getattr(item, "bbox_source", "")]
    return {"reads": len(items), "with_box": len(boxed), "with_source": len(sourced)}


def summarize_provenance(sheets: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Per drawing class, the reads with a box and with a box source, pooled over the sheets.

    A sheet entry: ``{"class", "reads", "with_box", "with_source"}``.
    """
    by_class: dict[str, dict[str, int]] = {}
    for sheet in sheets:
        cell = by_class.setdefault(
            str(sheet.get("class", "clean")),
            {"sheets": 0, "reads": 0, "with_box": 0, "with_source": 0},
        )
        cell["sheets"] += 1
        for key in ("reads", "with_box", "with_source"):
            cell[key] += int(sheet.get(key, 0) or 0)
    out: dict[str, Any] = {}
    for drawing_class, cell in sorted(by_class.items()):
        reads = cell["reads"]
        out[drawing_class] = {
            **cell,
            "box_rate": round(cell["with_box"] / reads, 3) if reads else 0.0,
            "source_rate": round(cell["with_source"] / reads, 3) if reads else 0.0,
        }
    return {"by_class": out}
