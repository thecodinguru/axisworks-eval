"""Reads of a vector drawing checked against the drawing's own text layer.

The open-hardware test set (``fixtures/drawing-testset-v1`` in the public
repository: 604 single-page PDFs from public repositories, each with its
license) has no hand labels for dimensions or GD&T. A vector PDF carries its
text as text, though: every word the CAD tool placed, with its exact box. That
is partial ground truth, and this module scores against it.

- **Supported**: every number a dimension or frame read states appears in the
  text layer within ``MARGIN`` of the read's box (``near``), or anywhere on
  the sheet (``anywhere``). The text is taken in runs (``phrases``), so a
  number exported one character at a time still reads as one. A read that
  states no number cannot be checked and is counted apart.
- **Callouts**: text that can only be a toleranced or diameter dimension: a
  number with its tolerance on one line ("12.5 ±0.1", "⌀6 +0.1/-0"), a number
  with its deviations stacked to its right ("11.5" with "+0.01" over "0.05",
  the minus drawn as a line), or a diameter ("Ø8"); and how many of them a read
  covers: one that states the callout's first number, near it.

Neither is precision or recall. Text drawn as curves (common for GD&T symbols
and some title blocks) has no text layer, so a right read can go unsupported;
a callout is a pattern in the text, not a label (a stacked tolerance set in
rotated text is not found). Sheets with fewer than ``MIN_WORDS`` words are
left out of the rates.

Nothing here comes from the product: pdfplumber and the standard library, so
the public checker recomputes every number from the PDFs and the reads the
report lists.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

#: How far from a read's box (a fraction of the page) its numbers may sit.
MARGIN = 0.02
#: A sheet with fewer words than this has no text layer worth checking against.
MIN_WORDS = 20

Box = tuple[float, float, float, float]

_NUMBER = re.compile(r"\d*[.,]\d+|\d+")
# A nominal, then its tolerance on the same line: "12.5 ±0.1", "6 +0.1/-0".
_TOLERANCE = re.compile(r"\d[^±+]*±\s*\d*[.,]?\d")
_DEVIATIONS = re.compile(r"\d[^±+]*\+\s*\d*[.,]?\d+\s*/?\s*[-−]\s*\d*[.,]?\d")
_DIAMETER = re.compile(r"[⌀Ø∅]\s*\d*[.,]?\d")
_UPPER = re.compile(r"^\+\s*\d*[.,]?\d+$")
_LOWER = re.compile(r"^[-−]?\s*\d*[.,]?\d+$")
_NOMINAL = re.compile(r"^[⌀Ø∅R]?\d*[.,]?\d+$")
_LETTERS = re.compile(r"[A-Za-z]{2,}")
#: Spaces inside a number of a letter-spaced export ("Ø 5 . 8" is Ø5.8).
_SPACED = re.compile(r"(?<=[\d.,])\s+(?=[\d.,])")
_SPACED_POINT = re.compile(r"\d\s+[.,]\s+\d")
#: A general-tolerance line ("X.X ±0.1", ".XX ±.01") is a rule for the sheet, not a dimension.
_PLACEHOLDER = re.compile(r"(?<![A-Za-z])X?\.X+|X\.X")
#: Words a dimension callout may carry besides its numbers.
_CALLOUT_WORDS = frozenset(
    {"THRU", "TYP", "REF", "DEEP", "MAX", "MIN", "SQ", "EQ", "SP", "CSK", "CBORE", "PL", "MM", "IN"}
)


@dataclass(frozen=True)
class Word:
    text: str
    box: Box
    upright: bool = True


@dataclass(frozen=True)
class Read:
    kind: str  # "dimension" | "gdt"
    text: str
    box: Box | None


@dataclass
class TextLayerScore:
    sheets: int = 0
    #: Sheets whose text layer has at least MIN_WORDS words; only these count below.
    sheets_with_text: int = 0
    sheets_with_reads: int = 0
    reads: int = 0
    reads_without_numbers: int = 0
    supported_near: int = 0
    supported_anywhere: int = 0
    callouts: int = 0
    callouts_read: int = 0

    def add(self, other: TextLayerScore) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    @property
    def checkable(self) -> int:
        return self.reads - self.reads_without_numbers

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self)}
        checkable = self.checkable
        out["supported_near_rate"] = round(self.supported_near / checkable, 3) if checkable else 0.0
        out["supported_anywhere_rate"] = (
            round(self.supported_anywhere / checkable, 3) if checkable else 0.0
        )
        out["callouts_read_rate"] = (
            round(self.callouts_read / self.callouts, 3) if self.callouts else 0.0
        )
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextLayerScore:
        return cls(**{f.name: data[f.name] for f in fields(cls)})


def numbers(text: str) -> list[float]:
    """The values a text states: "4X ⌀6.35 ±.05" → [4, 6.35, 0.05]; a decimal comma reads as a point."""
    out = []
    for token in _NUMBER.findall(text or ""):
        try:
            out.append(round(float(token.replace(",", ".")), 6))
        except ValueError:
            continue
    return out


def read_words(pdf: str | Path | bytes, page: int = 1) -> list[Word]:
    """The page's words with boxes on the page (0..1, top-left origin), as the product reads them.

    Same pdfplumber settings as the extraction's text layer; the frame matches
    the rendered page, rotated pages included.
    """
    import io

    import pdfplumber

    source = io.BytesIO(pdf) if isinstance(pdf, bytes) else str(pdf)
    with pdfplumber.open(source) as document:
        sheet = document.pages[page - 1]
        width, height = float(sheet.width or 0), float(sheet.height or 0)
        if width <= 0 or height <= 0:
            return []
        raw = sheet.extract_words(
            extra_attrs=["size", "upright"],
            keep_blank_chars=False,
            use_text_flow=False,
            x_tolerance=1.5,
            y_tolerance=2.0,
        )
    out = []
    for w in raw:
        text = (w.get("text") or "").strip()
        box = (
            _clamp(float(w["x0"]) / width),
            _clamp(float(w["top"]) / height),
            _clamp(float(w["x1"]) / width),
            _clamp(float(w["bottom"]) / height),
        )
        if text and box[2] > box[0] and box[3] > box[1]:
            out.append(Word(text, box, bool(w.get("upright", True))))
    return out


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, round(v, 6)))


def _near(a: Box, b: Box, margin: float) -> bool:
    return (
        a[0] - margin <= b[2]
        and b[0] <= a[2] + margin
        and a[1] - margin <= b[3]
        and b[1] <= a[3] + margin
    )


def phrases(words: Sequence[Word]) -> list[Word]:
    """Words joined into the runs of text they form: same direction, same line, close together.

    A gap up to the line's height joins two words ("12.5" and "±0.1" exported
    as two words); vertical text runs along y.
    """
    out: list[Word] = []
    for upright in (True, False):
        group = [w for w in words if w.upright == upright]
        along = 0 if upright else 1  # the axis text runs along
        group.sort(
            key=lambda w: (round((w.box[1 - along] + w.box[3 - along]) / 2, 3), w.box[along])
        )
        current: list[Word] = []
        for word in group:
            if current and _same_run(current[-1], word, along):
                current.append(word)
                continue
            if current:
                out.append(_join(current))
            current = [word]
        if current:
            out.append(_join(current))
    return out


def _same_run(a: Word, b: Word, along: int) -> bool:
    across = 1 - along
    a_lo, a_hi = a.box[across], a.box[across + 2]
    b_lo, b_hi = b.box[across], b.box[across + 2]
    height = max(a_hi - a_lo, b_hi - b_lo)
    overlap = min(a_hi, b_hi) - max(a_lo, b_lo)
    gap = b.box[along] - a.box[along + 2]
    return overlap >= 0.5 * min(a_hi - a_lo, b_hi - b_lo) and -height <= gap <= height


def _join(run: list[Word]) -> Word:
    text = " ".join(w.text for w in run)
    # Some exports place every character as its own word ("Ø 5 . 8"): a run with a
    # decimal point set apart has its numbers rejoined. Zone labels ("6 5 4") stay apart.
    if _SPACED_POINT.search(text):
        text = _SPACED.sub("", text)
    return Word(
        text,
        (
            min(w.box[0] for w in run),
            min(w.box[1] for w in run),
            max(w.box[2] for w in run),
            max(w.box[3] for w in run),
        ),
        run[0].upright,
    )


def is_callout(text: str) -> bool:
    """A phrase that can only be a toleranced or diameter dimension."""
    if not (_TOLERANCE.search(text) or _DEVIATIONS.search(text) or _DIAMETER.search(text)):
        return False
    if _PLACEHOLDER.search(text):
        return False
    return {w.upper() for w in _LETTERS.findall(text)} <= _CALLOUT_WORDS


def _stacked(words: Sequence[Word]) -> list[Word]:
    """Nominals with their deviations stacked to the right: "11.5" then "+0.01" over "0.05"."""
    upright = [w for w in words if w.upright]
    out = []
    for nominal in (w for w in upright if _NOMINAL.match(w.text)):
        x0, y0, x1, y1 = nominal.box
        h, middle = y1 - y0, (y0 + y1) / 2
        upper = next(
            (
                w
                for w in upright
                if _UPPER.match(w.text)
                and x1 - 0.5 * h <= w.box[0] <= x1 + h
                and y0 - h <= (w.box[1] + w.box[3]) / 2 < middle
            ),
            None,
        )
        if upper is None:
            continue
        lower = next(
            (
                w
                for w in upright
                if w is not upper
                and _LOWER.match(w.text)
                and abs(w.box[0] - upper.box[0]) <= h
                and middle < (w.box[1] + w.box[3]) / 2 <= y1 + h
            ),
            None,
        )
        if lower is not None:
            out.append(_join([nominal, upper, Word("-" + lower.text.lstrip("-−"), lower.box)]))
    return out


def callouts(words: Sequence[Word]) -> list[Word]:
    """The dimension callouts the text layer holds, one line or stacked."""
    return [p for p in phrases(words) if is_callout(p.text)] + _stacked(words)


def score_sheet(
    words: Sequence[Word], reads: Iterable[Read], *, margin: float = MARGIN
) -> TextLayerScore:
    """One sheet's reads against its text layer; a sheet without enough text counts only as a sheet."""
    reads = list(reads)
    score = TextLayerScore(sheets=1)
    if len(words) < MIN_WORDS:
        return score
    score.sheets_with_text = 1
    score.sheets_with_reads = int(bool(reads))
    runs = phrases(words)
    everywhere = {n for run in runs for n in numbers(run.text)}
    for read in reads:
        score.reads += 1
        stated = set(numbers(read.text))
        if not stated:
            score.reads_without_numbers += 1
            continue
        if stated <= everywhere:
            score.supported_anywhere += 1
        if read.box is not None:
            nearby = {
                n for run in runs if _near(read.box, run.box, margin) for n in numbers(run.text)
            }
            if stated <= nearby:
                score.supported_near += 1
    for callout in [r for r in runs if is_callout(r.text)] + _stacked(words):
        score.callouts += 1
        first = numbers(callout.text)[0]
        if any(
            read.box is not None
            and first in numbers(read.text)
            and _near(read.box, callout.box, margin)
            for read in reads
        ):
            score.callouts_read += 1
    return score


def merge(scores: Iterable[TextLayerScore]) -> TextLayerScore:
    total = TextLayerScore()
    for score in scores:
        total.add(score)
    return total


def reads_from_details(rows: Iterable[Sequence[Any]]) -> list[Read]:
    """The reads a report lists for a sheet: ``[kind, text, [x0, y0, x1, y1] | null]``."""
    return [
        Read(str(kind), str(text), tuple(float(v) for v in box) if box else None)  # type: ignore[misc]
        for kind, text, box in rows
    ]


def reads_to_details(reads: Iterable[Read]) -> list[list[Any]]:
    return [
        [r.kind, r.text, [round(v, 4) for v in r.box] if r.box is not None else None] for r in reads
    ]


def summarize(sheets: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Per-category and overall counts from the per-sheet scores a report lists."""
    by_category: dict[str, TextLayerScore] = {}
    for sheet in sheets:
        if sheet.get("failed"):
            continue
        score = TextLayerScore.from_dict(sheet["score"])
        by_category.setdefault(sheet["category"], TextLayerScore()).add(score)
    return {
        "by_category": {c: s.to_dict() for c, s in sorted(by_category.items())},
        "overall": merge(by_category.values()).to_dict(),
    }
