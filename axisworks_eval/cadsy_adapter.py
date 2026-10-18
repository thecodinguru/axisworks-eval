"""Cadsy labels -> plain characteristic rows and box lists.

Cadsy annotation types: ``label`` (zone letters/numbers), ``dimension``,
``callout``, ``feature_control_frame`` (``"symbol|tol|A|B"``),
``datum_definition``, ``title_block_value``, ``title_block_label`` and
``projection_*``. Only the first five carry engineering content. ISO
profiles use a decimal comma and ``×``; the adapter normalizes both so the
ASME/ISO twins compare on meaning.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Cadsy annotation type -> AxisWorks char_type (None = not a characteristic).
CHAR_TYPE_BY_ANNOTATION: dict[str, str | None] = {
    "dimension": "dimension",
    "callout": "dimension",
    "feature_control_frame": "gdt",
    "datum_definition": "gdt",
    "title_block_value": None,
    "title_block_label": None,
    "label": None,
    "projection_first_angle": None,
    "projection_third_angle": None,
}

#: Annotation types whose text is a real drawing string worth an OCR benchmark.
OCR_TYPES = frozenset({"dimension", "callout", "feature_control_frame", "title_block_value"})

_NUM = re.compile(r"\d+(?:\.\d+)?")
_MULT = re.compile(r"^\s*(\d+)\s*X\s*(.*)$", re.IGNORECASE)
_TOL_SYM = re.compile(r"±\s*(\d+(?:\.\d+)?)")
_TOL_BIL = re.compile(r"\+\s*(\d+(?:\.\d+)?)\s*/?\s*-\s*(\d+(?:\.\d+)?)")
_DEPTH = re.compile(r"(?:↧|DEEP|DP)\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*DEEP", re.IGNORECASE)
_MC = re.compile(r"\((M|L|S)\)|Ⓜ|Ⓛ")


@dataclass(frozen=True)
class CadsyAnnotation:
    type: str
    view: str
    text: str
    box: tuple[float, float, float, float]

    @property
    def bbox_dict(self) -> dict[str, float]:
        x0, y0, x1, y1 = self.box
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


@dataclass
class ExpectedCharacteristic:
    """A label-derived row with every attribute the matcher/differ read."""

    id: str
    char_type: str
    raw_text: str
    normalized_text: str
    bbox: dict[str, float]
    view_name: str = ""
    source_type: str = ""
    # dimension fields
    dimension_type: str = ""
    nominal: float | None = None
    nominal_text: str = ""
    upper_tol: float | None = None
    lower_tol: float | None = None
    tolerance_type: str = ""
    units: str = "mm"
    is_basic: bool = False
    is_reference: bool = False
    quantity: int | None = None
    depth: float | None = None
    # gdt fields
    gdt_symbol: str = ""
    material_condition: str = ""
    datum_refs: list[str] = field(default_factory=list)
    zone_shape: str = ""
    # note-family fields (unused here, present so getattr() is uniform)
    spec_text: str = ""
    spec_source: str = ""
    spec_category: str = ""
    spec_codes: list[str] = field(default_factory=list)
    #: None, as a database row without a printed balloon (Cadsy prints none).
    balloon_number: str | None = None
    feature_tag: str = ""
    # canonical mirrors (Cadsy drawings are millimetre)
    nominal_mm: float | None = None
    upper_tol_mm: float | None = None
    lower_tol_mm: float | None = None
    units_canonical: str = "mm"
    stable_id: str | None = None


def load_labels(path: str | Path) -> list[CadsyAnnotation]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[CadsyAnnotation] = []
    for a in data.get("annotations", []):
        box = a.get("box") or []
        if len(box) != 4:
            continue
        out.append(
            CadsyAnnotation(
                type=str(a.get("type", "")),
                view=str(a.get("view", "")),
                text=str(a.get("text", "")),
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
        )
    return out


def normalize_text(text: str) -> str:
    """ASME/ISO-neutral form: decimal point, ``X`` for ``×``, single spaces, upper.

    The diameter sign is written three ways: ⌀ (the drafting symbol), Ø and ∅
    (the empty-set sign, which the extraction prompts themselves use).
    """
    t = re.sub(r"(?<=\d),(?=\d)", ".", text)
    t = t.replace("×", "X").replace("Ø", "⌀").replace("∅", "⌀")
    t = re.sub(r"\s+", " ", t).strip().upper()
    # "4X 7" and "4X7" are the same callout.
    t = re.sub(r"(\d)\s*X\s*", r"\1X ", t)
    # So are "C 0.5" and "C0.5", "R 5" and "R5": a symbol and its value, spaced or not.
    t = re.sub(r"(^|[\s(])(S⌀|⌀|SR|R|C|M)\s+(?=\d)", r"\1\2", t)
    return t.strip()


def parse_dimension(text: str) -> dict[str, Any]:
    """Fields for a dimension or callout string. Conservative: unparsed -> None."""
    t = normalize_text(text)
    out: dict[str, Any] = {
        "normalized_text": t,
        "dimension_type": "LINEAR",
        "nominal": None,
        "nominal_text": "",
        "upper_tol": None,
        "lower_tol": None,
        "tolerance_type": "",
        "quantity": None,
        "depth": None,
    }
    rest = t
    m = _MULT.match(rest)
    if m:
        out["quantity"] = int(m.group(1))
        rest = m.group(2)
    if rest.startswith("⌀"):
        out["dimension_type"] = "DIAMETER"
    elif rest.startswith("S⌀"):
        out["dimension_type"] = "SPHERICAL_DIAMETER"
    elif re.match(r"^SR\d", rest):
        out["dimension_type"] = "SPHERICAL_RADIUS"
    elif re.match(r"^R\d", rest):
        out["dimension_type"] = "RADIUS"
    elif re.match(r"^C\d", rest):
        out["dimension_type"] = "CHAMFER"
    elif "°" in rest:
        out["dimension_type"] = "ANGULAR"
    body = re.sub(r"^(S⌀|⌀|SR|R|C)", "", rest)
    num = _NUM.search(body)
    if num:
        out["nominal"] = float(num.group(0))
        out["nominal_text"] = num.group(0)
    sym = _TOL_SYM.search(rest)
    bil = _TOL_BIL.search(rest)
    if sym:
        v = float(sym.group(1))
        out.update(upper_tol=v, lower_tol=-v, tolerance_type="SYMMETRIC")
    elif bil:
        out.update(
            upper_tol=float(bil.group(1)),
            lower_tol=-float(bil.group(2)),
            tolerance_type="BILATERAL_UNEQUAL",
        )
    d = _DEPTH.search(rest)
    if d:
        out["depth"] = float(d.group(1) or d.group(2))
    return out


def parse_fcf(text: str) -> dict[str, Any]:
    """``cylindricity|0,05|A|B(M)`` -> symbol, tolerance, datums, modifier."""
    parts = [p.strip() for p in normalize_text(text).split("|") if p.strip()]
    out: dict[str, Any] = {
        "gdt_symbol": parts[0].lower() if parts else "",
        "nominal": None,
        "nominal_text": "",
        "datum_refs": [],
        "material_condition": "",
        "zone_shape": "",
    }
    if len(parts) > 1:
        tol = parts[1]
        if tol.startswith("⌀"):
            out["zone_shape"] = "cylindrical"
        num = _NUM.search(tol)
        if num:
            out["nominal"] = float(num.group(0))
            out["nominal_text"] = num.group(0)
        mc = _MC.search(tol)
        if mc:
            out["material_condition"] = {"M": "MMC", "L": "LMC", "S": "RFS"}.get(
                mc.group(1) or ("M" if "Ⓜ" in tol else "L"), ""
            )
    for seg in parts[2:]:
        letters = re.match(r"^([A-Z](?:-[A-Z])?)", seg)
        if letters:
            out["datum_refs"].append(letters.group(1))
    return out


_SLOT = re.compile(r"^\s*SLOT\s+(\d+(?:[.,]\d+)?)\s*[X×x]\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)


def split_annotation(a: CadsyAnnotation) -> list[CadsyAnnotation]:
    """One annotation per inspected requirement.

    The two profiles of a part label the same requirements at different
    granularity: frames stacked under one leader are one label on one sheet
    and separate labels on the other, and a slot is "SLOT 10 X 17.5" on one
    and two dimensions on the other. Each stacked frame and each slot size is
    its own characteristic on a FAIR, so both become one row each. A stacked
    frame's box is split into equal bands, top frame first.
    """
    if a.type == "feature_control_frame" and "\n" in a.text:
        lines = [line.strip() for line in a.text.splitlines() if line.strip()]
        x0, y0, x1, y1 = a.box
        band = (y1 - y0) / len(lines)
        return [
            CadsyAnnotation(a.type, a.view, line, (x0, y0 + i * band, x1, y0 + (i + 1) * band))
            for i, line in enumerate(lines)
        ]
    if a.type in ("dimension", "callout"):
        m = _SLOT.match(a.text)
        if m:
            return [CadsyAnnotation("dimension", a.view, size, a.box) for size in m.groups()]
    return [a]


def annotations_to_rows(
    annotations: list[CadsyAnnotation], *, prefix: str
) -> list[ExpectedCharacteristic]:
    """Engineering annotations -> rows, deterministically ordered and identified.

    Order is by view, then top-to-bottom, left-to-right, so the same drawing
    always yields the same ids (``<prefix>-<n>``).
    """
    keep = [
        part
        for a in annotations
        if CHAR_TYPE_BY_ANNOTATION.get(a.type)
        for part in split_annotation(a)
    ]
    keep.sort(key=lambda a: (a.view, round(a.box[1], 4), round(a.box[0], 4), a.text))
    rows: list[ExpectedCharacteristic] = []
    for i, a in enumerate(keep):
        char_type = CHAR_TYPE_BY_ANNOTATION[a.type] or ""
        row = ExpectedCharacteristic(
            id=f"{prefix}-{i}",
            char_type=char_type,
            raw_text=a.text,
            normalized_text=normalize_text(a.text),
            bbox=a.bbox_dict,
            view_name=a.view,
            source_type=a.type,
        )
        if a.type in ("dimension", "callout"):
            parsed = parse_dimension(a.text)
            for k, v in parsed.items():
                setattr(row, k, v)
            row.nominal_mm = row.nominal
            row.upper_tol_mm = row.upper_tol
            row.lower_tol_mm = row.lower_tol
        elif a.type == "feature_control_frame":
            parsed = parse_fcf(a.text)
            for k, v in parsed.items():
                setattr(row, k, v)
            row.nominal_mm = row.nominal
        elif a.type == "datum_definition":
            row.gdt_symbol = "datum_feature"
            letters = re.match(r"^\s*([A-Z])", a.text.upper())
            row.datum_refs = [letters.group(1)] if letters else []
        rows.append(row)
    return rows


def load_rows(
    labels_path: str | Path, *, prefix: str | None = None
) -> list[ExpectedCharacteristic]:
    path = Path(labels_path)
    return annotations_to_rows(load_labels(path), prefix=prefix or path.stem)


def expected_boxes(
    annotations: list[CadsyAnnotation],
) -> list[tuple[str, tuple[float, float, float, float], str]]:
    """``(char_type, box, text)`` for every engineering annotation."""
    return [
        (CHAR_TYPE_BY_ANNOTATION[a.type] or "", a.box, a.text)
        for a in annotations
        if CHAR_TYPE_BY_ANNOTATION.get(a.type)
    ]


def iter_fixture_parts(root: str | Path) -> list[tuple[str, Path, Path, Path, Path]]:
    """``(part_id, asme.png, asme_labels.json, iso.png, iso_labels.json)`` per part dir."""
    base = Path(root)
    out = []
    if not base.is_dir():
        return out
    for part_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("part_")):
        files = [
            part_dir / n for n in ("asme.png", "asme_labels.json", "iso.png", "iso_labels.json")
        ]
        if all(f.exists() for f in files):
            out.append((part_dir.name, *files))
    return out
