"""CAD truth: the dimensions a DXF states, read from its DIMENSION entities.

A DXF exported from CAD keeps each dimension as an entity, and (in every file
of the test set) the text the CAD tool printed for it in the dimension's
geometry block: "240 MAX.", "(150.0)", "125" with "0.15" beside it, "⌀ 5".
That text is exact ground truth for what the sheet shows, with no hand
labelling and no box to draw: a read of the sheet is right when it states the
same numbers.

Two things it is not. It is not every number on the sheet: notes, hole
tables and title blocks are plain text and are not dimension entities, so a
read of them is neither right nor wrong here, and precision cannot be judged
against this truth, only how many of the printed dimensions were read. And it
has no coordinates in the picture: the DXF's model space maps to the rendered
PNG through the renderer's fit, which this module does not reproduce; values
are matched by number, not by place.

When a dimension has no geometry block (a file saved without rendering) the
printed text is computed from the measurement the way CAD would print it:
scaled by ``dimlfac``, rounded to ``dimdec`` places, the ``<>`` of the text
override filled in, tolerances or limits from the style. Such a dimension is
marked ``computed`` so a reader knows the truth was derived, not read.

Product-free: ezdxf and the standard library.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axisworks_eval.text_layer_metrics import numbers

#: ezdxf's dimtype values.
KINDS = {
    0: "linear",
    1: "aligned",
    2: "angular",
    3: "diameter",
    4: "radius",
    5: "angular",
    6: "ordinate",
}
#: MTEXT codes: a stacked fraction keeps its two parts, parametrised codes go, toggles go.
_STACKED = re.compile(r"\\S([^;^/#]*)[\^/#]([^;]*);")
_PARAMETRISED = re.compile(r"\\[fFHWQACTp][^;]*;")
_TOGGLES = re.compile(r"\\[LlOoKk]")
_BREAKS = re.compile(r"\\[P~]")
_SPACE_SYMBOL = {"%%c": "⌀", "%%C": "⌀", "%%d": "°", "%%D": "°", "%%p": "±", "%%P": "±"}


@dataclass(frozen=True)
class CadDimension:
    kind: str
    #: What the sheet prints for this dimension, formatting codes removed.
    text: str
    numbers: tuple[float, ...]
    measurement: float | None
    #: "block": read from the rendered geometry; "computed": derived from the measurement.
    source: str
    space: str = "model"

    def to_details(self) -> list[Any]:
        return [self.kind, self.text, list(self.numbers), self.source]

    @classmethod
    def from_details(cls, row: Sequence[Any]) -> CadDimension:
        kind, text, values, source = row
        return cls(str(kind), str(text), tuple(float(v) for v in values), None, str(source))


def plain(text: str) -> str:
    """MTEXT formatting codes and %% symbols out; stacked text keeps its parts."""
    out = text or ""
    for code, symbol in _SPACE_SYMBOL.items():
        out = out.replace(code, symbol)
    out = _STACKED.sub(lambda m: f"{m.group(1)} {m.group(2)}", out)
    out = _PARAMETRISED.sub("", out)
    out = _BREAKS.sub(" ", out)
    out = _TOGGLES.sub("", out)
    out = out.replace("{", "").replace("}", "").replace("^", " ")
    return re.sub(r"\s+", " ", out).strip()


def _open(path: str | Path):
    import ezdxf
    from ezdxf import recover

    try:
        return ezdxf.readfile(str(path))
    except (OSError, ezdxf.DXFStructureError, UnicodeDecodeError):
        document, _auditor = recover.readfile(str(path))
        return document


def _block_text(dimension: Any) -> str:
    try:
        block = dimension.get_geometry_block()
    except Exception:  # noqa: BLE001 - a broken block reference reads as no block
        return ""
    if block is None:
        return ""
    parts = []
    for entity in block:
        if entity.dxftype() in ("MTEXT", "TEXT", "ATTRIB"):
            try:
                parts.append(entity.plain_text())
            except Exception:  # noqa: BLE001
                parts.append(str(getattr(entity.dxf, "text", "")))
    return plain(" ".join(p for p in parts if p))


def _number_text(value: float, places: int) -> str:
    text = f"{value:.{max(0, places)}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _computed_text(dimension: Any, kind: str) -> tuple[str, float | None]:
    try:
        measurement = float(dimension.get_measurement())
    except Exception:  # noqa: BLE001
        return "", None
    override = dimension.override()
    factor = override.get("dimlfac") or 1.0
    places = override.get("dimdec")
    places = 2 if places is None else int(places)
    value = measurement if kind == "angular" else measurement * factor
    printed = _number_text(value, places)
    template = plain(str(dimension.dxf.text or ""))
    if template and template != " ":
        text = template.replace("<>", printed) if "<>" in template else f"{template} {printed}"
    else:
        text = printed
    if kind == "diameter" and "⌀" not in text:
        text = f"⌀{text}"
    elif kind == "radius" and not text.upper().startswith("R"):
        text = f"R{text}"
    elif kind == "angular":
        text = f"{text}°"
    upper, lower = float(override.get("dimtp") or 0), float(override.get("dimtm") or 0)
    if override.get("dimlim"):
        text = f"{_number_text(value + upper, places)} {_number_text(value - lower, places)}"
    elif override.get("dimtol"):
        if upper == lower:
            text = f"{text} ±{_number_text(upper, places)}"
        else:
            text = f"{text} +{_number_text(upper, places)} -{_number_text(lower, places)}"
    return text, measurement


def read_dimensions(path: str | Path, *, spaces: Iterable[str] = ("model",)) -> list[CadDimension]:
    """Every DIMENSION of the layouts named (``model`` and/or paper-space layout names)."""
    document = _open(path)
    wanted = set(spaces)
    layouts = []
    if "model" in wanted:
        layouts.append(("model", document.modelspace()))
    for layout in document.layouts:
        if layout.name != "Model" and (layout.name in wanted or "paper" in wanted):
            layouts.append((layout.name, layout))
    out = []
    for space, layout in layouts:
        for dimension in layout.query("DIMENSION"):
            kind = KINDS.get(int(getattr(dimension, "dimtype", 0)), "linear")
            text = _block_text(dimension)
            measurement: float | None
            if text:
                source = "block"
                try:
                    measurement = float(dimension.get_measurement())
                except Exception:  # noqa: BLE001
                    measurement = None
            else:
                source = "computed"
                text, measurement = _computed_text(dimension, kind)
            if not text:
                continue
            out.append(
                CadDimension(kind, text, tuple(numbers(text)), measurement, source, space)
            )
    return out


def truth_for_sheet(root: str | Path, row: dict[str, Any]) -> tuple[list[CadDimension], str]:
    """The CAD truth a holdout row has, and where it came from: the sheet's own DXF
    (``dxf``), the DXF exports attached to a PDF sheet (``alt_dxf``), or none (``""``)."""
    base = Path(root)
    vector = str(row.get("file_vector") or "")
    if vector.lower().endswith(".dxf"):
        return read_dimensions(base / vector), "dxf"
    alternates = [
        str(a) for a in row.get("alt_formats") or [] if str(a).lower().endswith(".dxf")
    ]
    if alternates:
        dims: list[CadDimension] = []
        for name in alternates:
            dims += read_dimensions(base / name)
        return dims, "alt_dxf"
    return [], ""


def to_details(dims: Iterable[CadDimension]) -> list[list[Any]]:
    return [d.to_details() for d in dims]


def from_details(rows: Iterable[Sequence[Any]]) -> list[CadDimension]:
    return [CadDimension.from_details(r) for r in rows]
