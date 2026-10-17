"""The "scanned" drawing class: a clean sheet degraded the way a scan degrades it.

Deterministic, so a report can be regenerated exactly: the same bytes in give
the same bytes out. Three steps, in the order a scanner applies them:

- resample to 200 dpi: the Cadsy PNGs state no resolution, so they are taken
  as 300 dpi originals and scaled to two thirds (2480 x 1754 becomes
  1653 x 1169), the loss of a 300 dpi sheet scanned at 200 dpi;
- rotate 0.4 degrees, a sheet fed slightly askew, the corners filled white;
- encode as JPEG at quality 60, the compression a scan-to-email leaves, and
  decode back to PNG, which is what an upload of the scan would carry.

Boxes stay valid as fractions of the sheet: the scale keeps them, and the
rotation moves a corner of the scaled sheet about 7 pixels (0.4 degrees over a
half-diagonal of 1012 pixels).
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass

from PIL import Image


@dataclass(frozen=True)
class ScanProfile:
    source_dpi: int = 300
    target_dpi: int = 200
    rotation_deg: float = 0.4
    jpeg_quality: int = 60

    @property
    def scale(self) -> float:
        return self.target_dpi / self.source_dpi

    def to_dict(self) -> dict:
        return {**asdict(self), "scale": round(self.scale, 3)}


SCANNED = ScanProfile()


def degrade(png_bytes: bytes, profile: ScanProfile = SCANNED) -> bytes:
    """The sheet as a scan of it would arrive: smaller, askew, JPEG-compressed, as PNG."""
    with Image.open(io.BytesIO(png_bytes)) as source:
        sheet = source.convert("RGBA")
    # Transparent areas are paper.
    paper = Image.new("RGBA", sheet.size, (255, 255, 255, 255))
    sheet = Image.alpha_composite(paper, sheet).convert("RGB")
    width = max(1, round(sheet.width * profile.scale))
    height = max(1, round(sheet.height * profile.scale))
    sheet = sheet.resize((width, height), Image.Resampling.LANCZOS)
    sheet = sheet.rotate(
        profile.rotation_deg,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )
    jpeg = io.BytesIO()
    sheet.save(jpeg, format="JPEG", quality=profile.jpeg_quality, optimize=False, progressive=False)
    jpeg.seek(0)
    with Image.open(jpeg) as scanned:
        out = io.BytesIO()
        scanned.convert("RGB").save(out, format="PNG", optimize=False)
    return out.getvalue()
