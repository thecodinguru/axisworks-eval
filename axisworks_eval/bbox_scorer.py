"""Box agreement between predicted characteristics and Cadsy labels.

Two views of the same question: IoU (how tight the box is) and centre-in-box
(did the box land on the right annotation at all). A VLM box that is 30 %
oversized still fails IoU 0.5 but passes centre-in-box; the text-layer snap
should pass both. Matching is greedy by IoU within the same char_type.

Boxes are normalized to their sheet, so score one sheet at a time and pool the
counts with :func:`merge_scores`; one call over boxes from several sheets would
match a box on one drawing to a label on another.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_in(a: Box, b: Box) -> bool:
    """Centre of ``a`` lies inside ``b``."""
    cx, cy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def box_of(obj) -> Box | None:
    """Accept dicts, objects with x0..y1, or 4-sequences."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        try:
            return (float(obj["x0"]), float(obj["y0"]), float(obj["x1"]), float(obj["y1"]))
        except (KeyError, TypeError, ValueError):
            return None
    if all(hasattr(obj, k) for k in ("x0", "y0", "x1", "y1")):
        try:
            return (float(obj.x0), float(obj.y0), float(obj.x1), float(obj.y1))
        except (TypeError, ValueError):
            return None
    try:
        seq = list(obj)
        if len(seq) == 4:
            return tuple(float(v) for v in seq)  # type: ignore[return-value]
    except TypeError:
        pass
    return None


@dataclass
class TypeScore:
    char_type: str
    expected: int = 0
    predicted: int = 0
    matched: int = 0
    center_hits: int = 0
    iou_sum: float = 0.0

    @property
    def precision(self) -> float:
        return self.matched / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.expected if self.expected else 0.0

    @property
    def mean_iou(self) -> float:
        return self.iou_sum / self.matched if self.matched else 0.0

    @property
    def center_hit_rate(self) -> float:
        return self.center_hits / self.expected if self.expected else 0.0

    def to_dict(self) -> dict:
        return {
            "char_type": self.char_type,
            "expected": self.expected,
            "predicted": self.predicted,
            "matched": self.matched,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "mean_iou": round(self.mean_iou, 3),
            "center_hit_rate": round(self.center_hit_rate, 3),
        }


@dataclass
class BoxScore:
    by_type: dict[str, TypeScore] = field(default_factory=dict)
    iou_threshold: float = 0.5

    def to_dict(self) -> dict:
        return {
            "iou_threshold": self.iou_threshold,
            "by_type": {k: v.to_dict() for k, v in sorted(self.by_type.items())},
        }


def score_boxes(
    predicted: list[tuple[str, Box]],
    expected: list[tuple[str, Box]],
    *,
    iou_threshold: float = 0.5,
) -> BoxScore:
    """Greedy one-to-one matching by IoU within each char_type."""
    score = BoxScore(iou_threshold=iou_threshold)
    types = {t for t, _ in predicted} | {t for t, _ in expected}
    for char_type in types:
        ts = score.by_type.setdefault(char_type, TypeScore(char_type=char_type))
        preds = [b for t, b in predicted if t == char_type]
        exps = [b for t, b in expected if t == char_type]
        ts.predicted, ts.expected = len(preds), len(exps)
        pairs = sorted(
            ((iou(p, e), pi, ei) for pi, p in enumerate(preds) for ei, e in enumerate(exps)),
            reverse=True,
        )
        used_p: set[int] = set()
        used_e: set[int] = set()
        for value, pi, ei in pairs:
            if value < iou_threshold or pi in used_p or ei in used_e:
                continue
            used_p.add(pi)
            used_e.add(ei)
            ts.matched += 1
            ts.iou_sum += value
        # Centre hits are independent of the IoU matching: does any expected
        # box contain a predicted centre of the same type.
        for e in exps:
            if any(center_in(p, e) for p in preds):
                ts.center_hits += 1
    return score


def merge_scores(scores: Iterable[BoxScore]) -> BoxScore:
    """Pool per-sheet scores: counts and IoU sums add up per char_type."""
    merged = BoxScore()
    for score in scores:
        merged.iou_threshold = score.iou_threshold
        for char_type, ts in score.by_type.items():
            total = merged.by_type.setdefault(char_type, TypeScore(char_type=char_type))
            total.expected += ts.expected
            total.predicted += ts.predicted
            total.matched += ts.matched
            total.center_hits += ts.center_hits
            total.iou_sum += ts.iou_sum
    return merged
