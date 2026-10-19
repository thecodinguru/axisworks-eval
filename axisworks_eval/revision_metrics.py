"""The arithmetic of the revision gates, with nothing from the product in it.

The matcher and differ that produce a comparison stay in the product
(``revision_scorer`` runs them); everything here can be checked without them:
which rows are requirements, the seeded edits and their truth, and how a
comparison's output is turned into counts, precision, recall and the report's
summaries.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from axisworks_eval.cadsy_adapter import ExpectedCharacteristic

REQUIREMENT_TYPES = frozenset({"dimension", "gdt"})


def requirements(rows: Iterable[ExpectedCharacteristic]) -> list[ExpectedCharacteristic]:
    return [r for r in rows if r.char_type in REQUIREMENT_TYPES]


@dataclass
class TwinGateResult:
    part_id: str
    old_rows: int
    new_rows: int
    counts: dict[str, int]
    matched: int
    manufacturing_impacting: int
    #: Impacting deltas on MATCHED pairs: a matcher/differ error on twins,
    #: since both sheets carry the same requirements. Added/removed rows can be
    #: genuine annotation-count differences in the synthetic set.
    impacting_modified: int
    format_only: int
    unmatched_examples: list[str] = field(default_factory=list)
    impacting_examples: list[str] = field(default_factory=list)

    @property
    def matched_fraction(self) -> float:
        denom = min(self.old_rows, self.new_rows)
        return self.matched / denom if denom else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "old_rows": self.old_rows,
            "new_rows": self.new_rows,
            "counts": self.counts,
            "matched": self.matched,
            "matched_fraction": round(self.matched_fraction, 3),
            "manufacturing_impacting": self.manufacturing_impacting,
            "impacting_modified": self.impacting_modified,
            "format_only": self.format_only,
            "unmatched_examples": self.unmatched_examples[:5],
            "impacting_examples": self.impacting_examples[:5],
        }


@dataclass
class PerturbationTruth:
    modified: set[str] = field(default_factory=set)
    removed: set[str] = field(default_factory=set)
    added: set[str] = field(default_factory=set)


def perturb(
    rows: list[ExpectedCharacteristic],
    *,
    seed: int = 7,
    modify: int = 3,
    remove: int = 2,
    add: int = 2,
) -> tuple[list[ExpectedCharacteristic], PerturbationTruth]:
    """Return an edited copy of ``rows`` and the truth set of what was changed.

    Modifications change a dimension nominal by +12 % (outside the matcher's
    numeric decay, inside text similarity) or mark a frame ``MOD``; removals
    drop rows; additions append new position frames.
    """
    rng = random.Random(seed)
    work = [copy.deepcopy(r) for r in requirements(rows)]
    truth = PerturbationTruth()
    candidates = list(range(len(work)))
    rng.shuffle(candidates)
    for idx in candidates[:modify]:
        row = work[idx]
        if row.nominal is not None:
            old_text = row.nominal_text
            row.nominal = round(row.nominal * 1.12, 3)
            row.nominal_mm = row.nominal
            row.nominal_text = f"{row.nominal:g}"
            if old_text:
                row.raw_text = row.raw_text.replace(old_text, row.nominal_text, 1)
                row.normalized_text = row.normalized_text.replace(old_text, row.nominal_text, 1)
        else:
            row.raw_text = row.raw_text + " MOD"
            row.normalized_text = row.normalized_text + " MOD"
            row.spec_codes = ["MOD"]
        truth.modified.add(row.id)
    remove_ids = {work[idx].id for idx in candidates[modify : modify + remove]}
    truth.removed |= remove_ids
    work = [r for r in work if r.id not in remove_ids]
    for n in range(add):
        new_id = f"added-{n}"
        work.append(
            ExpectedCharacteristic(
                id=new_id,
                char_type="gdt",
                raw_text=f"position|⌀0.{n + 1}|A|B",
                normalized_text=f"POSITION|⌀0.{n + 1}|A|B",
                bbox={"x0": 0.5 + 0.05 * n, "y0": 0.5, "x1": 0.55 + 0.05 * n, "y1": 0.53},
                gdt_symbol="position",
                nominal=0.1 * (n + 1),
                nominal_text=f"0.{n + 1}",
                datum_refs=["A", "B"],
                zone_shape="cylindrical",
                nominal_mm=0.1 * (n + 1),
            )
        )
        truth.added.add(new_id)
    return work, truth


@dataclass
class PerturbationScore:
    truth: dict[str, int]
    predicted: dict[str, int]
    tp: dict[str, int]
    #: The delta FAI scope drafted from the comparison against the edited rows
    #: (modified and added): ``{"truth", "predicted", "tp"}``. Absent in reports
    #: written before 18 October 2026.
    scope: dict[str, int] | None = None

    def prf(self, kind: str) -> tuple[float, float, float]:
        tp = self.tp.get(kind, 0)
        p = tp / self.predicted[kind] if self.predicted.get(kind) else 0.0
        r = tp / self.truth[kind] if self.truth.get(kind) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    @property
    def f1(self) -> float:
        """Micro F1 across the three change kinds."""
        tp = sum(self.tp.values())
        pred = sum(self.predicted.values())
        truth = sum(self.truth.values())
        p = tp / pred if pred else 0.0
        r = tp / truth if truth else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"micro_f1": round(self.f1, 3)}
        for kind in ("modified", "removed", "added"):
            p, r, f = self.prf(kind)
            out[kind] = {
                "truth": self.truth.get(kind, 0),
                "predicted": self.predicted.get(kind, 0),
                "tp": self.tp.get(kind, 0),
                "precision": round(p, 3),
                "recall": round(r, 3),
                "f1": round(f, 3),
            }
        if self.scope is not None:
            tp, pred, truth = self.scope["tp"], self.scope["predicted"], self.scope["truth"]
            p = tp / pred if pred else 0.0
            r = tp / truth if truth else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            out["delta_scope"] = {
                "truth": truth,
                "predicted": pred,
                "tp": tp,
                "precision": round(p, 3),
                "recall": round(r, 3),
                "f1": round(f, 3),
            }
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerturbationScore:
        kinds = ("modified", "removed", "added")
        scope = data.get("delta_scope")
        return cls(
            truth={k: data[k]["truth"] for k in kinds},
            predicted={k: data[k]["predicted"] for k in kinds},
            tp={k: data[k]["tp"] for k in kinds},
            scope=(
                {k: scope[k] for k in ("truth", "predicted", "tp")}
                if isinstance(scope, dict)
                else None
            ),
        )


def summarize_twins(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The twin gate over every part: mean matched fraction, impacting changes summed."""
    if not results:
        return {}
    return {
        "mean_matched_fraction": round(
            sum(r["matched_fraction"] for r in results) / len(results), 3
        ),
        "impacting_total": sum(r["manufacturing_impacting"] for r in results),
        "impacting_on_matched_pairs": sum(r["impacting_modified"] for r in results),
    }


def summarize_perturbations(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Seeded edits over every part: the mean of each part's micro F1, and of the
    delta FAI scope's F1 when the scores carry one."""
    if not scores:
        return {}
    out = {"mean_micro_f1": round(sum(s["micro_f1"] for s in scores) / len(scores), 3)}
    scoped = [s["delta_scope"]["f1"] for s in scores if "delta_scope" in s]
    if scoped:
        out["mean_delta_scope_f1"] = round(sum(scoped) / len(scoped), 3)
    return out
