"""The dated accuracy report that ``make eval-report`` writes to ``reports/``.

One JSON file per day, ``reports/accuracy_<YYYY-MM-DD>.json``. It says where
the numbers came from (dataset, size, commit), gives a short headline block a
public page can show, keeps the full detail behind it, and lists what it does
not measure, so a synthetic-set number is never read as more than it is.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from axisworks_eval.real_metrics import headline_lines

SCHEMA_VERSION = 1

DATASET = {
    "name": "cadsy/cad-technical-drawings",
    "url": "https://huggingface.co/datasets/cadsy/cad-technical-drawings",
    "license": "Apache-2.0",
    "kind": "synthetic",
    "sheets_per_part": 2,
}


def report_path(directory: Path, day: dt.date) -> Path:
    return directory / f"accuracy_{day.isoformat()}.json"


def headline(details: dict[str, Any]) -> dict[str, Any]:
    """The numbers a reader looks for first; each one names its detail block."""
    out: dict[str, Any] = {}
    twins = details.get("twin_gate_summary") or {}
    if twins:
        out["twin_requirements_matched"] = twins.get("mean_matched_fraction")
        out["twin_impacting_changes_on_matched_pairs"] = twins.get("impacting_on_matched_pairs")
    perturbation = details.get("perturbation_summary") or {}
    if perturbation:
        out["seeded_change_detection_micro_f1"] = perturbation.get("mean_micro_f1")
        if "mean_delta_scope_f1" in perturbation:
            out["seeded_delta_fai_scope_f1"] = perturbation["mean_delta_scope_f1"]
    balloons = (details.get("balloons") or {}).get("summary") or {}
    for scenario in ("twins", "seeded"):
        pairs = (balloons.get(scenario) or {}).get("pairs")
        if pairs:
            out[f"balloon_numbers_kept_{scenario}"] = pairs.get("kept_fraction")
            out[f"balloon_numbers_misplaced_{scenario}"] = pairs.get("misplaced")
            out[f"balloon_misplaced_unflagged_{scenario}"] = pairs.get("misplaced_unflagged")
    extraction = (details.get("live_extraction") or {}).get("classes") or {}
    for drawing_class in ("clean", "scanned"):
        overall = (extraction.get(drawing_class) or {}).get("overall") or {}
        if overall:
            out[f"extraction_precision_{drawing_class}"] = overall.get("precision")
            out[f"extraction_recall_{drawing_class}"] = overall.get("recall")
    text_layer = (details.get("text_layer") or {}).get("overall") or {}
    if text_layer:
        out["text_layer_reads_supported_near"] = text_layer.get("supported_near_rate")
        out["text_layer_reads_supported_anywhere"] = text_layer.get("supported_anywhere_rate")
        out["text_layer_callouts_read"] = text_layer.get("callouts_read_rate")
    out.update(headline_lines(details.get("holdout") or {}))
    live = ((details.get("live_boxes") or {}).get("overall") or {}).get("by_type") or {}
    for char_type, score in sorted(live.items()):
        out[f"live_{char_type}_box_recall"] = score.get("recall")
        out[f"live_{char_type}_box_precision"] = score.get("precision")
    return out


def not_measured(details: dict[str, Any]) -> list[str]:
    """What this report leaves out, in words a reader of the public page can follow."""
    out: list[str] = []
    if "live_extraction" not in details:
        out.append(
            "Characteristic extraction precision and recall: needs a run with --live, "
            "which sends every sheet to the vision model."
        )
    if "ocr" not in details:
        out.append("OCR character error rate: needs a run with --ocr.")
    out.append("Form 3 field accuracy: there are no labelled FAIRs to score against yet.")
    if "holdout" in details:
        out.append(
            "Precision and recall on real drawings: the held-out open-hardware sheets carry "
            "machine truth only, the dimensions their DXF states (matched by value) and the "
            "PDF's own text layer (reads supported near their box, callouts read). GD&T frames, "
            "notes and title blocks have no truth there and are not scored, so nothing on real "
            "drawings is a precision or a recall. Customer and export-controlled drawings are "
            "never evaluated or published."
        )
    elif "text_layer" in details:
        out.append(
            "Precision and recall on real drawings: the labelled numbers come from synthetic "
            "drawings; the open-hardware PDFs have no hand labels and are checked against their "
            "own text layer instead. Customer and export-controlled drawings are never evaluated "
            "or published."
        )
    else:
        out.append(
            "Customer drawings: every number comes from synthetic drawings. "
            "Export-controlled samples are never evaluated or published."
        )
    return out


def build_report(
    details: dict[str, Any],
    *,
    root: str,
    parts: int,
    generated_at: dt.datetime,
    commit: str = "",
    uncommitted_code: bool = False,
    versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "commit": commit,
        "uncommitted_code": uncommitted_code,
        "versions": versions or {},
        "dataset": {**DATASET, "root": root, "parts": parts},
        "headline": headline(details),
        "not_measured": not_measured(details),
        "details": details,
    }
