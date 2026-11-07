"""The dated accuracy report that ``make eval-report`` writes to ``reports/``.

One JSON file per day, ``reports/accuracy_<YYYY-MM-DD>.json``. It says where
the numbers came from (dataset, size, commit), gives a short headline block a
public page can show, keeps the full detail behind it, and lists what it does
not measure, so a synthetic-set number is never read as more than it is.

The MechVQA track (``scripts/run_mechvqa.py``) writes its own file,
``reports/mechvqa_<YYYY-MM-DD>.json``, with the same shape: its ``dataset``
block names the public set and its pin instead of Cadsy, and its ``details``
hold the one ``mechvqa`` block. The two never share a file, so a reader of an
accuracy report never finds a judged benchmark number inside it.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import axisworks_eval.mechvqa_metrics as mechvqa_metrics
from axisworks_eval.real_metrics import headline_lines

SCHEMA_VERSION = 1

DATASET = {
    "name": "cadsy/cad-technical-drawings",
    "url": "https://huggingface.co/datasets/cadsy/cad-technical-drawings",
    "license": "Apache-2.0",
    "kind": "synthetic",
    "sheets_per_part": 2,
}
#: The judges the MechVQA paper's Table 2 was scored with; a report whose judges are
#: not these three publishes no comparison with that table.
PAPER_JUDGES = ("GPT-OSS-120B", "DeepSeek-V3.2", "Kimi-k2")


def report_path(directory: Path, day: dt.date, prefix: str = "accuracy") -> Path:
    return directory / f"{prefix}_{day.isoformat()}.json"


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
    provenance = (details.get("provenance") or {}).get("by_class") or {}
    for drawing_class, cell in sorted(provenance.items()):
        out[f"provenance_box_rate_{drawing_class}"] = cell.get("box_rate")
    live = ((details.get("live_boxes") or {}).get("overall") or {}).get("by_type") or {}
    for char_type, score in sorted(live.items()):
        out[f"live_{char_type}_box_recall"] = score.get("recall")
        out[f"live_{char_type}_box_precision"] = score.get("precision")
    out.update(mechvqa_metrics.headline_lines(details.get("mechvqa") or {}))
    return out


def _mechvqa_not_measured(block: dict[str, Any]) -> list[str]:
    """What a MechVQA number is not, written from the block itself so the words follow
    the run: which judges voted, whether they are the paper's, how much of the set is
    Chinese, which configuration abstains by design."""
    dataset = block.get("dataset") or {}
    configurations = block.get("configurations") or {}
    judges = [str(j.get("id")) for j in (block.get("judges") or []) if j.get("id")]
    images = dataset.get("images")
    out = [
        "Industrial drawings: MechVQA's "
        + (f"{images:,} " if isinstance(images, int) else "")
        + "drawings are public textbook and handbook sheets, not the drawings a customer "
        "sends, so no MechVQA number is a number on industrial work.",
        "Machine truth: every MechVQA score is a language model's judgment "
        + (f"(judges {', '.join(judges)}) " if judges else "")
        + "of an answer against the set's reference answer, not a comparison with a DXF or "
        "a text layer; a disagreement between the judges is published per judge.",
    ]
    names = ", ".join(PAPER_JUDGES[:-1]) + f" and {PAPER_JUDGES[-1]}"
    if block.get("judges_as_paper"):
        out.append(
            f"The judges are the paper's ({names}), so its Table 2 is the comparison; the "
            "judge prompt is the repository's, which differs from the paper's figure."
        )
    else:
        out.append(
            f"The judges are not the paper's ({names}), so no comparison with its Table 2 is "
            "published; the numbers stand on their own."
        )
    out.append(
        "Contamination: the benchmark is public on GitHub and Hugging Face, and a model "
        "trained on the web may have seen its drawings or its answers; prior exposure "
        "cannot be ruled out for any model scored here."
    )
    if "record" in configurations:
        out.append(
            "The record configuration abstains by design when the record it built from the "
            "drawing holds no answer; its abstentions score 0 and are counted, never dropped."
        )
    summary = next(
        (c.get("summary") for c in configurations.values() if (c or {}).get("summary")), None
    )
    if summary:
        total = (summary.get("total") or {}).get("questions") or 0
        chinese = ((summary.get("by_language") or {}).get("zh") or {}).get("questions") or 0
        if total:
            out.append(
                f"Language: {chinese:,} of {total:,} questions ({chinese / total:.1%}) are in "
                "Chinese; the answer-language match rate says how often an answer came back "
                "in the question's language."
            )
    mismatches = dataset.get("capability_mismatches")
    if mismatches is None:
        mismatches = sum(
            1
            for c in configurations.values()
            for r in (c or {}).get("questions") or []
            if r.get("stated_capability") and r.get("stated_capability") != r.get("capability")
        )
    out.append(
        f"Taxonomy: {mismatches} record{'s' if mismatches != 1 else ''} name a capability the "
        "paper's taxonomy gives to another; each row keeps the record's label and the "
        "capability means use the taxonomy."
    )
    return out


def not_measured(details: dict[str, Any]) -> list[str]:
    """What this report leaves out, in words a reader of the public page can follow."""
    out: list[str] = []
    if "mechvqa" in details:
        out += _mechvqa_not_measured(details["mechvqa"] or {})
        if set(details) <= {"mechvqa"}:
            return out
    if "live_extraction" not in details:
        out.append(
            "Characteristic extraction precision and recall: needs a run with --live, "
            "which sends every sheet to the vision model."
        )
    if "ocr" not in details:
        out.append("OCR character error rate: needs a run with --ocr.")
    out.append("Form 3 field accuracy: there are no labelled FAIRs to score against yet.")
    if "provenance" in details:
        out.append(
            "Provenance of manual rows and derived limits: the report counts the reads that "
            "carry a box and name the stage that placed it; rows a person adds and tolerances "
            "taken from a title block are checked in the product (GET /compliance/provenance), "
            "not here."
        )
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
    dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The report; ``dataset`` replaces the Cadsy block when the run was on another set
    (MechVQA names its repository, pin and hashes there)."""
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "commit": commit,
        "uncommitted_code": uncommitted_code,
        "versions": versions or {},
        "dataset": {**(DATASET if dataset is None else dataset), "root": root, "parts": parts},
        "headline": headline(details),
        "not_measured": not_measured(details),
        "details": details,
    }
