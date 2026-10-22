"""Every published report checks out, and its truth rebuilds from the fixture parts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from axisworks_eval.cadsy_adapter import load_rows
from axisworks_eval.degrade import degrade
from axisworks_eval.holdout import check as check_split
from axisworks_eval.verify import check_holdout, check_report, check_text_layer, check_truth

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "cadsy"
REPORTS = sorted((ROOT / "reports").glob("accuracy_*.json"))
FIXTURE_PARTS = sorted(p.name for p in FIXTURES.iterdir() if p.name.startswith("part_"))


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.name)
def test_every_derived_number_follows_from_its_details(path):
    assert check_report(json.loads(path.read_text(encoding="utf-8"))) == []


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.name)
def test_the_truth_of_the_fixture_parts_rebuilds_from_their_labels(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    assert check_truth(report, FIXTURES, only=FIXTURE_PARTS) == []


def test_the_labels_parse():
    rows = load_rows(FIXTURES / "part_0001" / "asme_labels.json")
    assert rows and {r.char_type for r in rows} >= {"dimension", "gdt"}


def test_a_scan_is_deterministic():
    png = (FIXTURES / "part_0001" / "asme.png").read_bytes()
    assert hashlib.sha256(degrade(png)).digest() == hashlib.sha256(degrade(png)).digest()


TESTSET = ROOT / "fixtures" / "drawing-testset-v1"


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.name)
def test_test_set_scores_rebuild_from_the_pdfs(path):
    """Every 25th sheet here; scripts/check_report.py --testset rebuilds them all."""
    report = json.loads(path.read_text(encoding="utf-8"))
    section = report.get("details", {}).get("text_layer")
    if not section or not TESTSET.is_dir():
        pytest.skip("no test-set section in this report, or the test set is not here")
    sample = {**report, "details": {"text_layer": {"sheets": section["sheets"][::25]}}}
    assert check_text_layer(sample, TESTSET) == []


def test_the_sealed_split_follows_from_the_manifest():
    if not TESTSET.is_dir():
        pytest.skip("the test set is not here")
    assert check_split(TESTSET) == []


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.name)
def test_holdout_scores_rebuild_from_the_vectors(path):
    """Every 20th held-out sheet here; scripts/check_report.py --holdout rebuilds them all."""
    report = json.loads(path.read_text(encoding="utf-8"))
    section = report.get("details", {}).get("holdout")
    if not section or not TESTSET.is_dir():
        pytest.skip("no holdout section in this report, or the test set is not here")
    sample = {**report, "details": {"holdout": {**section, "sheets": section["sheets"][::20]}}}
    assert check_holdout(sample, TESTSET) == []


def test_a_holdout_headline_names_the_worst_slice():
    """A report with a holdout section checks out, and its headline names the worst slice."""
    import datetime as dt

    from axisworks_eval.real_metrics import ValueScore
    from axisworks_eval.real_metrics import summarize as summarize_holdout
    from axisworks_eval.report import build_report

    def entry(sheet_id, category, full):
        return {
            "sheet_id": sheet_id,
            "category": category,
            "class": "clean",
            "split": "holdout",
            "values": ValueScore(
                truth=10, reads=10, matched_full=full, matched_nominal=10
            ).to_dict(),
        }

    sheets = [entry(f"m{i}", "mechanical_drawing", 9) for i in range(5)] + [
        entry(f"d{i}", "cad_dxf", 7) for i in range(5)
    ]
    details = {"holdout": {**summarize_holdout(sheets), "sheets": sheets}}
    report = build_report(
        details, root="fixtures", parts=0, generated_at=dt.datetime(2026, 10, 22, tzinfo=dt.UTC)
    )
    assert check_report(report) == []
    assert report["headline"]["holdout_values_matched_clean"] == 0.8
    assert report["headline"]["holdout_worst_slice_clean"] == "cad_dxf"
    assert report["headline"]["holdout_worst_slice_values_matched_clean"] == 0.7
