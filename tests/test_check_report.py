"""Every published report checks out, and its truth rebuilds from the fixture parts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from axisworks_eval.cadsy_adapter import load_rows
from axisworks_eval.degrade import degrade
from axisworks_eval.verify import check_report, check_truth

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
