# AxisWorks evaluation, published for verification

The accuracy figures on [superpositionlabs.ai/accuracy](https://superpositionlabs.ai/accuracy)
come from a dated report in `reports/`. This repository lets you check that
report without our product: the arithmetic behind every number, and the truth
each number was scored against, rebuilt from the public dataset.

© Superposition Labs, Inc. All rights reserved. The code is published so the
report can be verified; no license is granted to use, copy, modify or
distribute it. The Cadsy drawings in `fixtures/cadsy/` are Apache-2.0 (see the
`LICENSE` and `ATTRIBUTION.md` beside them).

## The dataset

[Cadsy CAD technical drawings](https://huggingface.co/datasets/cadsy/cad-technical-drawings):
65 synthetic parts, each drawn twice (an ASME-style and an ISO-style sheet),
with a box and the text of every dimension, callout, feature control frame and
datum on each sheet. Apache-2.0. No customer drawing, and nothing
export-controlled, is in any number here.

## What you can check

**1. Every derived number follows from its details.** The summaries, the
headline and the list of what is not measured are recomputed from the
per-part results the report carries:

    python scripts/check_report.py reports/accuracy_2026-10-18.json

**2. The truth is the public labels.** With the dataset downloaded, the truth
each result was scored against is rebuilt from the labels: the requirement
rows of every sheet, the seeded revision edits (seed 7), the twin pairs, the
moved view and the labelled characteristics the extraction is scored on:

    python scripts/fetch_cadsy_dataset.py
    python scripts/check_report.py reports/accuracy_2026-10-18.json --labels datasets/cadsy/samples

**3. The scanned drawings.** `axisworks_eval/degrade.py` makes the scanned
class from each clean sheet: two-thirds scale (a 300 dpi original scanned at
200 dpi), rotated 0.4 degrees, JPEG at quality 60, back to PNG. The same
sheet always gives the same image.

## What stays private

The product's own work: the revision matcher and differ that pair
characteristics, the balloon numbering that carries numbers into a new
revision, and the extraction that reads a sheet with the vision model. The
report publishes their output as counts per part; this repository checks
everything around them. A report says which commit of the product produced
it, which model read the sheets and how the run was configured.

## What a report measures

- **Revision comparison (twins).** The ASME and ISO sheets of one part carry
  the same requirements laid out differently. A correct comparison pairs them
  and reports no change that affects manufacturing.
- **Change detection (seeded).** Known edits are made to a sheet's
  requirements: values changed, rows removed, frames added. Precision and
  recall of the changes found.
- **Balloon numbers.** Whether a feature keeps its balloon number in a new
  revision, and whether a number ever lands on the wrong feature without a
  review flag.
- **Extraction.** What the extractor reads from each sheet against the labels:
  precision and recall by characteristic type, drawing standard and drawing
  class (clean, scanned). A read counts as correct only when its value matches;
  a callout found but misread counts against both precision and recall.

Each report also lists what it does not measure.

## Layout

- `axisworks_eval/`: label parsing (`cadsy_adapter`), box scoring
  (`bbox_scorer`), the scan (`degrade`), the arithmetic of each measure
  (`revision_metrics`, `balloon_metrics`, `extraction_metrics`), the report
  (`report`) and the checks (`verify`).
- `reports/`: the dated reports.
- `fixtures/cadsy/`: four of the 65 parts, enough for the tests.
- `tests/`: `python -m pytest`.

Python 3.11 or later and Pillow; `pip install -e ".[test]"` adds pytest.
