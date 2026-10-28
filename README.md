# AxisWorks evaluation, published for verification

The accuracy figures on [superpositionlabs.ai](https://superpositionlabs.ai/)
come from a dated report in `reports/`. This repository lets you check that
report without our product: the arithmetic behind every number, and the truth
each number was scored against, rebuilt from the public dataset.

© Superposition Labs, Inc. All rights reserved. The code is published so the
report can be verified; no license is granted to use, copy, modify or
distribute it. The Cadsy drawings in `fixtures/cadsy/` are Apache-2.0 (see the
`LICENSE` and `ATTRIBUTION.md` beside them). The drawings in
`fixtures/drawing-testset-v1/` keep their own open licenses, one per source
(`licenses/`, `ATTRIBUTION.md` and `manifest.jsonl` there).

## The dataset

[Cadsy CAD technical drawings](https://huggingface.co/datasets/cadsy/cad-technical-drawings):
65 synthetic parts, each drawn twice (an ASME-style and an ISO-style sheet),
with a box and the text of every dimension, callout, feature control frame and
datum on each sheet. Apache-2.0. Beside it, the open-hardware test set (below).
No customer drawing, and nothing export-controlled, is in any number here.

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

**3. The open-hardware test set.** `fixtures/drawing-testset-v1/` holds 604
single-page PDFs from public repositories (machined parts, enclosures, PCB
fabrication and assembly drawings, block diagrams), each with its source,
commit and license in `manifest.jsonl` and `licenses/`. They have no hand
labels, so a report reads each one once and checks every dimension and frame
it read against the PDF's own text layer: whether each number the read states
is printed near its box, and how many of the toleranced and diameter callouts
in the text were read. The reads are listed sheet by sheet, so every score
rebuilds from the PDFs:

    python scripts/check_report.py <report> --testset fixtures/drawing-testset-v1

These are not precision and recall: text a CAD tool drew as curves has no text
layer, and a callout is a pattern in the text, not a label.

**4. The held-out real drawings.** The vector sheets of the test set are split
once, by publisher, into a `holdout` and a `dev` side
(`fixtures/drawing-testset-v1/holdout_v1.jsonl`, written by
`axisworks_eval/holdout.py`; its SHA-256 is below). The product is tuned on the
synthetic set and the `dev` side only; the `holdout` side is read by the report
run and nothing else. Two kinds of machine truth stand in for labels there:
the dimensions a DXF states, read from its DIMENSION entities
(`dxf_truth.py`: 68 DXF-native sheets and the DXF exports attached to 9 PDF
sheets), and the PDF's own text layer. A report scores each read against them
by drawing category and class (`real_metrics.py`): how many printed dimensions
a read states in full (value match), whether every number a read states is
printed near its box, and how many callouts were read. Every score rebuilds
from the sealed split and the vectors:

    python scripts/check_report.py <report> --holdout fixtures/drawing-testset-v1

The split itself is checked the same way; a `holdout_v1.jsonl` that does not
follow from `manifest.jsonl` fails the check. Frames, notes and title blocks
have no truth there and are not scored; nothing on real drawings is a
precision or a recall. The run is preregistered in [`PROTOCOL.md`](PROTOCOL.md):
the sealed split, the measures by function, the gates, the slices and the
worst-slice rule, the configurations to be run, and the no-tuning and
commit-before-run rules, committed before the first scored read.

**5. The scanned drawings.** `axisworks_eval/degrade.py` makes the scanned
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
- **Real drawings (text layer).** The open-hardware PDFs read once each:
  reads whose numbers the drawing's text supports, near the read and anywhere
  on the sheet, and the share of the text's dimension callouts read, by
  drawing category.
- **Real drawings (holdout).** The held-out sheets read once each, clean and
  scanned: printed dimensions matched by value against CAD truth, reads
  supported by the text layer, callouts read, by category and drawing class;
  the headline names the worst slice, the category with the lowest value
  match among those with enough CAD truth.
- **Extraction.** What the extractor reads from each sheet against the labels:
  precision and recall by characteristic type, drawing standard and drawing
  class (clean, scanned). A read counts as correct only when its value matches;
  a callout found but misread counts against both precision and recall.

Each report also lists what it does not measure.

## Layout

- `axisworks_eval/`: label parsing (`cadsy_adapter`), box scoring
  (`bbox_scorer`), the scan (`degrade`), the arithmetic of each measure
  (`revision_metrics`, `balloon_metrics`, `extraction_metrics`), the text-layer
  check (`text_layer_metrics`), the sealed split (`holdout`), CAD truth
  (`dxf_truth`) and the real-drawing scores (`real_metrics`), the report
  (`report`) and the checks (`verify`).
- `PROTOCOL.md`: the preregistered protocol of AxisBench v0.
- `reports/`: the dated reports.
- `fixtures/cadsy/`: four of the 65 parts, enough for the tests.
- `fixtures/drawing-testset-v1/`: the open-hardware vectors (PDF and DXF) with
  their licenses, the DXF renders the set ships (`sheets/cad_dxf/`), the
  revision groups, and the sealed split `holdout_v1.jsonl`
  (SHA-256 `7bc69609b98130ec3e81672f82127a3a89ec09397bb9f96e3f5c98eba59a19d0`,
  tag `holdout-v1`).
- `tests/`: `python -m pytest`.

Python 3.11 or later, Pillow, pdfplumber 0.11.9 (the version the report read
words with) and ezdxf 1.4 or later (CAD truth); `pip install -e ".[test]"`
adds pytest.
