# AxisBench v0: the preregistered protocol

**Status.** Preregistered on 22 October 2026, before any scored read of the
holdout. The first commit of this file is tagged `protocol-v0` in this
repository; the commits that carry it are named in §11. Everything above the
"Amendments" heading is fixed from that tag; a later change is appended there,
dated, with its reason.

## 1. What is measured, by whom

AxisBench v0 scores the AxisWorks drawing reader on real engineering drawings
it was not tuned on. Superposition Labs runs the benchmark; AxisWorks is the
product under test. The product repository produces a dated report
(`scripts/run_eval.py --holdout`, writing `details.holdout`); this repository
checks every number in it without the product:

    python scripts/check_report.py <report> --holdout fixtures/drawing-testset-v1

The check rebuilds each sheet's truth from the vectors, rescores the reads the
report lists, recomputes every table and the headline, and exits 1 on any
difference. The report's `schema` is 1.

## 2. Prior exposure, stated

- Cadsy (`fixtures/cadsy`, 65 synthetic parts) is the tuning set. Two changes
  were fitted to its errors on 17 October 2026: the frame-echo rule (extractor
  version 3) and the revision matcher. No Cadsy number is a holdout number.
- Before the split was sealed on 18 October, one partial run read 277 of the
  604 test-set PDFs against their own text layers (product commit `6251d43`,
  extractor version 3). Those sheets now lie on both sides of the split. The
  run was never committed, no report was published from it, and no change was
  fitted to its reads. It is listed so that nobody has to take that on trust.
- Scored holdout reads to date: none.

## 3. Dataset and split

`drawing-testset-v1` (`fixtures/drawing-testset-v1/`): 2,000 sheets from 27
publishers, each with its public repository, pinned commit and license
(`manifest.jsonl`, `licenses/`). Its vector sheets of the three classes a
dimension benchmark can score are split once, by publisher, into a `holdout`
and a `dev` side by `axisworks_eval/holdout.py`; the split is committed as
`holdout_v1.jsonl` (SHA-256
`7bc69609b98130ec3e81672f82127a3a89ec09397bb9f96e3f5c98eba59a19d0`, tag
`holdout-v1`). The dev side is a fixed list of five publishers; a revision
group or a same-drawing pair that would straddle the sides goes to the holdout
whole. `holdout.check` recomputes the split from the manifest and refuses a
file that differs; `check_report.py --holdout` runs it first.

What each side holds (`holdout.summary`, with the CAD truth and text-layer
counts below taken from the vectors on 22 October 2026):

| Side | Category | Sheets | Sheets with CAD truth | Numbered printed dimensions | Sheets with ≥ 20 text-layer words |
|---|---|---:|---:|---:|---:|
| holdout | mechanical_drawing | 311 | 7 | 420 | 297 |
| holdout | pcb_fab_assembly_drawing | 162 | 1 | 35 | 109 |
| holdout | cad_dxf | 56 | 23 | 937 | 0 |
| holdout | **all** (18 publishers) | **529** | **31** | **1,392** | **406** |
| dev | mechanical_drawing | 122 | 0 | 0 | 118 |
| dev | cad_dxf | 12 | 3 | 36 | 0 |
| dev | **all** (5 publishers) | **134** | **3** | **36** | **118** |

CAD truth is the DIMENSION entities of a DXF (`axisworks_eval/dxf_truth.py`):
the 56 DXF-native holdout sheets, of which 23 state at least one numbered
dimension, and the DXF exports attached to 8 holdout PDFs (`alt_formats`).
The text layer is the words the CAD tool embedded in a PDF
(`text_layer_metrics.read_words`, pdfplumber 0.11.9).

## 4. Exclusions

- Not in the split: electrical schematics (1,321 sheets), block diagrams (9)
  and outline images (7), which carry no dimensions in the sense of a
  characteristic table; and any sheet without a vector source.
- In the split but unscored: a sheet with fewer than 20 text-layer words
  (`MIN_WORDS`) gets no text-layer score; a sheet with neither CAD truth nor a
  text layer gets no score and counts in `sheets` only. Feature control
  frames, notes and title blocks have no truth on these drawings (frames are
  drawn as curves) and are not scored.
- A read that fails (an extraction error, or a DXF sheet without the render
  the set ships) is listed in `failed` and stays in its cell's `sheets`;
  nothing is removed from the holdout after a run.

## 5. Measures, by the function that computes them

Each holdout sheet is read once per configuration in each drawing class:
`clean`, the sheet rendered as the product renders an upload, and `scanned`,
`axisworks_eval/degrade.py` applied to that image (two-thirds scale, rotated
0.4°, JPEG quality 60, back to PNG; deterministic). A read is the product's
`dimensions` and `gdt_frames` calls on the image; the report lists every read
(`kind`, `text`, `box`) per sheet.

- **Dimensional value match** (`real_metrics.score_values`): each printed
  dimension with at least one number is matched to at most one dimension
  read, in order; `value_match_full_rate` when a read states every number the
  dimension prints (nominal and tolerance), `value_match_nominal_rate` when a
  read states its first number first. Only sheets with CAD truth contribute.
  This is the plan's "dimensional value match". It cannot give precision: a
  note or a hole table is not a dimension entity.
- **Supported near** (`text_layer_metrics.score_sheet`,
  `supported_near_rate`): the share of reads whose every number is printed in
  the text layer within `MARGIN` (0.02 of the sheet) of the read's box. This
  is the precision proxy, and is named as a proxy. `supported_anywhere_rate`
  (printed anywhere on the sheet) is reported beside it.
- **Callouts read** (`callouts_read_rate`): the share of the text layer's
  toleranced, diameter, radius, thread and angular callouts that some read
  states. A callout is a pattern in the text, not a label.
- **Tables** (`real_metrics.summarize`): every measure per drawing class ×
  category, plus a per-class `overall` and the clean overall as the report's
  `overall`. Each cell carries `sheets`, `failed` and `sheets_with_cad_truth`.
- **Headline** (`real_metrics.headline_lines`): per class,
  `holdout_values_matched`, `holdout_reads_supported_near`,
  `holdout_callouts_read`, and the worst slice (§7).

Nothing on these drawings is a precision or a recall, and the report's
`not_measured` list says so.

## 6. Gates

From the plan, measured per drawing class (clean and scanned separately):

| Gate | Target | On this holdout |
|---|---|---|
| Dimensional value match | ≥ 0.97 | `holdout_values_matched_{class}` (`value_match_full_rate` of the class's overall). The one plan gate the holdout can measure. |
| Characteristic precision and recall | ≥ 0.95 | **Not measurable here**: no hand labels. Reported on Cadsy only, as a synthetic-set number. |
| GD&T structured parse | ≥ 0.93 | **Not measurable here**: frames are drawn as curves, so there is no frame truth. Reported on Cadsy only. |
| Revision added, changed and removed F1 | ≥ 0.95 | **Not measurable here**: the set's revision groups carry no labelled differences. Reported on Cadsy's seeded edits only. |

The text-layer proxies have no target; they are reported so that a value-match
figure can be read next to the sheets the product reads where no CAD truth
exists. "An unsafe false positive blocks a release whatever the macro score"
(the plan) stands over every gate.

## 7. Slices

Every category × class cell of the table is published. The headline also
names, per class, the **worst slice**: the category with the lowest
`value_match_full_rate` among categories with at least five sheets carrying
CAD truth (`real_metrics.worst_slice`, `WORST_SLICE_MIN_SHEETS = 5`), as
`holdout_worst_slice_{class}` and `holdout_worst_slice_values_matched_{class}`.
On this holdout the candidates are `mechanical_drawing` (7 sheets, 420
dimensions) and `cad_dxf` (23 sheets, 937 dimensions); `pcb_fab_assembly_drawing`
has one sheet of CAD truth and is reported in its cell but can never be the
slice. The worst slice leads the public page, before the aggregate.

A **critical slice** (tight tolerances, feature control frames) is **not
measurable on this holdout**: a read carries a kind, a text and a box, and
CAD truth carries no tolerance class or criticality. It is computed on Cadsy
only, whose labels carry tolerances, is labelled a dev-set number, and is
never a gate.

## 8. Configurations to be run

Each configuration reads every holdout sheet once in each class (529 sheets,
1,058 reads), with the same prompts and extractor:

1. **The hosted commercial model**, as the product's hosted tier is
   configured on the day of the run. On 18 October 2026 that was provider
   `claude_api`, model `claude-opus-5-5` with `claude-sonnet-4-6` for the
   metadata call, concise prompts, the GD&T reference image and keyword
   knowledge-base retrieval (`vlm_provider`, `vlm_model`,
   `vlm_model_overrides`, `extraction_profile`, `concise_prompts`,
   `gdt_reference_image`, `kb_injection`, `kb_retrieval` in the report's
   `configuration`). This is the last report on a commercial model.
2. **Open-weight models on vLLM**: Qwen3-VL-8B-Instruct (Apache-2.0) and
   MechVL-4B (the drawing specialist and the public baseline), guided JSON on,
   the exact model identifier and the vLLM image tag recorded in
   `configuration`.

Every report records `prompt_version` and `extractor_version` (both `3` at
preregistration), the product commit, and `uncommitted_code`, which must be
`false`. Reads are cached by image hash and version stamp, so a run that
stops (the plan's circuit breaker) resumes without a second read of a sheet.

## 9. No tuning on the holdout

Prompts, the extractor, the matcher, grounding and the rules are tuned on
Cadsy and the `dev` side only. The `holdout` side is read by the report run
and nothing else: once per configuration. A second read of the same
configuration is a new dated report that states why it was made. The split
(`holdout_v1.jsonl`) is never regenerated; a sheet is never dropped.

## 10. Commit before run

A report's `commit` must descend from the product commit that carries this
protocol (§11). A change to `real_metrics.py`, `text_layer_metrics.py`,
`dxf_truth.py`, `holdout.py` or `degrade.py` after the first scored holdout
read is an amendment (below), and the report that follows it says so.

## 11. Versions and tooling

- Protocol commits: product `1d366ef` (axisworks-service); this repository
  `8947ed2`, tag `protocol-v0`.
- Prompt version 3, extractor version 3 (`versions.py` in the product).
- pdfplumber 0.11.9 (the words of the text layer), ezdxf ≥ 1.4 (CAD truth),
  Pillow (the scan); Python 3.11 or later.
- `scripts/check_report.py <report> --holdout fixtures/drawing-testset-v1`
  is the verification command; `python -m pytest` runs this repository's own
  tests, which rebuild a sample of the holdout scores from the vectors.

## 12. Failure reporting

Every cell, every `failed` sheet and every `unread_examples` list is
published with the report. No sheet, category, class or configuration is
left out of the public page because of its number; the worst slice is shown
first.

## 13. Reviewers and the contamination audit

- Two reviewers unaffiliated with Superposition Labs run the verification
  command on the published report and the sealed split, and their names and
  the checked commit are recorded beside the report.
- Contamination audit: open-weight vision models do not publish their
  training lists, so the audit can only state, per holdout publisher, the
  public repository the sheets come from and whether it predates the model's
  stated data cutoff. That is the limit of what can be audited, and the
  result is published as such. Result: pending.

## Amendments

None.
