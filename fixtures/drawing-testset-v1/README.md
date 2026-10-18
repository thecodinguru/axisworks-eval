# Engineering Drawing Test Set v1 — public, openly licensed

**2000 drawing sheets** from 27 publishers, every one traceable to a public repository, a pinned commit,
and a license that permits redistribution and commercial evaluation use. Built 2026-10-17.

## What is in the box

| Category | Sheets | Share |
|---|---:|---:|
| electrical_schematic | 1321 | 66.0% |
| mechanical_drawing | 433 | 21.6% |
| pcb_fab_assembly_drawing | 162 | 8.1% |
| cad_dxf | 68 | 3.4% |
| block_diagram | 9 | 0.5% |
| mechanical_outline_drawing | 7 | 0.3% |

Category definitions:
- **mechanical_drawing** — Detail / assembly / sheet-metal / machined-part drawings with title blocks, dimensions, tolerances, notes
- **pcb_fab_assembly_drawing** — PCB fabrication, assembly, outline and board-dimension drawings
- **electrical_schematic** — Electrical schematic sheets (capped at 14 sheets per source file for diversity)
- **cad_dxf** — CAD-native DXF drawings rendered to PNG, original DXF included; rows with `same_drawing_as` are the DXF export of a PDF sheet in the set (format-robustness pairs, not new drawings)
- **mechanical_outline_drawing** — Dimensioned mechanical outline drawings published as images
- **block_diagram** — System / block diagrams

| Domain | Sheets | Share |
|---|---:|---:|
| electronics | 844 | 42.2% |
| enclosure | 361 | 18.1% |
| electronics-rf | 320 | 16.0% |
| aerospace | 209 | 10.4% |
| machine | 140 | 7.0% |
| robotics | 67 | 3.4% |
| aerospace-uav | 51 | 2.5% |
| aerospace-nasa | 8 | 0.4% |

| License (SPDX) | Sheets | Share |
|---|---:|---:|
| CC-BY-SA-4.0 | 563 | 28.1% |
| CC-BY-4.0 | 368 | 18.4% |
| SHL-2.1 | 201 | 10.1% |
| GPL-3.0-only | 168 | 8.4% |
| CERN-OHL-2.0 | 160 | 8.0% |
| CC-BY-3.0 | 121 | 6.0% |
| CC-BY-SA-3.0-US | 99 | 5.0% |
| MIT | 59 | 3.0% |
| CC-BY-SA-3.0 | 55 | 2.8% |
| Apache-2.0 | 55 | 2.8% |
| CERN-OHL-1.2 | 53 | 2.6% |
| 0BSD | 53 | 2.6% |
| CERN-OHL-S-2.0 | 34 | 1.7% |
| BSD-3-Clause | 11 | 0.6% |

- **672** sheets ship with their original vector source (single-page PDF or DXF) next to the raster — every
  mechanical, PCB, DXF and block-diagram sheet. Schematic vector pages are left out to keep the zip near 0.5 GB;
  `python tools/fetch_vectors.py --root .` re-extracts them byte-exactly from the pinned repo/commit/path/page
  (sha256-checked against `source_file_sha256`).
- **1662** sheets ship with a **text-layer label file** (`labels/text_layer/*.json`): every word the CAD tool embedded
  in the PDF, with its bounding box in PNG pixel coordinates. That is free ground truth for title-block, dimension and
  note extraction on vector drawings. It is exact where text is real text and absent where the CAD export drew text as
  curves (common for GD&T glyphs and some title blocks) — treat it as a high-precision, partial-recall label.
- **73 revision / variant groups** (`revision_groups.csv`, 598 sheets): the same part number or
  board across revisions (e.g. System76 PN-000164 v1/v2/v3, Foundation Devices FD_1842300 rev letters, moteus r4.3 vs
  r4.5, OLIMEX Rev C…J) and metric/inch variants (ODRI). Use them for revision-compare and change-detection tests.
  Grouping is by normalised file name — spot-check before using a group as a ground-truth diff pair.

## Quick start

```python
import json
rows = [json.loads(l) for l in open("manifest.jsonl")]
mech = [r for r in rows if r["category"] == "mechanical_drawing"]
pairs = {}                                # revision-compare cases
for r in rows:
    if r["revision_group"]: pairs.setdefault(r["revision_group"], []).append(r["sheet_id"])
# r["file_png"] -> raster, r["file_vector"] -> single-page PDF/DXF, r["file_text_layer"] -> word boxes
```

Unique-drawing count: 1979 — the 21 Framework DXF sheets flagged `same_drawing_as` duplicate the content of PDF sheets in another format.

## Layout

```
manifest.jsonl / manifest.csv   one row per sheet: files, category, domain, page, source repo + commit + path,
                                permalink, license, publisher, sha256, modifications
sheets/<category>/DRW-xxxxx.png  grayscale raster, long edge 2400 px
source_vector/<category>/        single-page vector PDF or DXF for the same sheet (alt formats as .altN.dxf)
labels/text_layer/               embedded-text word boxes per sheet (PDF sources only)
revision_groups.csv              sheets that are revisions / variants of one another
licenses/<source_id>/            the source repository's own license file(s) + README licensing excerpt + SOURCE.txt
ATTRIBUTION.md                   publisher, repo@commit, license, evidence, sheet count — keep this with any copy
stats.json                       the counts above, machine-readable
tools/                           the scripts that built this set and the top-up fetcher for federal sources
```

## How sheets were selected

1. **Sources.** Only repositories whose license was verified from a LICENSE file or an explicit README licensing
   statement. Non-commercial (CC BY-NC*) and unlicensed repositories were excluded even when they held excellent
   drawings (Ultimaker, Rat Rig, OpenROV — non-commercial; PSAS Launch Tower, ArduPilot schematics, M5Stack,
   OSE LifeTrac — no license). iCub electronics (CERN-OHL-W) was dropped because GitHub does not serve its LFS files, and AMSAT CubeSatSim was
   dropped because its board artwork says CC BY-NC-SA even though the repository LICENSE is GPL-3.0.
2. **Vendor material excluded.** A repository's license covers its authors' work, not datasheets or connector drawings
   they copied in, so paths matching vendor/datasheet/COTS/connector/manual/report/certificate patterns were dropped.
3. **Drawing gate.** Every PDF page was rendered and kept only if it is line-dense and not prose
   (Hough line segments ≥ 40, ≤ 1,500 words, not "> 400 words with < 80 segments", non-blank).
4. **Dedupe.** Byte-identical files across repositories (Ender-3 and CR-10 share parts) and visually identical
   pages (same 64-bit dHash, page size and text count) were collapsed. Near-identical pages from *different
   revisions* were kept deliberately — they are the revision-compare cases. A DXF whose name matches a PDF drawing
   is attached to that sheet as an alternate format; Framework DXFs whose names did not match are kept as `cad_dxf`
   sheets and linked to their PDF twin via `same_drawing_as`.
5. **Balance.** Every mechanical, DXF and outline sheet that passed was kept. Schematic and PCB sheets were capped
   at 14 per source file and 320 per publisher, then round-robined across publishers to fill the remainder of the
   2,000, so no single board family dominates.
6. **Markings.** Pages whose text layer carries ITAR / EXPORT CONTROLLED / DISTRIBUTION STATEMENT B–F / CUI / FOUO /
   NOFORN markings are dropped (none were found), as are pages whose own text claims a non-commercial license.
   Pages with an "All rights reserved" notice are kept (the publisher's repository license governs) but flagged
   (`all_rights_reserved_notice`, 0 sheets). Sheets whose title block carries a template "PROPRIETARY /
   CONFIDENTIAL" notice are kept — the owner later released them under an open license — but flagged
   (`proprietary_boilerplate_in_title_block`) so you can exclude them if your counsel prefers. Sheets exported from
   student/educational CAD seats carry a vendor watermark ("SOLIDWORKS Educational Product…"); they are flagged
   (`cad_educational_watermark`, 136 sheets). The watermark is a CAD-EULA matter for whoever
   manufactures from the design, not a restriction on evaluating against the drawing, but it is also a realistic
   artefact your extractor should ignore.

## What this set is not — read before benchmarking

- **No NIST / NASA NTRS / DoD (DLA, DTIC) / Library of Congress HAER / Hugging Face content.** The build environment's
  network policy blocked those hosts; only public GitHub was reachable. NASA content is limited to NASA JPL's Open
  Source Rover. `tools/fetch_public_drawings.py` pulls those portals on your machine with the same license, marking
  and drawing gates, writing rows with the same core manifest fields. It was written against each portal's documented API but could not
  be exercised end to end from the build sandbox — run it with `--dry-run` first.
- **Schematic-heavy.** Openly licensed mechanical drawings are scarce on GitHub; schematics are not. Electrical
  schematics are 66% of the set; mechanical, PCB-mechanical, DXF and outline drawings are
  34%. Every mechanical sheet that passed the gates is in; filter on `category` for a mechanical-only benchmark.
- **Aerospace-flavoured, not aerospace-production.** Real flight-hardware drawings come from OreSat (Portland State
  Aerospace Society CubeSat), FOSSASAT, PyCubed and Pixhawk/PX4; the rest is precision mechanical,
  robotics, machine and enclosure drawings from professional design teams (Framework, System76, Foundation Devices,
  Prusa, Voron, Creality, ODRI). ASME Y14.5 GD&T density is lower than in prime-contractor drawings. For a defensible
  aerospace benchmark, top up with NIST MBE PMI test cases (FTC/CTC — the closest public analogue to aerospace
  production PMI) and HAER measured drawings of NASA facilities.
- **No human labels** beyond the embedded text layer. Category labels are rule-based from file names and source
  context and were spot-checked visually on random samples per source, not reviewed sheet by sheet.
- **Copyleft travels with the data.** GPL-3.0, CC BY-SA and CERN-OHL-S sheets are fine to evaluate on and to
  redistribute with attribution, but anything you *distribute* that adapts them inherits the license. Keep the set
  out of any shipped model artefact you would not want to license that way, and get counsel's view on training use.

## Reproducing

`tools/sources.py` lists every repository, include/exclude rule and license evidence. `tools/select.py`,
`fetch.py`, `lfs_fix.py`, `classify.py`, `build.py`, `package.py`, `slim.py`, `recompress.py`, `licenses.py`, `readme.py` rebuild the set from scratch
(set `DTS_WORK` to a scratch directory; requires git, poppler-utils, qpdf, ImageMagick, Python 3 with numpy, opencv,
pypdf, Pillow, matplotlib, ezdxf).
