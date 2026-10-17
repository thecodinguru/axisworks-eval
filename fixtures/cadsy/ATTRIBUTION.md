# Cadsy CAD Technical Drawings (subset)

Source: https://huggingface.co/datasets/cadsy/cad-technical-drawings
License: Apache License 2.0 (see `LICENSE` in this directory). The source CAD
geometry comes from ADSKAILab/Zero-To-CAD-100k
(https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-100k), also released
under Apache License 2.0.

This directory holds four of the 65 parts (`part_0001` to `part_0004`): the
ASME-profile and ISO-profile drawing PNGs (2480 x 1754) and their label files.
`part_0001` also carries `model.step`, the STEP model its drawings were
generated from (AP214, millimetres, 70 x 45 x 13 mm overall): it exercises the
CAD upload and the Feature Tree's 3D view against a drawing with the same
part. The other parts' STEP models are not included. The drawings are
synthetic and are not certified manufacturing drawings; they are used here
only as an evaluation set for box accuracy, OCR, revision comparison and the
3D view.

Fetch the full set with `python scripts/fetch_cadsy_dataset.py` (lands in
`datasets/cadsy`, which is gitignored); add `--with-step` for every part's STEP
model.
