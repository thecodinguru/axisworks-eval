"""Check a published accuracy report against itself and against the public labels.

Two checks anyone can run without the product:

- ``check_report``: every summary and headline number is recomputed from the
  per-part results the report carries, with the arithmetic the evaluation
  used. A number that does not follow from its details is named.
- ``check_truth``: with the Cadsy labels at hand, the truth each result was
  scored against (requirement rows per sheet, the seeded edits, the twin
  pairs, the labelled characteristics) is rebuilt from the labels and compared
  with the counts the report states.

Neither can check the product's own output: which rows the matcher paired,
which numbers the balloon carry gave, what the model read. Those come from
the private code and are published as counts.

The open-hardware test set is the exception: its reads are listed sheet by
sheet, so ``check_text_layer`` rebuilds every test-set score from the PDFs in
``fixtures/drawing-testset-v1`` and those reads.

A MechVQA report lists every answer and every judge's vote, so
``check_mechvqa`` recomputes each question's score from the votes, every
summary cell from the rows, and, with the committed manifest, that no
question was dropped or counted twice; with the fetched set it also rebuilds
the judge-free numeric check from the reference answers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import axisworks_eval.mechvqa_metrics as mv
from axisworks_eval.balloon_metrics import (
    SCENARIOS,
    BalloonResult,
    move_top_view,
    summarize,
    twin_truth,
)
from axisworks_eval.cadsy_adapter import load_labels, load_rows
from axisworks_eval.extraction_metrics import (
    CHAR_TYPES,
    TypeExtraction,
    rows_of,
    summarize_provenance,
    truth_annotations,
)
from axisworks_eval.holdout import FILE_NAME as HOLDOUT_FILE
from axisworks_eval.holdout import check as check_split
from axisworks_eval.holdout import load as load_holdout
from axisworks_eval.holdout import sha256_of
from axisworks_eval.real_metrics import ValueScore, score_entry
from axisworks_eval.real_metrics import summarize as summarize_holdout
from axisworks_eval.report import headline, not_measured
from axisworks_eval.revision_metrics import (
    PerturbationScore,
    perturb,
    requirements,
    summarize_perturbations,
    summarize_twins,
)
from axisworks_eval.text_layer_metrics import (
    TextLayerScore,
    read_words,
    reads_from_details,
    score_sheet,
)
from axisworks_eval.text_layer_metrics import summarize as summarize_text_layer

#: The seed every published perturbation used.
PERTURBATION_SEED = 7


def _same(problems: list[str], what: str, stated: Any, recomputed: Any) -> None:
    if stated != recomputed:
        problems.append(f"{what}: the report says {stated!r}, its details give {recomputed!r}")


def _type_sum(blocks: list[dict[str, Any]]) -> TypeExtraction:
    total = TypeExtraction()
    for block in blocks:
        total.add(TypeExtraction.from_dict(block))
    return total


def _check_extraction(problems: list[str], drawing_class: str, data: dict[str, Any]) -> None:
    profiles = data.get("by_profile") or {}
    for profile, block in profiles.items():
        for char_type, counts in (block.get("by_type") or {}).items():
            where = f"live_extraction {drawing_class} {profile} {char_type}"
            _same(problems, where, counts, TypeExtraction.from_dict(counts).to_dict())
        _same(
            problems,
            f"live_extraction {drawing_class} {profile} overall",
            block.get("overall"),
            _type_sum(list((block.get("by_type") or {}).values())).to_dict(),
        )
    for char_type in CHAR_TYPES:
        pooled = _type_sum(
            [
                b["by_type"][char_type]
                for b in profiles.values()
                if char_type in (b.get("by_type") or {})
            ]
        )
        _same(
            problems,
            f"live_extraction {drawing_class} {char_type}",
            (data.get("by_type") or {}).get(char_type),
            pooled.to_dict(),
        )
    _same(
        problems,
        f"live_extraction {drawing_class} overall",
        data.get("overall"),
        _type_sum(list((data.get("by_type") or {}).values())).to_dict(),
    )


def check_report(report: dict[str, Any]) -> list[str]:
    """Every number the report derives, recomputed from the details it carries."""
    problems: list[str] = []
    details = report.get("details") or {}

    twins = details.get("twin_gate") or []
    for part in twins:
        smaller = min(part["old_rows"], part["new_rows"])
        fraction = round(part["matched"] / smaller, 3) if smaller else 0.0
        _same(
            problems,
            f"twin_gate {part['part_id']} matched_fraction",
            part["matched_fraction"],
            fraction,
        )
    if twins:
        _same(
            problems, "twin_gate_summary", details.get("twin_gate_summary"), summarize_twins(twins)
        )

    scores = details.get("perturbation") or []
    for part_id, score in zip(details.get("parts") or [], scores, strict=False):
        _same(
            problems, f"perturbation {part_id}", score, PerturbationScore.from_dict(score).to_dict()
        )
    if scores:
        _same(
            problems,
            "perturbation_summary",
            details.get("perturbation_summary"),
            summarize_perturbations(scores),
        )

    balloons = details.get("balloons") or {}
    for scenario in SCENARIOS:
        if scenario not in balloons:
            continue
        results = [BalloonResult.from_dict(r) for r in balloons[scenario]]
        for stated, result in zip(balloons[scenario], results, strict=True):
            _same(problems, f"balloons {scenario} {result.part_id}", stated, result.to_dict())
        _same(
            problems,
            f"balloons summary {scenario}",
            (balloons.get("summary") or {}).get(scenario),
            summarize(results),
        )

    for drawing_class, data in (
        (details.get("live_extraction") or {}).get("classes") or {}
    ).items():
        _check_extraction(problems, drawing_class, data)

    text_layer = details.get("text_layer") or {}
    if text_layer:
        sheets = text_layer.get("sheets") or []
        for sheet in sheets:
            if "score" in sheet:
                _same(
                    problems,
                    f"text_layer {sheet['sheet_id']}",
                    sheet["score"],
                    TextLayerScore.from_dict(sheet["score"]).to_dict(),
                )
        recomputed = summarize_text_layer(sheets)
        _same(
            problems,
            "text_layer by_category",
            text_layer.get("by_category"),
            recomputed["by_category"],
        )
        _same(problems, "text_layer overall", text_layer.get("overall"), recomputed["overall"])

    provenance = details.get("provenance") or {}
    if provenance:
        recomputed = summarize_provenance(provenance.get("sheets") or [])
        _same(problems, "provenance by_class", provenance.get("by_class"), recomputed["by_class"])

    held = details.get("holdout") or {}
    if held:
        sheets = held.get("sheets") or []
        for sheet in sheets:
            where = f"holdout {sheet['sheet_id']} {sheet.get('class', 'clean')}"
            if "text" in sheet:
                _same(
                    problems,
                    f"{where} text",
                    sheet["text"],
                    TextLayerScore.from_dict(sheet["text"]).to_dict(),
                )
            if "values" in sheet:
                _same(
                    problems,
                    f"{where} values",
                    sheet["values"],
                    ValueScore.from_dict(sheet["values"]).to_dict(),
                )
        recomputed = summarize_holdout(sheets)
        _same(problems, "holdout by_class", held.get("by_class"), recomputed["by_class"])
        _same(problems, "holdout overall", held.get("overall"), recomputed["overall"])

    if details.get("mechvqa"):
        problems += check_mechvqa(report)

    _same(problems, "headline", report.get("headline"), headline(details))
    _same(problems, "not_measured", report.get("not_measured"), not_measured(details))
    return problems


#: The per-question fields a manifest row fixes; a report row must agree with it.
_MECHVQA_FIXED = ("subcategory", "capability", "difficulty", "language", "qualityscore")


def _check_mechvqa_rows(problems: list[str], name: str, rows: list[dict[str, Any]]) -> None:
    """Each row's derived fields from what it carries: the vote, the answer hash and language."""
    for row in rows:
        where = f"mechvqa {name} {row.get('question_id')}"
        answer = str(row.get("answer") or "")
        if row.get("abstained") or row.get("failed"):
            expected = 0.0
        else:
            expected = mv.vote(list(row.get("judges") or []))
        _same(problems, f"{where} score", row.get("score"), expected)
        if "answer_sha256" in row:
            _same(problems, f"{where} answer_sha256", row["answer_sha256"], mv.sha256_text(answer))
        if "answer_language" in row:
            _same(
                problems, f"{where} answer_language", row["answer_language"], mv.language_of(answer)
            )


def _check_mechvqa_summary(problems: list[str], name: str, config: dict[str, Any]) -> None:
    """Every cell of the configuration's summary from its rows, bands included."""
    stated = config.get("summary") or {}
    recomputed = mv.summarize(list(config.get("questions") or []))
    for key in sorted(set(stated) | set(recomputed)):
        a, b = stated.get(key), recomputed.get(key)
        if isinstance(a, dict) and isinstance(b, dict) and key.startswith("by_"):
            for cell_name in sorted(set(a) | set(b)):
                _same(
                    problems,
                    f"mechvqa {name} summary {key} {cell_name}",
                    a.get(cell_name),
                    b.get(cell_name),
                )
        else:
            _same(problems, f"mechvqa {name} summary {key}", a, b)


def _check_mechvqa_manifest(
    problems: list[str], block: dict[str, Any], fixture_dir: Path
) -> dict[str, dict[str, Any]]:
    manifest = {r["question_id"]: r for r in mv.load_manifest(fixture_dir)}
    meta = mv.load_manifest_meta(fixture_dir)
    dataset = block.get("dataset") or {}
    for key in ("commit", "jsonl_sha256", "questions", "images"):
        if key in dataset:
            _same(problems, f"mechvqa dataset {key}", dataset[key], meta.get(key))
    if "manifest_sha256" in dataset:
        _same(
            problems,
            "mechvqa dataset manifest_sha256",
            dataset["manifest_sha256"],
            sha256_of(fixture_dir / mv.MANIFEST_NAME),
        )
    for name, config in (block.get("configurations") or {}).items():
        rows = list(config.get("questions") or [])
        seen: dict[str, int] = {}
        for row in rows:
            qid = str(row.get("question_id"))
            seen[qid] = seen.get(qid, 0) + 1
            fixed = manifest.get(qid)
            if fixed is None:
                problems.append(f"mechvqa {name} {qid}: not in {mv.MANIFEST_NAME}")
                continue
            for key in _MECHVQA_FIXED:
                _same(problems, f"mechvqa {name} {qid} {key}", row.get(key), fixed.get(key))
            if "answer_sha256" in fixed and "reference_sha256" in row:
                _same(
                    problems,
                    f"mechvqa {name} {qid} reference_sha256",
                    row["reference_sha256"],
                    fixed["answer_sha256"],
                )
        for qid, count in seen.items():
            if count > 1:
                problems.append(f"mechvqa {name} {qid}: appears {count} times")
        subset = config.get("subset")
        if subset:
            _same(problems, f"mechvqa {name} subset questions", subset.get("questions"), len(rows))
        else:
            dropped = sorted(set(manifest) - set(seen))
            problems += [f"mechvqa {name} {qid}: dropped from the configuration" for qid in dropped]
    return manifest


def check_mechvqa(
    report: dict[str, Any],
    root: str | Path | None = None,
    fixture_dir: str | Path | None = None,
) -> list[str]:
    """The MechVQA block against itself, the committed manifest and the fetched set.

    Alone, every question's score is recomputed from the judges' votes (an
    abstained or failed row is 0) and every summary cell from the rows. With
    ``fixture_dir`` (the committed ``manifest.jsonl`` and ``MANIFEST.json``)
    the dataset block must name the same pin and hashes, every stored
    question must be in the manifest with the manifest's subtask, difficulty,
    language and quality, and every manifest question must appear exactly once
    in each configuration unless the configuration declares the subset it ran.
    With ``root`` (the fetched set) the manifest must rebuild from it and the
    judge-free ``numeric_exact`` of every row is recomputed from the reference
    answer. The headline and ``not_measured`` are ``check_report``'s.
    """
    problems: list[str] = []
    block = (report.get("details") or {}).get("mechvqa") or {}
    if not block:
        return problems
    configurations = block.get("configurations") or {}
    for name, config in configurations.items():
        _check_mechvqa_rows(problems, name, list(config.get("questions") or []))
        _check_mechvqa_summary(problems, name, config)
    if fixture_dir is not None:
        _check_mechvqa_manifest(problems, block, Path(fixture_dir))
    if root is not None:
        root = Path(root)
        if fixture_dir is not None:
            problems += [f"mechvqa manifest: {p}" for p in mv.check(root, fixture_dir)]
        if (root / mv.BENCHMARK_PATH).is_file():
            references = {q.question_id: q.answer for q in mv.load(root)}
            for name, config in configurations.items():
                for row in config.get("questions") or []:
                    qid = str(row.get("question_id"))
                    if qid not in references:
                        problems.append(f"mechvqa {name} {qid}: not in the fetched set")
                        continue
                    _same(
                        problems,
                        f"mechvqa {name} {qid} numeric_exact",
                        row.get("numeric_exact"),
                        mv.numeric_exact(references[qid], str(row.get("answer") or "")),
                    )
        else:
            problems.append(f"mechvqa: {mv.BENCHMARK_PATH} is not under {root}")
    return problems


def check_truth(
    report: dict[str, Any],
    root: str | Path,
    *,
    only: Iterable[str] | None = None,
    labels_of: Callable[[Path, str], Path] = lambda part_dir, profile: (
        part_dir / f"{profile}_labels.json"
    ),
) -> list[str]:
    """The truth behind each result, rebuilt from the Cadsy labels under ``root``.

    ``only`` checks those parts alone (the four-part fixture subset, say); the
    extraction totals, which pool every part, are then not checked.
    """
    problems: list[str] = []
    details = report.get("details") or {}
    base = Path(root)
    every = details.get("parts") or []
    parts = [p for p in every if p in set(only)] if only is not None else every
    twins = {t["part_id"]: t for t in details.get("twin_gate") or []}
    perturbations = dict(zip(parts, details.get("perturbation") or [], strict=False))
    balloons = {
        scenario: {r["part_id"]: r for r in (details.get("balloons") or {}).get(scenario) or []}
        for scenario in SCENARIOS
    }
    labelled = {char_type: 0 for char_type in CHAR_TYPES}
    # The live reads may cover only some parts; their truth is pooled over those.
    live_parts = (details.get("live_extraction") or {}).get("parts") or every

    for part_id in parts:
        part_dir = base / part_id
        if not part_dir.is_dir():
            problems.append(f"{part_id}: not under {base}")
            continue
        asme = requirements(load_rows(labels_of(part_dir, "asme"), prefix=f"{part_id}-asme"))
        iso = requirements(load_rows(labels_of(part_dir, "iso"), prefix=f"{part_id}-iso"))

        if part_id in twins:
            _same(problems, f"twin_gate {part_id} old_rows", twins[part_id]["old_rows"], len(asme))
            _same(problems, f"twin_gate {part_id} new_rows", twins[part_id]["new_rows"], len(iso))

        edited, truth = perturb(asme, seed=PERTURBATION_SEED)
        if part_id in perturbations:
            for kind, ids in (
                ("modified", truth.modified),
                ("removed", truth.removed),
                ("added", truth.added),
            ):
                _same(
                    problems,
                    f"perturbation {part_id} {kind} truth",
                    perturbations[part_id][kind]["truth"],
                    len(ids),
                )
            scope = perturbations[part_id].get("delta_scope")
            if scope is not None:
                _same(
                    problems,
                    f"perturbation {part_id} delta_scope truth",
                    scope["truth"],
                    len(truth.modified | truth.added),
                )

        if part_id in balloons["twins"]:
            pairs = len(twin_truth(asme, iso))
            for method, score in balloons["twins"][part_id]["methods"].items():
                _same(problems, f"balloons twins {part_id} {method} truth", score["truth"], pairs)
        if part_id in balloons["seeded"]:
            moved, view = move_top_view(edited)
            old_ids = {r.id for r in asme}
            kept = sum(1 for r in moved if r.id in old_ids)
            added = sum(1 for r in moved if r.id not in old_ids)
            result = balloons["seeded"][part_id]
            _same(
                problems,
                f"balloons seeded {part_id} moved_view",
                result.get("moved_view", ""),
                view,
            )
            for method, score in result["methods"].items():
                _same(problems, f"balloons seeded {part_id} {method} truth", score["truth"], kept)
                _same(
                    problems, f"balloons seeded {part_id} {method} added", score.get("added"), added
                )

        for profile in ("asme", "iso") if part_id in live_parts else ():
            for row in rows_of(
                truth_annotations(load_labels(labels_of(part_dir, profile))), "truth"
            ):
                labelled[row.char_type] += 1

    if not set(live_parts) <= set(parts):
        return problems
    for drawing_class, data in (
        (details.get("live_extraction") or {}).get("classes") or {}
    ).items():
        if data.get("failed"):
            continue  # a failed read leaves its sheet out of the truth too
        for char_type in CHAR_TYPES:
            stated = ((data.get("by_type") or {}).get(char_type) or {}).get("truth")
            _same(
                problems,
                f"live_extraction {drawing_class} {char_type} truth",
                stated,
                labelled[char_type],
            )
    return problems


def check_text_layer(report: dict[str, Any], root: str | Path) -> list[str]:
    """Every test-set score rebuilt from the PDF under ``root`` and the reads the report lists."""
    problems: list[str] = []
    base = Path(root)
    for sheet in (report.get("details") or {}).get("text_layer", {}).get("sheets") or []:
        if sheet.get("failed"):
            continue
        pdf = base / sheet["file"]
        if not pdf.is_file():
            problems.append(f"text_layer {sheet['sheet_id']}: {sheet['file']} is not under {base}")
            continue
        words = read_words(pdf)
        where = f"text_layer {sheet['sheet_id']}"
        _same(problems, f"{where} words", sheet.get("words"), len(words))
        rebuilt = score_sheet(words, reads_from_details(sheet.get("reads") or []))
        _same(problems, f"{where} from the PDF", sheet.get("score"), rebuilt.to_dict())
    return problems


def check_holdout(report: dict[str, Any], root: str | Path) -> list[str]:
    """The held-out section rebuilt from the sealed split and the vectors under ``root``.

    The committed split must still follow from the manifest and carry the
    hash the report names; every sheet must be on the side the report says;
    and each sheet's text-layer score, CAD truth and value score are rebuilt
    from its PDF or DXF and the reads the report lists.
    """
    base = Path(root)
    section = (report.get("details") or {}).get("holdout") or {}
    if not section:
        return []
    problems = check_split(base)
    stated = (section.get("set") or {}).get("sha256")
    path = base / HOLDOUT_FILE
    if stated and path.is_file() and stated != sha256_of(path):
        problems.append(
            f"holdout: the report names {HOLDOUT_FILE} {stated[:12]}…, the file differs"
        )
    if not path.is_file():
        return problems
    committed = {r["sheet_id"]: r for r in load_holdout(base)}
    for sheet in section.get("sheets") or []:
        if sheet.get("failed"):
            continue
        where = f"holdout {sheet['sheet_id']} {sheet.get('class', 'clean')}"
        row = committed.get(sheet["sheet_id"])
        if row is None:
            problems.append(f"{where}: not in {HOLDOUT_FILE}")
            continue
        _same(problems, f"{where} split", sheet.get("split"), row["split"])
        rebuilt = score_entry(
            base,
            {"file": row["file_vector"], "alt_formats": row.get("alt_formats") or []},
            reads_from_details(sheet.get("reads") or []),
        )
        for key in ("words", "text", "cad_truth", "values"):
            _same(problems, f"{where} {key} from the vectors", sheet.get(key), rebuilt.get(key))
    return problems
