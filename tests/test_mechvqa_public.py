"""The MechVQA track checks out from this repository alone: the manifest agrees with its
meta file and with the protocol's counts, the judge prompt hashes as pinned, and a report
built the way the runner builds one passes the check while a tampered one does not."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import re
from collections import Counter
from pathlib import Path

from axisworks_eval import mechvqa_judge as mj
from axisworks_eval import mechvqa_metrics as mv
from axisworks_eval.report import build_report
from axisworks_eval.verify import check_mechvqa, check_report

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "mechvqa"
PROTOCOL = ROOT / "PROTOCOL-MECHVQA.md"


def _manifest():
    return mv.load_manifest(FIXTURES), mv.load_manifest_meta(FIXTURES)


def test_the_manifest_counts_are_its_meta_file():
    rows, meta = _manifest()
    assert len(rows) == meta["questions"] == 1185
    assert len({r["image"] for r in rows}) == meta["images"] == 562
    assert dict(Counter(r["subcategory"] for r in rows)) == meta["by_subtask"]
    assert dict(Counter(r["capability"] for r in rows)) == meta["by_capability"]
    assert dict(Counter(r["difficulty"] for r in rows)) == meta["by_difficulty"]
    assert dict(Counter(r["language"] for r in rows)) == meta["by_language"]
    assert (
        sum(1 for r in rows if r["stated_capability"] != r["capability"])
        == meta["capability_mismatches"]
        == 2
    )
    assert all(r["capability"] == mv.SUBTASKS[r["subcategory"]] for r in rows)
    assert len({r["question_id"] for r in rows}) == len(rows)
    assert meta["commit"] == mv.COMMIT and meta["repo"] == mv.REPO
    assert meta["version"] == mv.VERSION


def _protocol_counts() -> dict[str, int]:
    """Every table row of the protocol: each label cell mapped to the row's last number."""
    counts: dict[str, int] = {}
    for line in PROTOCOL.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        numbers = [c for c in cells if re.fullmatch(r"[\d,]+", c)]
        if not numbers:
            continue
        value = int(numbers[-1].replace(",", ""))
        for cell in cells[:-1]:
            if cell and not re.fullmatch(r"[\d,]+", cell):
                counts[cell] = value
    return counts


def test_the_protocol_states_the_manifest_counts():
    rows, meta = _manifest()
    stated = _protocol_counts()
    for subtask, count in meta["by_subtask"].items():
        assert stated[subtask] == count, subtask
    for difficulty, count in meta["by_difficulty"].items():
        assert stated[difficulty] == count, difficulty
    for language, count in meta["by_language"].items():
        assert stated[language] == count, language
    for quality, count in Counter(str(r["qualityscore"]) for r in rows).items():
        assert stated[f"qualityscore {quality}"] == count, quality
    assert stated["all"] == meta["questions"]
    text = PROTOCOL.read_text(encoding="utf-8")
    assert meta["jsonl_sha256"] in text and mv.COMMIT in text
    assert hashlib.sha256((FIXTURES / mv.MANIFEST_NAME).read_bytes()).hexdigest() in text
    assert mj.JUDGE_PROMPT_SHA256 in text and mj.EVALUATOR_SHA256 in text
    assert "Amendments" in text


def test_the_judge_prompt_hashes_as_pinned():
    assert hashlib.sha256(mj.JUDGE_PROMPT_TEMPLATE.encode("utf-8")).hexdigest() == (
        mj.JUDGE_PROMPT_SHA256
    )
    prompt = mj.build_judge_prompt("Q?", "<answer>42</answer>", "42")
    assert "Reference answer:\n<answer>42</answer>" in prompt
    assert '{"score": 0 or 1, "reason": "short explanation"}' in prompt


def _vote(judge_id: str, score: float | None, parse_mode: str = "json", error=None) -> dict:
    return {
        "id": judge_id,
        "model": f"{judge_id}-model",
        "returned_model": f"{judge_id}-model-2026",
        "score": score,
        "reason": "",
        "parse_mode": parse_mode,
        "error": error,
    }


def _row(n: int, subcategory: str, language: str, answer: str, votes: list[dict], **extra):
    abstained = extra.pop("abstained", not answer)
    failed = extra.pop("failed", False)
    row = {
        "question_id": f"{n:016x}",
        "subcategory": subcategory,
        "capability": mv.SUBTASKS[subcategory],
        "stated_capability": mv.SUBTASKS[subcategory],
        "difficulty": "Hard" if n % 3 == 0 else "Easy",
        "language": language,
        "qualityscore": 1.0 if n % 4 else 0.75,
        "answer": answer,
        "answer_sha256": mv.sha256_text(answer),
        "abstained": abstained,
        "failed": failed,
        "answer_language": mv.language_of(answer),
        "numeric_exact": mv.numeric_exact("12.5 mm", answer),
        "judges": [] if abstained or failed else votes,
        "hit_token_limit": None,
        "finish_reason": None,
        "output_tokens": None,
        **extra,
    }
    row["score"] = 0.0 if abstained or failed else mv.vote(row["judges"])
    return row


def synthetic_report() -> dict:
    """What run_mechvqa writes for a small set: two configurations, three judges."""
    right = [_vote("j1", 1.0), _vote("j2", 1.0), _vote("j3", 0.0)]
    wrong = [_vote("j1", 0.0), _vote("j2", 0.0), _vote("j3", 1.0, "text_fallback")]
    split = [_vote("j1", 1.0), _vote("j2", 0.0), _vote("j3", None, "failed", "HTTP Error 500")]
    rows = [
        _row(1, "Dimension & Annotation", "en", "12.5 mm", right),
        _row(2, "Dimension & Annotation", "zh", "直径 12.5", wrong),
        _row(3, "Geometric Calculation", "en", "about 12", split),
        _row(4, "Anomaly Detection", "zh", "", []),
        _row(5, "Item Localization", "en", "top left", right),
        _row(6, "Consistency Judgment", "en", "", [], failed=True, abstained=False),
    ]
    record = [{**r, "judges": [], "answer": "", "abstained": True, "failed": False} for r in rows]
    for r in record:
        r["answer_sha256"] = mv.sha256_text("")
        r["answer_language"] = "und"
        r["numeric_exact"] = False
        r["score"] = 0.0
    judges = [
        {"id": j, "model": f"{j}-model", "host": "judges.example", "temperature": 0.1}
        for j in ("j1", "j2", "j3")
    ]
    block = {
        "dataset": {
            "repo": mv.REPO,
            "commit": mv.COMMIT,
            "benchmark_path": mv.BENCHMARK_PATH,
            "questions": 6,
            "images": 6,
            "capability_mismatches": 0,
        },
        "judges": judges,
        "judges_as_paper": False,
        "configurations": {
            "hosted": {
                "configuration": {"model": "hosted-model"},
                "subset": {"questions": 6, "of": 1185},
                "summary": mv.summarize(rows),
                "questions": rows,
            },
            "record": {
                "configuration": {"ask_model": "text-model"},
                "subset": {"questions": 6, "of": 1185},
                "summary": mv.summarize(record),
                "questions": record,
            },
        },
    }
    return build_report(
        {"mechvqa": block},
        root="datasets/mechvqa",
        parts=6,
        generated_at=dt.datetime(2026, 11, 7, tzinfo=dt.UTC),
        dataset={"name": "MechVQA", "repo": mv.REPO, "commit": mv.COMMIT},
    )


def test_a_report_built_like_the_runners_checks_out():
    report = synthetic_report()
    assert check_report(report) == []
    assert report["dataset"]["repo"] == mv.REPO and "root" in report["dataset"]
    head = report["headline"]
    assert head["mechvqa_total_hosted"] == round(
        2 / 6, 4
    )  # two right, a tie, an abstention, a failure
    assert head["mechvqa_abstained_hosted"] == round(1 / 6, 4)
    assert head["mechvqa_abstained_record"] == 1.0
    assert head["mechvqa_recognition_hosted"] == 0.75  # D&A 0.5, Item Localization 1.0
    assert "mechvqa_hard_hosted" in head
    gaps = " ".join(report["not_measured"])
    assert "judges j1, j2, j3" in gaps and "Table 2" in gaps and "abstains by design" in gaps
    assert "Chinese" in gaps and "textbook" in gaps


def test_a_tampered_score_or_cell_is_named():
    report = synthetic_report()
    tampered = copy.deepcopy(report)
    row = tampered["details"]["mechvqa"]["configurations"]["hosted"]["questions"][1]
    row["score"] = 1.0
    problems = check_report(tampered)
    assert any(p.startswith(f"mechvqa hosted {row['question_id']} score") for p in problems)

    tampered = copy.deepcopy(report)
    summary = tampered["details"]["mechvqa"]["configurations"]["hosted"]["summary"]
    summary["by_subtask"]["Dimension & Annotation"]["score"] = 0.9
    assert any(
        p.startswith("mechvqa hosted summary by_subtask Dimension & Annotation")
        for p in check_report(tampered)
    )

    tampered = copy.deepcopy(report)
    tampered["headline"]["mechvqa_total_hosted"] = 0.9
    assert any(p.startswith("headline") for p in check_report(tampered))


def test_a_synthetic_report_is_not_the_manifests_questions():
    """With the manifest at hand, rows the set never had and a subset that lies are named."""
    report = synthetic_report()
    problems = check_mechvqa(report, fixture_dir=FIXTURES)
    assert any(p.endswith(f"not in {mv.MANIFEST_NAME}") for p in problems)
    assert any(p.startswith("mechvqa dataset questions") for p in problems)
