"""MechVQA as a measurement track: the public set pinned, scored the way its authors score it.

MechVQA (ICML 2026, arXiv 2605.30794) is a public visual question answering
benchmark over mechanical engineering drawings: 1,185 questions on 562 drawing
images, each with a reference answer, a difficulty (Easy, Medium, Hard), a
language (中文 or 英文) and one of ten subtasks grouped under three capabilities
(Recognition, Reasoning, Judging). Nobody here wrote a question or tuned a
rule against one, so it is the one external yardstick a reader can run without
trusting us: fetch the repository at ``COMMIT``, point ``load`` at it, and the
manifest committed beside the reports must come out byte for byte (``check``).

What this module copies from the repository's own evaluator
(``evaluation/mechvqa_eval/evaluate_vqa.py`` at the pinned commit):

* ``extract_answer``: the first closed ``<answer>…</answer>``, else an
  unclosed ``<answer>`` to the end, else the text after ``</think>``, else the
  whole response. The reference answer passes through the same function.
* ``parse_judge_response``: the first flat JSON object in the judge's reply,
  its ``score`` clamped to [0, 1]; failing that, a ``1`` with no ``0`` is 1.0
  and any ``0`` is 0.0; failing that, 0.0.
* The vote: scores rounded to one decimal, the plurality wins.
* The aggregates ``total`` (mean score) and ``full_score_rate``.

Every departure from that evaluator:

1. A judge reply that parses neither as JSON nor as a bare 1/0 is an
   **invalid** vote and is dropped (the evaluator counts it as a valid 0).
   An empty judge reply is invalid for the same reason.
2. A **tie** between the most common rounded scores is 0.0 (the evaluator's
   tie goes to whichever judge happens to be listed first).
3. An **empty answer** after extraction is 0.0 without judging and is counted
   as abstained, so refusals are visible instead of being judged as wrong.
4. A **capability** score is the unweighted mean of its subtasks' scores (the
   paper's convention; the evaluator reports only the overall mean), so the
   445 Dimension & Annotation questions do not swamp Recognition.
5. Two records carry ``Judging`` for Geometric Calculation, which the paper's
   taxonomy puts under Reasoning. A row keeps the record's stated capability,
   the capability means use the taxonomy, and the manifest counts the
   mismatches.
6. Added, not in the evaluator: 95 % bootstrap bands over questions
   (``random.Random(BOOTSTRAP_SEED)``, ``BOOTSTRAP_DRAWS`` draws), the worst
   subtask among those with at least ``WORST_MIN_QUESTIONS`` questions, a
   judge-free ``numeric_exact`` (every number of the reference, NFKC-folded so
   full-width digits count, appears in the answer), the answer-language match
   rate, the token-limit rate and per-judge agreement with the vote.
7. Questions are cleaned defensively: the SFT release prefixes ``<image>``
   and appends the ``<think>/<answer>`` instruction block, the benchmark
   release does neither, and both are stripped before hashing so a question
   id is the same in either. Nine records repeat an earlier record's image
   and question (with another reference answer or difficulty); a repeat's id
   hashes its occurrence number in, so every record has its own id and
   ``load`` refuses a set whose ids still collide.

``language_of`` reads an answer's language from the share of CJK letters among
its letters (0.3 or more is ``zh``); an answer with no letters at all (a bare
number) is ``und`` and counts as matching the question's language.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = "xiaofengShi/MechVQA"
COMMIT = "c8d61513aef05b4f5d117bb0c6d1a055451d6f33"
VERSION = "mechvqa_public_v1"
BENCHMARK_PATH = "benchmark_data/vqa_benchmark/mechvqa_benchmark.jsonl"
#: Image paths in the records are relative to this folder of the repository.
IMAGE_DIR = "benchmark_data"
MANIFEST_NAME = "manifest.jsonl"
MANIFEST_META = "MANIFEST.json"

CAPABILITIES = ("Recognition", "Reasoning", "Judging")
#: The paper's ten subtasks, verbatim, and the capability each belongs to.
SUBTASKS: dict[str, str] = {
    "Identification & Counting": "Recognition",
    "Dimension & Annotation": "Recognition",
    "Text & Table": "Recognition",
    "Item Localization": "Recognition",
    "Structure Understanding": "Reasoning",
    "Geometric Calculation": "Reasoning",
    "Assembly Relationship": "Reasoning",
    "Projection & Multi-view": "Reasoning",
    "Anomaly Detection": "Judging",
    "Consistency Judgment": "Judging",
}
LANGUAGES = {"中文": "zh", "英文": "en"}
DIFFICULTIES = ("Easy", "Medium", "Hard")
WORST_MIN_QUESTIONS = 20
BOOTSTRAP_SEED = 7
BOOTSTRAP_DRAWS = 1000
#: The SFT release's image placeholder and the first words of its answer-format suffix.
IMAGE_PLACEHOLDER = "<image>"
ANSWER_SUFFIX_MARK = "A conversation between User and Assistant"
#: What a manifest row carries, in this order.
MANIFEST_FIELDS = (
    "question_id",
    "index",
    "image",
    "image_sha256",
    "question_sha256",
    "answer_sha256",
    "capability",
    "stated_capability",
    "subcategory",
    "difficulty",
    "language",
    "qualityscore",
)
#: The share of a CJK answer's letters that are CJK, at or above which it reads as Chinese.
CJK_RATIO = 0.3
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿぀-ヿ]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)
_ANSWER_CLOSED = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_ANSWER_OPEN = re.compile(r"<answer>(.*)", re.DOTALL | re.IGNORECASE)
_AFTER_THINK = re.compile(r"</think>(.*)", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Question:
    question_id: str
    index: int
    image: str
    question: str
    answer: str
    capability: str
    stated_capability: str
    subcategory: str
    difficulty: str
    language: str
    language_raw: str
    qualityscore: float
    explanation: str


@dataclass
class JudgeParse:
    score: float
    reason: str
    parse_mode: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- the set ---------------------------------------------------------------


def bare_question(text: str) -> str:
    """The question without the SFT release's ``<image>`` placeholder and answer-format suffix."""
    text = (text or "").replace(IMAGE_PLACEHOLDER, "")
    at = text.find(ANSWER_SUFFIX_MARK)
    if at >= 0:
        text = text[:at]
    return text.strip()


def question_id(image: str, bare: str, occurrence: int = 0) -> str:
    """Sixteen hex digits of the image path and the bare question, stable across releases.

    Nine records of the public set repeat an earlier record's image and
    question with a different reference answer or difficulty, so the pair
    alone is not a key: a repeat hashes its occurrence number in as well
    (the first occurrence is the plain pair, as before).
    """
    text = f"{image}\n{bare}" + (f"\n{occurrence}" if occurrence else "")
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _user_content(record: dict[str, Any]) -> str:
    for message in record.get("messages") or []:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return str(record.get("question") or "")


def _assistant_content(record: dict[str, Any]) -> str:
    for message in record.get("messages") or []:
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return str(record.get("answer") or "")


def _question(index: int, record: dict[str, Any], occurrence: int = 0) -> Question:
    meta = record.get("metadata") or {}
    subcategory = str(meta.get("subcategory") or "")
    if subcategory not in SUBTASKS:
        raise ValueError(f"record {index}: unknown subcategory {subcategory!r}")
    stated = str(meta.get("capability") or "")
    if stated not in CAPABILITIES:
        raise ValueError(f"record {index}: unknown capability {stated!r}")
    language_raw = str(meta.get("language") or "")
    if language_raw not in LANGUAGES:
        raise ValueError(f"record {index}: unknown language {language_raw!r}")
    difficulty = str(meta.get("difficulty") or "")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"record {index}: unknown difficulty {difficulty!r}")
    images = record.get("images") or []
    if isinstance(images, str):
        images = [images]
    if len(images) != 1:
        raise ValueError(f"record {index}: {len(images)} images, expected one")
    image = str(images[0])
    bare = bare_question(_user_content(record))
    return Question(
        question_id=question_id(image, bare, occurrence),
        index=index,
        image=image,
        question=bare,
        answer=_assistant_content(record),
        capability=SUBTASKS[subcategory],
        stated_capability=stated,
        subcategory=subcategory,
        difficulty=difficulty,
        language=LANGUAGES[language_raw],
        language_raw=language_raw,
        qualityscore=float(record.get("qualityscore") or 0.0),
        explanation=str(meta.get("explanation") or ""),
    )


def load(root: str | Path) -> list[Question]:
    """Every question of the benchmark file under ``root``, in file order."""
    path = Path(root) / BENCHMARK_PATH
    out: list[Question] = []
    seen: Counter[tuple[str, str]] = Counter()
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(line for line in handle if line.strip()):
            record = json.loads(line)
            images = record.get("images") or []
            image = str(images if isinstance(images, str) else (images or [""])[0])
            pair = (image, bare_question(_user_content(record)))
            out.append(_question(index, record, seen[pair]))
            seen[pair] += 1
    ids = Counter(q.question_id for q in out)
    if len(ids) != len(out):
        clashes = sorted(qid for qid, n in ids.items() if n > 1)
        raise ValueError(f"question ids collide: {', '.join(clashes[:5])}")
    return out


def manifest_rows(questions: Iterable[Question], root: str | Path) -> list[dict[str, Any]]:
    """One row per question with the hashes a reader needs to prove the set is the same."""
    root = Path(root)
    image_hashes: dict[str, str] = {}
    rows = []
    for q in questions:
        if q.image not in image_hashes:
            image_hashes[q.image] = sha256_of(root / IMAGE_DIR / q.image)
        rows.append(
            {
                "question_id": q.question_id,
                "index": q.index,
                "image": q.image,
                "image_sha256": image_hashes[q.image],
                "question_sha256": sha256_text(q.question),
                "answer_sha256": sha256_text(q.answer),
                "capability": q.capability,
                "stated_capability": q.stated_capability,
                "subcategory": q.subcategory,
                "difficulty": q.difficulty,
                "language": q.language,
                "qualityscore": q.qualityscore,
            }
        )
    return rows


def dumps(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n" for row in rows)


def dumps_meta(meta: dict[str, Any]) -> str:
    return json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def _counts(rows: Iterable[dict[str, Any]], key: str, order: Sequence[str]) -> dict[str, int]:
    counter = Counter(str(r.get(key)) for r in rows)
    names = [n for n in order if n in counter] + sorted(n for n in counter if n not in order)
    return {name: counter[name] for name in names}


def manifest_meta(rows: Sequence[dict[str, Any]], root: str | Path) -> dict[str, Any]:
    """What the README states about the set: where it came from, its hash and its counts."""
    return {
        "version": VERSION,
        "repo": REPO,
        "commit": COMMIT,
        "benchmark_path": BENCHMARK_PATH,
        "jsonl_sha256": sha256_of(Path(root) / BENCHMARK_PATH),
        "questions": len(rows),
        "images": len({r["image"] for r in rows}),
        "by_subtask": _counts(rows, "subcategory", list(SUBTASKS)),
        "by_capability": _counts(rows, "capability", CAPABILITIES),
        "by_difficulty": _counts(rows, "difficulty", DIFFICULTIES),
        "by_language": _counts(rows, "language", list(LANGUAGES.values())),
        "capability_mismatches": sum(
            1 for r in rows if r.get("stated_capability") != r.get("capability")
        ),
    }


def load_manifest(fixture_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(fixture_dir) / MANIFEST_NAME
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_manifest_meta(fixture_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(fixture_dir) / MANIFEST_META).read_text(encoding="utf-8"))


def check(root: str | Path, fixture_dir: str | Path) -> list[str]:
    """The committed manifest against one rebuilt from the fetched set; problems, if any."""
    root, fixture_dir = Path(root), Path(fixture_dir)
    if not (root / BENCHMARK_PATH).is_file():
        return [f"{BENCHMARK_PATH} is not under {root}"]
    rows = manifest_rows(load(root), root)
    problems = []
    path = fixture_dir / MANIFEST_NAME
    if not path.is_file():
        problems.append(f"{MANIFEST_NAME} is not under {fixture_dir}")
    else:
        expected = dumps(rows)
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            before = {
                json.loads(line)["question_id"]: line
                for line in actual.splitlines()
                if line.strip()
            }
            after = {
                json.loads(line)["question_id"]: line
                for line in expected.splitlines()
                if line.strip()
            }
            differing = [
                f"{MANIFEST_NAME}: {qid} differs from the fetched set"
                for qid in sorted(set(before) | set(after))
                if before.get(qid) != after.get(qid)
            ]
            problems.extend(differing or [f"{MANIFEST_NAME}: differs from the fetched set"])
    meta_path = fixture_dir / MANIFEST_META
    if not meta_path.is_file():
        problems.append(f"{MANIFEST_META} is not under {fixture_dir}")
    else:
        expected_meta = manifest_meta(rows, root)
        actual_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in sorted(set(expected_meta) | set(actual_meta)):
            if expected_meta.get(key) != actual_meta.get(key):
                problems.append(f"{MANIFEST_META}: {key} differs from the fetched set")
    return problems


# --- the evaluator's scoring, copied ----------------------------------------


def extract_answer(text: str) -> str:
    """The evaluator's ``extract_answer``: the answer tag's body, else after the thinking, else all."""
    if not text:
        return ""
    match = _ANSWER_CLOSED.search(text)
    if match:
        return match.group(1).strip()
    match = _ANSWER_OPEN.search(text)
    if match:
        return match.group(1).strip()
    match = _AFTER_THINK.search(text)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return text.strip()


def parse_judge_response(text: str) -> JudgeParse:
    """The evaluator's ``parse_judge_response``, with the branch it took named in ``parse_mode``."""
    if not text or not text.strip():
        return JudgeParse(0.0, "empty judge response", "failed")
    match = _JSON_OBJECT.search(text)
    if match:
        try:
            payload = json.loads(match.group(0))
            score = max(0.0, min(1.0, float(payload.get("score", 0))))
            return JudgeParse(score, str(payload.get("reason", "")), "json")
        except Exception:
            pass
    if "1" in text and "0" not in text:
        return JudgeParse(1.0, "parsed as 1", "text_fallback")
    if "0" in text:
        return JudgeParse(0.0, "parsed as 0", "text_fallback")
    return JudgeParse(0.0, f"parse failed: {text[:120]}", "failed")


def vote(votes: list[dict[str, Any]]) -> float:
    """The plurality of the valid judges' scores, rounded to one decimal.

    A vote is valid when the judge answered (no ``error``) and its reply parsed
    (``parse_mode`` other than ``failed``). No valid vote is 0.0; so is a tie
    between the most common scores, since nobody won.
    """
    valid = [
        round(float(v["score"]), 1)
        for v in votes
        if not v.get("error") and v.get("parse_mode") != "failed" and v.get("score") is not None
    ]
    if not valid:
        return 0.0
    ranked = Counter(valid).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return 0.0
    return float(ranked[0][0])


# --- the judge-free checks ----------------------------------------------------


def _canonical_number(token: str) -> str:
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    token = token.lstrip("0")
    return token or "0"


def numbers_in(text: str) -> list[str]:
    """The numbers in ``text`` in order, NFKC-folded so full-width digits count, as canonical
    strings (``12.50`` and ``１２.５`` both read ``12.5``)."""
    folded = unicodedata.normalize("NFKC", text or "")
    return [_canonical_number(m.group(0)) for m in _NUMBER.finditer(folded)]


def numeric_exact(reference: str, answer: str) -> bool:
    """Whether every number of the reference appears in the answer; False when the reference
    carries no number, since there is nothing to match."""
    wanted = set(numbers_in(reference))
    if not wanted:
        return False
    return wanted <= set(numbers_in(answer))


def language_of(text: str) -> str:
    """``zh`` when at least ``CJK_RATIO`` of the letters are CJK, ``en`` otherwise, ``und`` when
    the text has no letters at all."""
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if not letters:
        return "und"
    cjk = sum(1 for ch in letters if _CJK.match(ch))
    return "zh" if cjk / len(letters) >= CJK_RATIO else "en"


# --- the aggregates -----------------------------------------------------------


def _rate(numerator: float, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _bootstrap(values: Sequence[float], seed: int, draws: int) -> tuple[float | None, float | None]:
    """The 2.5th and 97.5th percentiles of the mean over ``draws`` resamples of ``values``."""
    if not values or draws <= 0:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(draws))
    low = means[round(0.025 * (draws - 1))]
    high = means[round(0.975 * (draws - 1))]
    return round(low, 4), round(high, 4)


def _score(row: dict[str, Any]) -> float:
    return float(row.get("score") or 0.0)


def _answered(row: dict[str, Any]) -> bool:
    return not row.get("abstained") and not row.get("failed")


def cell(
    rows: Sequence[dict[str, Any]], *, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS
) -> dict[str, Any]:
    """One slice's counts and rates; abstained and failed questions score 0 and are counted."""
    scores = [_score(r) for r in rows]
    low, high = _bootstrap(scores, seed, draws)
    return {
        "questions": len(rows),
        "answered_right": sum(1 for r in rows if _answered(r) and _score(r) == 1.0),
        "answered_wrong": sum(1 for r in rows if _answered(r) and _score(r) < 1.0),
        "abstained": sum(1 for r in rows if r.get("abstained")),
        "failed": sum(1 for r in rows if r.get("failed") and not r.get("abstained")),
        "score": _rate(sum(scores), len(rows)),
        "full_score_rate": _rate(sum(1 for s in scores if s == 1.0), len(rows)),
        "numeric_exact_rate": _rate(sum(1 for r in rows if r.get("numeric_exact")), len(rows)),
        "ci_low": low,
        "ci_high": high,
    }


def _by(rows: Sequence[dict[str, Any]], key: str, order: Sequence[str]) -> dict[str, list]:
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(str(r.get(key)), []).append(r)
    names = [n for n in order if n in groups] + sorted(n for n in groups if n not in order)
    return {name: groups[name] for name in names}


def _capability(
    groups: dict[str, list[dict[str, Any]]], names: list[str], seed: int, draws: int
) -> dict[str, Any]:
    """The unweighted mean of the subtasks' mean scores, with bands from resampling each
    subtask's questions and taking the same mean."""
    per = {name: [_score(r) for r in groups[name]] for name in names}
    means = [statistics.fmean(per[name]) for name in names]
    low = high = None
    if draws > 0:
        rng = random.Random(seed)
        resampled = sorted(
            statistics.fmean(
                statistics.fmean(rng.choices(per[name], k=len(per[name]))) for name in names
            )
            for _ in range(draws)
        )
        low = round(resampled[round(0.025 * (draws - 1))], 4)
        high = round(resampled[round(0.975 * (draws - 1))], 4)
    return {
        "score": round(statistics.fmean(means), 4),
        "ci_low": low,
        "ci_high": high,
        "subtasks": names,
    }


def _judges(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    agree: dict[str, list[int]] = {}
    for row in rows:
        for j in row.get("judges") or []:
            jid = str(j.get("id"))
            s = stats.setdefault(jid, {"votes": 0, "invalid": 0, "text_fallback": 0})
            s["votes"] += 1
            invalid = (
                bool(j.get("error")) or j.get("parse_mode") == "failed" or j.get("score") is None
            )
            if invalid:
                s["invalid"] += 1
                continue
            if j.get("parse_mode") == "text_fallback":
                s["text_fallback"] += 1
            if _answered(row):
                agree.setdefault(jid, []).append(
                    int(round(float(j["score"]), 1) == round(_score(row), 1))
                )
    for jid, s in stats.items():
        s["agreement_with_majority"] = _rate(sum(agree.get(jid, [])), len(agree.get(jid, [])))
    return stats


def summarize(
    rows: Sequence[dict[str, Any]], *, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS
) -> dict[str, Any]:
    """Every cell the report shows for one configuration's rows, bands included."""
    rows = list(rows)
    subtasks = _by(rows, "subcategory", list(SUBTASKS))
    by_subtask = {name: cell(group, seed=seed, draws=draws) for name, group in subtasks.items()}
    by_capability = {}
    for capability in CAPABILITIES:
        names = [n for n in subtasks if SUBTASKS.get(n) == capability]
        if names:
            by_capability[capability] = _capability(subtasks, names, seed, draws)
    by_difficulty = {
        name: cell(group, seed=seed, draws=draws)
        for name, group in _by(rows, "difficulty", DIFFICULTIES).items()
    }
    worst = None
    for name, c in by_subtask.items():
        if c["questions"] >= WORST_MIN_QUESTIONS and (worst is None or c["score"] < worst["score"]):
            worst = {"subcategory": name, "score": c["score"]}
    known_limit = [r for r in rows if r.get("hit_token_limit") is not None]
    answered = [r for r in rows if _answered(r)]
    matched = sum(
        1
        for r in answered
        if (r.get("answer_language") or language_of(r.get("answer") or ""))
        in (r.get("language"), "und")
    )
    return {
        "total": cell(rows, seed=seed, draws=draws),
        "by_subtask": by_subtask,
        "by_capability": by_capability,
        "by_difficulty": by_difficulty,
        "by_language": {
            name: cell(group, seed=seed, draws=draws)
            for name, group in _by(rows, "language", list(LANGUAGES.values())).items()
        },
        "by_quality": {
            name: cell(group, seed=seed, draws=draws)
            for name, group in sorted(
                _by(rows, "qualityscore", ()).items(), key=lambda kv: -float(kv[0])
            )
        },
        "worst_subtask": worst,
        "hard": by_difficulty.get("Hard"),
        "hit_token_limit_rate": _rate(
            sum(1 for r in known_limit if r.get("hit_token_limit")), len(known_limit)
        ),
        "answer_language_match_rate": _rate(matched, len(answered)),
        "judges": _judges(rows),
    }


def headline_lines(block: dict[str, Any]) -> dict[str, Any]:
    """What the report's headline shows for MechVQA, per configuration: the total, the three
    capabilities, the Hard slice, the abstention rate and the worst subtask."""
    out: dict[str, Any] = {}
    for name, config in (block.get("configurations") or {}).items():
        summary = config.get("summary") or {}
        total = summary.get("total") or {}
        if not total:
            continue
        out[f"mechvqa_total_{name}"] = total.get("score")
        for capability in CAPABILITIES:
            c = (summary.get("by_capability") or {}).get(capability)
            if c:
                out[f"mechvqa_{capability.lower()}_{name}"] = c.get("score")
        hard = summary.get("hard")
        if hard:
            out[f"mechvqa_hard_{name}"] = hard.get("score")
        out[f"mechvqa_abstained_{name}"] = _rate(
            total.get("abstained", 0), total.get("questions", 0)
        )
        worst = summary.get("worst_subtask")
        if worst:
            out[f"mechvqa_worst_subtask_{name}"] = worst["subcategory"]
            out[f"mechvqa_worst_subtask_score_{name}"] = worst["score"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True, help=f"a checkout of {REPO} at {COMMIT}")
    ap.add_argument("--fixture-dir", type=Path, required=True, help="where the manifest lives")
    ap.add_argument("--write", action="store_true", help="write the manifest and its meta file")
    args = ap.parse_args(argv)
    rows = manifest_rows(load(args.root), args.root)
    meta = manifest_meta(rows, args.root)
    if args.write:
        args.fixture_dir.mkdir(parents=True, exist_ok=True)
        (args.fixture_dir / MANIFEST_NAME).write_text(dumps(rows), encoding="utf-8")
        (args.fixture_dir / MANIFEST_META).write_text(dumps_meta(meta), encoding="utf-8")
        print(f"wrote {args.fixture_dir / MANIFEST_NAME} and {MANIFEST_META}")
    print(f"{meta['questions']} questions on {meta['images']} images, jsonl {meta['jsonl_sha256']}")
    for name, count in meta["by_subtask"].items():
        print(f"  {name:28} {count:4}  ({SUBTASKS[name]})")
    print(f"  capability mismatches: {meta['capability_mismatches']}")
    problems = check(args.root, args.fixture_dir)
    for p in problems[:10]:
        print(" ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
