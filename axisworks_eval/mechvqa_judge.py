"""The MechVQA judge: the benchmark's own prompt, called over an OpenAI-style API.

MechVQA scores an answer by asking a language model whether it matches the
reference, so a score is only reproducible when the judge prompt is the
benchmark's own. ``JUDGE_PROMPT_TEMPLATE`` is the body of ``build_judge_prompt``
in the repository's ``evaluation/mechvqa_eval/evaluate_vqa.py`` at the pinned
commit, character for character, and ``JUDGE_PROMPT_SHA256`` and
``EVALUATOR_SHA256`` are asserted by the tests so a drift in either is a test
failure rather than a quiet change of protocol.

The judge is reached through ``POST {base_url}/chat/completions`` with the
standard library only, because this module is published to the verification
repository and a reader should be able to re-judge a stored report with
nothing but Python and an API key. Everything the protocol does with the judge's
text (answer extraction, JSON parsing, the plurality vote) lives in
``mechvqa_metrics``; this module only builds the prompt, makes the call and
caches what came back.

Two choices that are not the repository's: a judge call that fails (HTTP or
URL error, malformed body) returns ``score None`` with ``error`` set and is an
invalid vote rather than a 0, and only successful calls are cached, so a
network failure during a run is retried on the next one instead of being
frozen into the report.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from axisworks_eval.mechvqa_metrics import extract_answer, parse_judge_response, vote

#: ``build_judge_prompt`` from the pinned evaluate_vqa.py, verbatim, with the
#: f-string's doubled braces of the JSON example un-doubled. The three
#: placeholders are substituted by ``build_judge_prompt`` below, never by
#: ``str.format``, so the JSON example's braces stay as they are.
JUDGE_PROMPT_TEMPLATE = """Please judge whether the model answer is semantically correct for this VQA item.

Question:
{question}

Reference answer:
{correct_answer}

Model answer:
{model_answer}

Scoring rules:
- Return 1 if the model answer is correct or captures all key information.
- Return 0 if it is wrong, irrelevant, or missing key information.
- Only use score 0 or 1.

Return only a JSON object:
{"score": 0 or 1, "reason": "short explanation"}"""

#: SHA-256 of ``JUDGE_PROMPT_TEMPLATE`` (UTF-8); a test recomputes it.
JUDGE_PROMPT_SHA256 = "d4276ab5c4c2bf4914f031574d3b5ec317d48fe75de16b5d08c936840eeb290c"
#: SHA-256 of evaluation/mechvqa_eval/evaluate_vqa.py at the pinned commit.
EVALUATOR_SHA256 = "59ee510b7271eba160153737df3916457670c3f664a8fad4dd466a76c47e60c5"

_PLACEHOLDER = re.compile(r"\{(question|correct_answer|model_answer)\}")
#: The keys a judge entry carries in a report row; the cache file keeps more.
VOTE_KEYS = ("id", "model", "returned_model", "score", "reason", "parse_mode", "error")


def build_judge_prompt(question: str, correct_answer: str, model_answer: str) -> str:
    """The repository's prompt for one answer.

    A single regex pass substitutes the three placeholders, so a question or an
    answer that itself contains ``{model_answer}`` is not substituted twice,
    and the JSON example's braces are never interpreted.
    """
    values = {
        "question": str(question),
        "correct_answer": str(correct_answer),
        "model_answer": str(model_answer),
    }
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], JUDGE_PROMPT_TEMPLATE)


@dataclass
class Judge:
    """One judge endpoint. The key is named by environment variable, never stored."""

    id: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_s: float = 120.0


_REQUIRED = ("id", "model", "base_url", "api_key_env")


def load_judges(source: str | Path | list[dict]) -> list[Judge]:
    """Judges from dicts, a JSON string or a JSON file (a list, or ``{"judges": [...]}``).

    A string is a path when a file exists at it and JSON otherwise, so a
    command line can pass either. Ids must be unique: votes are keyed by id
    in the report and the cache.
    """
    if isinstance(source, Path):
        data = json.loads(source.read_text(encoding="utf-8"))
    elif isinstance(source, str):
        text = source
        candidate = Path(source)
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
        data = json.loads(text)
    else:
        data = list(source)
    if isinstance(data, Mapping):
        if "judges" not in data:
            raise ValueError("a judges object must carry a 'judges' list")
        data = data["judges"]
    if not isinstance(data, list):
        raise ValueError("judges must be a list of objects")
    judges: list[Judge] = []
    seen: set[str] = set()
    for entry in data:
        if not isinstance(entry, Mapping):
            raise ValueError(f"judge entry is not an object: {entry!r}")
        missing = [key for key in _REQUIRED if not entry.get(key)]
        if missing:
            raise ValueError(f"judge entry missing {', '.join(missing)}: {dict(entry)!r}")
        judge = Judge(
            id=str(entry["id"]),
            model=str(entry["model"]),
            base_url=str(entry["base_url"]).rstrip("/"),
            api_key_env=str(entry["api_key_env"]),
            temperature=float(entry.get("temperature", 0.1)),
            max_tokens=int(entry.get("max_tokens", 1024)),
            timeout_s=float(entry.get("timeout_s", 120.0)),
        )
        if judge.id in seen:
            raise ValueError(f"duplicate judge id {judge.id!r}")
        seen.add(judge.id)
        judges.append(judge)
    return judges


def _content_text(content: Any) -> str:
    """The text of a chat message whose content is a string or a list of parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, Mapping):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def chat_completion(judge: Judge, prompt: str) -> tuple[str, str]:
    """One chat completion; ``(text, returned_model)``.

    Raises ``urllib.error.URLError`` (``HTTPError`` for a non-2xx status, with
    the start of the body as its message), ``TimeoutError`` or
    ``ValueError`` for a body that is not the expected JSON. The caller decides
    what a failure means for the vote.
    """
    body = json.dumps(
        {
            "model": judge.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": judge.temperature,
            "max_tokens": judge.max_tokens,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = os.environ.get(judge.api_key_env, "") if judge.api_key_env else ""
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        f"{judge.base_url.rstrip('/')}/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=judge.timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200] if exc.fp else ""
        raise urllib.error.HTTPError(
            exc.url, exc.code, detail or str(exc.reason), exc.headers, None
        ) from None
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("chat completion body is not an object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat completion has no choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    if not isinstance(message, Mapping):
        raise ValueError("chat completion choice has no message")
    text = _content_text(message.get("content"))
    returned_model = str(data.get("model") or "")
    return text, returned_model


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(judge: Judge, question_or_id: str, answer: str) -> str:
    """The cache file stem for one (judge, question, answer); the id when the caller has one."""
    return _sha256(f"{judge.id}|{judge.model}|{question_or_id}|{_sha256(answer)}")


def judge_answer(
    judge: Judge,
    *,
    question: str,
    reference: str,
    answer: str,
    cache_dir: Path | None,
    question_id: str | None = None,
) -> dict:
    """One judge's vote on one answer, from the cache when it has it.

    The reference goes through ``extract_answer`` as the repository does
    (``get_correct_answer``), so a reference that carries ``<answer>`` tags is
    judged by its bare text. The answer is judged as given: the runner has
    already extracted it. A failed call returns ``score None`` and ``error``
    set, and is not cached.
    """
    key = cache_key(judge, question_id or question, answer)
    cache_path = Path(cache_dir) / f"{key}.json" if cache_dir is not None else None
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, Mapping) and all(k in cached for k in VOTE_KEYS):
                return {k: cached[k] for k in VOTE_KEYS}
        except (OSError, ValueError):
            pass  # a damaged cache file is re-judged, not fatal
    prompt = build_judge_prompt(question, extract_answer(reference), answer)
    result: dict[str, Any] = {
        "id": judge.id,
        "model": judge.model,
        "returned_model": "",
        "score": None,
        "reason": "",
        "parse_mode": "failed",
        "error": None,
    }
    try:
        text, returned_model = chat_completion(judge, prompt)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    parsed = parse_judge_response(text)
    result.update(
        returned_model=returned_model,
        score=parsed.score,
        reason=parsed.reason,
        parse_mode=parsed.parse_mode,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(result)
        record["response"] = text
        record["prompt_sha256"] = _sha256(prompt)
        cache_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def _configurations(report: Mapping) -> Mapping[str, Mapping]:
    """The MechVQA configurations from a full report, its details, the block or the dict itself."""
    if "configurations" in report:
        return report["configurations"]
    details = report.get("details", report)
    block = details.get("mechvqa", details)
    return block.get("configurations", {})


def _question_lookup(questions: Any) -> dict[str, Any]:
    if questions is None:
        return {}
    if isinstance(questions, Mapping):
        return dict(questions)
    lookup: dict[str, Any] = {}
    for q in questions:
        qid = q.get("question_id") if isinstance(q, Mapping) else getattr(q, "question_id", None)
        if qid:
            lookup[str(qid)] = q
    return lookup


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def rejudge_sample(
    report: Mapping,
    judges: Iterable[Judge],
    *,
    n: int = 50,
    seed: int = 7,
    questions: Any = None,
    cache_dir: Path | None = None,
) -> dict:
    """Re-judge ``n`` stored answers per configuration and report agreement with the stored votes.

    A report row keeps the answer but not the question or the reference, so
    ``questions`` (``Question`` objects or dicts, by ``question_id``) supplies
    them; a row that also carries ``question`` and ``reference`` needs no
    lookup. Rows that were abstained, failed or have no text to judge are
    skipped and counted. The sample is drawn from the rows sorted by
    ``question_id`` with ``random.Random(seed)``, so the same report and seed
    name the same questions. A judge that cannot be reached contributes an
    error, never an exception: the point of this function is a number for the
    report, and a flaky endpoint should show up as disagreement and errors,
    not abort the run.
    """
    judges = list(judges)
    lookup = _question_lookup(questions)
    out: dict[str, Any] = {
        "n": n,
        "seed": seed,
        "judges": [asdict(j) for j in judges],
        "configurations": {},
    }
    for name, config in _configurations(report).items():
        rows = [r for r in (config.get("questions") or []) if isinstance(r, Mapping)]
        judgeable = []
        skipped = {"abstained": 0, "failed": 0, "missing_text": 0}
        for row in sorted(rows, key=lambda r: str(r.get("question_id", ""))):
            if row.get("abstained") or not (row.get("answer") or "").strip():
                skipped["abstained"] += 1
                continue
            if row.get("failed"):
                skipped["failed"] += 1
                continue
            source = lookup.get(str(row.get("question_id")))
            question = row.get("question") or _field(source, "question")
            reference = row.get("reference") or _field(source, "answer")
            if not question or reference is None:
                skipped["missing_text"] += 1
                continue
            judgeable.append((row, str(question), str(reference)))
        sample = random.Random(seed).sample(judgeable, min(n, len(judgeable)))
        per_judge: dict[str, dict[str, Any]] = {
            j.id: {
                "votes": 0,
                "valid": 0,
                "errors": 0,
                "compared": 0,
                "agreed": 0,
                "compared_score": 0,
                "agreed_score": 0,
            }
            for j in judges
        }
        sampled_rows = []
        revote_compared = revote_agreed = 0
        for row, question, reference in sample:
            stored = {
                str(v.get("id")): v
                for v in (row.get("judges") or [])
                if isinstance(v, Mapping) and v.get("id")
            }
            votes: list[dict] = []
            fresh: dict[str, float | None] = {}
            for judge in judges:
                try:
                    result = judge_answer(
                        judge,
                        question=question,
                        reference=reference,
                        answer=str(row.get("answer") or ""),
                        cache_dir=cache_dir,
                        question_id=str(row.get("question_id") or "") or None,
                    )
                except Exception as exc:  # a judge must never abort the sample
                    result = {
                        "id": judge.id,
                        "model": judge.model,
                        "returned_model": "",
                        "score": None,
                        "reason": "",
                        "parse_mode": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                votes.append(result)
                fresh[judge.id] = result["score"]
                tally = per_judge[judge.id]
                tally["votes"] += 1
                valid = result["error"] is None and result["parse_mode"] != "failed"
                if not valid:
                    if result["error"] is not None:
                        tally["errors"] += 1
                    continue
                tally["valid"] += 1
                score = round(float(result["score"]), 1)
                before = stored.get(judge.id)
                if before is not None and before.get("score") is not None:
                    tally["compared"] += 1
                    tally["agreed"] += int(score == round(float(before["score"]), 1))
                if row.get("score") is not None:
                    tally["compared_score"] += 1
                    tally["agreed_score"] += int(score == round(float(row["score"]), 1))
            revote = vote(votes)
            if row.get("score") is not None and any(
                v["error"] is None and v["parse_mode"] != "failed" for v in votes
            ):
                revote_compared += 1
                revote_agreed += int(round(revote, 1) == round(float(row["score"]), 1))
            sampled_rows.append(
                {
                    "question_id": row.get("question_id"),
                    "stored_score": row.get("score"),
                    "revote": revote,
                    "votes": fresh,
                }
            )
        out["configurations"][name] = {
            "rows": len(rows),
            "judgeable": len(judgeable),
            "sampled": len(sample),
            "skipped": skipped,
            "revote_agreement_with_stored_score": _rate(revote_agreed, revote_compared),
            "judges": {
                jid: {
                    "votes": t["votes"],
                    "valid": t["valid"],
                    "errors": t["errors"],
                    "agreement_with_stored_vote": _rate(t["agreed"], t["compared"]),
                    "agreement_with_stored_score": _rate(t["agreed_score"], t["compared_score"]),
                }
                for jid, t in per_judge.items()
            },
            "questions": sampled_rows,
        }
    return out
