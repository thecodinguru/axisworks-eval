# MechVQA: the measurement track, preregistered

**Status.** Dated 7 November 2026, before any scored answer. MechVQA is a
second, public track beside AxisBench v0 ([`PROTOCOL.md`](PROTOCOL.md)),
which stays frozen and is not changed by anything here. MechVQA is never a
gate: no product decision, no cut-over of the hosted tier and no shipping
rule depends on a MechVQA number. It is never a holdout: the set is public,
its answers are public, and every model scored on it may have seen it.
Everything above the "Amendments" heading is fixed from this date; a later
change is appended there, dated, with its reason.

## 1. What is measured, by whom

MechVQA (ICML 2026, arXiv 2605.30794) is a visual question answering
benchmark over mechanical engineering drawings, written and scored by its
authors, not by us. It measures whether a model can answer an engineer's
question about a drawing in words: read a dimension, count features, compute
a geometric quantity, judge a drawing for an anomaly. Superposition Labs runs
the benchmark; the product repository produces a dated report
(`scripts/run_mechvqa.py`, writing `details.mechvqa` of
`reports/mechvqa_<date>.json`); this repository checks every number in it
without the product:

    python scripts/check_report.py reports/mechvqa_<date>.json
    python scripts/check_report.py reports/mechvqa_<date>.json --mechvqa datasets/mechvqa

The check recomputes every question's score from the judges' votes the
report lists, every summary cell from its rows and the headline, compares the
rows with the committed manifest (no question dropped or doubled, every
subtask, difficulty, language and quality as the set states them) and, with
the set fetched, rebuilds the manifest from the set and the judge-free
numeric check from the reference answers. It exits 1 on any difference. The
report's `schema` is 1, the same as the accuracy report's; the two never
share a file.

## 2. Dataset and pin

Repository `xiaofengShi/MechVQA`, commit
`c8d61513aef05b4f5d117bb0c6d1a055451d6f33`, file
`benchmark_data/vqa_benchmark/mechvqa_benchmark.jsonl` (SHA-256
`e9ff49a26742d24ac6c1cdff5279aa4eb75e3a17787da54efe75faff2adaeba2`):
1,185 questions on 562 drawing images (302 questions on JPEG files, 883 on
PNG), one image per question, each with a reference answer, a difficulty, a
language, a subtask and a quality score. `axisworks_eval/mechvqa_metrics.py`
reads the set, hashes every question, answer and image into
`fixtures/mechvqa/manifest.jsonl` (SHA-256
`2531fbc14da24f6777af7239221a127de42f55b904cbd3940df99057b3ae49c9`) and
states the counts in `fixtures/mechvqa/MANIFEST.json`; `mechvqa_metrics.check`
rebuilds both from a fetched copy and refuses a file that differs. A question
is identified by the first sixteen hex digits of the SHA-256 of its image
path and its bare question text (the SFT release's `<image>` placeholder and
answer-format suffix stripped), so the same question has the same id in
either release. Nine records repeat an earlier record's image and question
with another reference answer or difficulty (1,176 distinct pairs in 1,185
records); a repeat's id hashes its occurrence number in as well, so every
record has its own id and is counted once, as the set counts it.

The paper's ten subtasks under three capabilities, with the count of each:

| Capability | Subtask | Questions |
|---|---|---:|
| Recognition | Identification & Counting | 41 |
| Recognition | Dimension & Annotation | 445 |
| Recognition | Text & Table | 43 |
| Recognition | Item Localization | 113 |
| Reasoning | Structure Understanding | 29 |
| Reasoning | Geometric Calculation | 207 |
| Reasoning | Assembly Relationship | 22 |
| Reasoning | Projection & Multi-view | 33 |
| Judging | Anomaly Detection | 192 |
| Judging | Consistency Judgment | 60 |
| **all** | | **1,185** |

By capability (taxonomy): Recognition 642, Reasoning 291, Judging 252. Two
records state `Judging` for Geometric Calculation, which the taxonomy puts
under Reasoning (`capability_mismatches: 2` in `MANIFEST.json`); a row keeps
the record's stated capability, the capability means use the taxonomy, and
the count is published.

| Difficulty | Questions |
|---|---:|
| Easy | 580 |
| Medium | 419 |
| Hard | 186 |

| Language | Code | Questions |
|---|---|---:|
| Chinese (中文) | zh | 850 |
| English (英文) | en | 335 |

| Quality score | Questions |
|---|---:|
| qualityscore 1.0 | 1,117 |
| qualityscore 0.75 | 48 |
| qualityscore 0.5 | 3 |
| qualityscore 0.25 | 10 |
| qualityscore 0.0 | 7 |

The quality score is the set's own; it is kept on every row and sliced
(`by_quality`), never used to filter. No question is dropped.

## 3. Licence

Three facts, as found on 7 November 2026: the repository's code is Apache-2.0
(its `LICENSE` file at the pinned commit); the Hugging Face dataset card
declares Apache-2.0 while the GitHub README is silent on the data; the paper
(Appendix A) defers to the redistribution terms of the source materials the
drawings came from. The drawings therefore cannot be relicensed by us: they
are fetched for evaluation only (`scripts/fetch_mechvqa_dataset.py`, pinned
to the commit) and never committed. What this repository commits is the
manifest of hashes.

## 4. Configurations

Each configuration answers every question once. The three are:

1. **`record`: the product as shipped.** The drawing is read into a record
   the way the product reads an uploaded sheet (`extract_page` with the
   metadata, specification, dimension and GD&T calls; the rules engine over
   the rows; `record_from_extraction`), and the question is put to Ask over
   that record's evidence with the product's own prompts plus one rule, that
   the answer is in the question's language. Ask answers only from evidence
   and abstains when the record does not hold the answer; an abstention is
   an empty answer, scored 0 and counted as abstained, and published. This
   row is the one that says what the product can answer about a drawing, and
   its abstention rate is as much the measure as its score.
2. **`hosted`: the hosted tier's vision model as a raw VQA model.** The
   question with the repository's answer-format suffix and the drawing go to
   the model the hosted tier reads with, through the product's model client
   (`chat`), temperature 0.1, 4,096 output tokens. This row separates what
   the model can see from what the product's record keeps.
   The record is built as the product builds one: the converters persistence
   uses (criticality, review reasons, stable ids, canonical units), the same
   intra-page merge (two identical callouts at one spot are one row), the
   same reading order, the catalogue's rationale and clause on every finding,
   and the rules evaluated under the sheet's own dimensioning standard with
   no tenant configuration. Where the title block gives no part number or
   revision letter, the record carries `-` for it; the product never has an
   empty one, so the prompt is never left with a dangling "Rev".
3. **`model`: any OpenAI-compatible target** (`--target`: model, base URL,
   key variable), the same prompt as `hosted`, temperature 0.1, 4,096 output
   tokens, the image as a data URI after the text. This is the row the
   open-weight bench-off (`docs/vllm-bench-off.md` in the product) and the
   MechVL-4B acceptance rule (§8) use.

Every configuration's block records how it was configured (model, host, the
product's pinned settings and versions, the answer-suffix hash), the time
inside model calls (`inference`), and `subset` when fewer than the 1,185
questions were run (a `--limit` run); a report without `subset` must hold
every manifest question exactly once, or the check fails.

## 5. Scoring

What is copied from the repository's own evaluator
(`evaluation/mechvqa_eval/evaluate_vqa.py` at the pinned commit, SHA-256
`59ee510b7271eba160153737df3916457670c3f664a8fad4dd466a76c47e60c5`):

- The answer-format suffix appended to every question (`ANSWER_SUFFIX`, read
  from the fetched file at run time and required to hash to
  `543e48f662ef9354f2e6a94c4c1e8c69ba727902c181a279ba938ce8a63e626c`).
- `extract_answer`: the first closed `<answer>…</answer>`, else an unclosed
  `<answer>` to the end, else the text after `</think>`, else the whole
  response. The reference answer passes through the same function.
- The judge prompt (`build_judge_prompt`), verbatim, SHA-256
  `d4276ab5c4c2bf4914f031574d3b5ec317d48fe75de16b5d08c936840eeb290c`
  (`mechvqa_judge.JUDGE_PROMPT_TEMPLATE`, asserted by a test).
- `parse_judge_response`: the first flat JSON object in the judge's reply,
  its `score` clamped to [0, 1]; failing that, a `1` with no `0` is 1.0 and
  any `0` is 0.0.
- The vote: scores rounded to one decimal, the plurality wins. `total` is
  the mean score over all questions; `full_score_rate` the share scored 1.0.

Every departure, named:

1. **The judge prompt is the repository's, not the paper's Figure 18.** The
   two differ in wording; the repository's is what its authors ran, and it is
   the one a reader can hash.
2. **A failed judge parse is an invalid vote**, dropped from the plurality
   (the evaluator counts it as a valid 0). An empty judge reply and a judge
   call that errors are invalid for the same reason. Every invalid vote is
   counted per judge (`summary.judges`).
3. **A tie between the most common scores is 0.0** (the evaluator's tie goes
   to whichever judge is listed first).
4. **Judge temperature 0.1**, where the repository's example configuration
   says 0.0; `max_tokens` 1,024. Both are recorded per judge.
5. **An empty answer after extraction is 0.0 without judging** and is
   counted as abstained, so refusals are visible instead of judged as wrong.
   A failed model call is 0.0 and counted as failed.
6. **A capability score is the unweighted mean of its subtasks' scores**
   (the paper's convention; the evaluator reports only the overall mean), so
   the 445 Dimension & Annotation questions do not swamp Recognition.
7. **The worst subtask** is the lowest-scoring subtask among those with at
   least 20 questions, and the headline names it beside the total.
8. **95 % bootstrap bands** over questions (`random.Random(7)`, 1,000 draws)
   on every cell; a capability's band resamples each subtask's questions and
   takes the same unweighted mean.
9. **`numeric_exact`**, judge-free: every number in the reference answer
   (NFKC-folded, so full-width digits count; `12.50` reads `12.5`) appears in
   the answer; false when the reference has no number. Published per cell
   beside the judged score, never in its place.
10. **Language**: `中文` is `zh`, `英文` is `en`; an answer's language is read
    from the share of CJK letters among its letters (0.3 or more is `zh`; no
    letters at all is `und` and counts as matching). The share of Chinese
    questions (850 of 1,185) and the answer-language match rate are
    published, since a model that answers Chinese questions in English is
    judged on meaning by judges that read both.
11. **The quality score is kept and sliced, not filtered** (§2).

## 6. Judges

The paper's judges are GPT-OSS-120B, DeepSeek-V3.2 and Kimi-k2, voting by
plurality. A report names its judges (`judges`: id, model, host, the model
identifier the endpoint returned, temperature, the dates of the run) and
states `judges_as_paper`: true only when the judges are those three and no
other. When it is false no comparison with the paper's Table 2 is published;
the report's numbers stand on their own, and the headline says so in
`not_measured`. Every vote of every judge is in the report, so agreement
between judges and with the vote is a number a reader can recompute;
`mechvqa_judge.rejudge_sample` re-judges a sample with a judge of the
reader's own.

## 7. Prior exposure, stated

- Scored MechVQA answers to date: none. No prompt, rule or extractor change
  has been fitted to a MechVQA question or answer.
- A smoke run of 30 questions against the hosted model is planned to prove
  the plumbing before the first full run; when it is made it is declared
  here as an amendment with its date and its `--limit`, and its report, if
  written, is kept.
- The set is public on GitHub and Hugging Face. The hosted commercial model
  and the open-weight candidates do not publish their training lists;
  contamination cannot be ruled out for any of them and the report says so.

## 8. The MechVL-4B acceptance rule

MechVL-4B-RL is the paper's own model and the open-weight bench-off's
baseline; its published total is 84.85, with Recognition 89.70, Judging
82.81 and Reasoning 77.04 (Table 2). Before any open-weight candidate reads
the AxisBench holdout (`PROTOCOL.md` §8), MechVL-4B-RL is served on the same
stack and run as the `model` configuration here, and both must hold:

1. **The total lands where the paper's does.** With `[lo, hi]` the run's 95 %
   bootstrap band on `total` and `h = (hi − lo) / 2`, the run's total must
   satisfy `|total − 0.8485| ≤ h + 0.05`: the band around the paper's number,
   widened by five points for the public-subset caveat (the paper's figure
   was scored by its authors' judges on their copy of the set; the public
   release is 1,185 questions and our judges may not be theirs).
2. **The capabilities order as in Table 2**: Recognition > Judging >
   Reasoning, by the unweighted capability means.

If either fails, the AxisBench §8 run of every open-weight candidate waits,
and the MechVQA report names the suspect in its `not_measured` or an
amendment here: the serving stack (image tag, model identifier), the image
handling (resizing, format), the prompt suffix, the judges, or the public
subset. A pass is not a product claim; it says the stack serves the model as
its authors did.

## 9. Failure reporting

Every cell, every abstained and failed row, every invalid vote and every
answer is published with the report. No question, subtask, difficulty,
language, quality band or configuration is left out because of its number;
the worst subtask is shown beside the total.

## 10. Versions and tooling

- `mechvqa_metrics.VERSION` is `mechvqa_public_v1`; the manifest and this
  protocol change together or not at all.
- The product's prompt version 3, extractor version 3 and rules version are
  recorded in every report (`versions`), with the product commit and
  `uncommitted_code`, which must be `false` for a published report.
- Python 3.11 or later; the standard library only for `mechvqa_metrics` and
  `mechvqa_judge`. `python -m pytest` runs this repository's tests, which
  check the manifest against `MANIFEST.json` and against the counts in this
  file, the judge prompt's hash, and that a report checks out and a tampered
  one does not.

## Amendments

None.
