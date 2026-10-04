# jev-eval

**Stop guessing your confidence threshold. Measure it, and get the cut-point the SDKs can load.**

jev-eval measures and calibrates [TypeSafe AI's Jev](https://typesafe.ai/) on *your own* labeled data. Give it a labeled dataset and your Jev questions. It reports per-question accuracy, whether Jev's confidence can be trusted (ECE, a reliability table), and a risk-coverage table. Then it recommends the confidence cut-points for the family's core pattern: run Jev on everything, auto-decide the confident cases, escalate the rest.

Every confidence-gated workflow (jev-sort `--escalate`, Leash `repairAt`/`noteAt`, a jev-guard hold threshold, an SDK consumer) needs a cut-point. Jev is ~68% accurate raw, so that number decides whether gating works at all. Set it too low and you auto-accept wrong answers. Set it too high and everything escalates, which loses the cost win that justified Jev.

Zero runtime dependencies (stdlib only), Python >= 3.10.

## Quick start

```console
$ export JEV_API_KEY=...
$ jev-eval run --data labeled.jsonl --config jev-eval.json --cache run.jsonl
$ jev-eval report --cache run.jsonl
$ jev-eval thresholds --cache run.jsonl --target-accuracy 0.95
team (choice, n=412): gate at 0.82
  auto-handle 58.7% (~242 rows) at 94.6% accuracy out-of-bag, escalate 41.3%
  95% range: gate 0.76-0.88, accuracy 91.2%-97.9%
  on the rows it was picked on it looks like 95.3% (optimistic)
wrote thresholds.json (3 questions)
```

Only `run` calls Jev (and costs money). Every other command is a pure function over the local cache, so you can re-slice with different bins, targets or guards for free.

| Command | What it does |
|---|---|
| `run` | One batched Jev call per labeled row; answers to a resumable JSONL cache |
| `report` | Accuracy, calibration, risk-coverage per question and for the row-level gate; `--slice`, `--svg` |
| `thresholds` | Recommend cut-points under an honesty guard; writes `thresholds.json` |
| `compare` | Two runs over the same rows: model-pin drift, or two question wordings |
| `calibrate` | Measure what a recalibration would fix; writes an optional `calibration.json` |

## Inputs

**Labeled data** (`--data`, JSONL): a `state` (string or object, sent to Jev verbatim), a `labels` map from question id to gold, and an optional `meta` object for per-slice reports. `meta` is never sent to Jev. Gold is a bool for Noul, a criterion key for Choice and a criterion label for Score. A row without a label for a question is skipped for that question only, so partial labels are fine. A gold value that does not fit its question (for example a typo in a Choice key) fails the run up front.

```json
{"state": {"subject": "Charged twice", "text": "billed two times, refund please"}, "labels": {"team": "billing", "urgent": true, "sentiment": "negative"}, "meta": {"source": "email"}}
```

**Questions**, as `--config` (JSON) and/or inline `-q` flags in jev-sort's syntax:

```console
-q 'team:choice(billing,tech,sales)'  -q 'urgent:noul'  -q 'sentiment:score(negative,neutral,positive)'
```

```json
{
  "model": "jev-latest",
  "questions": {
    "team": {"kind": "choice", "instructions": "Which team should handle this ticket?",
             "options": {"billing": "payment or charge issues", "tech": "bugs, errors, outages", "sales": "pre-purchase questions"}},
    "urgent": {"kind": "noul", "instructions": "Does the message convey urgency?"},
    "sentiment": {"kind": "score", "instructions": "Overall customer sentiment", "levels": ["negative", "neutral", "positive"]}
  }
}
```

The config uses jev-sort's question schema (`kind`/`options`/`levels`). The Jev wire shape (`type`/`criteria`) is accepted too. Config files are **JSON, not YAML**: YAML would need a runtime dependency. A jev-sort YAML config converts one to one.

## What it measures

One batched Jev call per row asks only that row's labeled questions. Each answer is compared to its gold label:

| Type | Correct when | Gate confidence | Reliability table bins |
|---|---|---|---|
| Choice | argmax label == gold | `confidence` | confidence vs "argmax was right" |
| Noul | `noul >= 0.5` == gold | `abs(noul - 0.5) * 2` | raw P(yes) vs the empirical yes-rate |
| Score | nearest criterion label == gold | `confidence` | confidence vs "exact level was right" |

- **Risk-coverage (the headline):** for each candidate gate `t`, the coverage (rows auto-decided at `>= t`), the selective accuracy on those rows and the escalation rate. Noul also reports precision and recall on "yes".
- **Calibration (the trust check):** ECE, MCE and the reliability table. When ECE is above 0.1 the report warns that thresholds built on it are shaky.
- **Raw accuracy:** reported, but on purpose not the headline. Score also reports **within-1 accuracy** and **mean level distance**, because levels are ordered and one level off is not as wrong as the opposite end.
- **Row-level gate:** jev-sort escalates a row on the *minimum* of its Choice/Score confidences. When more than one question is gated, jev-eval scores that composite too (correct = every gated answer correct), so the recommended cut-point matches what the tool actually does.
- **Per-slice** (`report --slice length` or `--slice meta.source`): accuracy and ECE per slice, to catch Jev doing worse on one source or on long states. `length` buckets the state at 500, 2k and 8k characters.
- **Chart** (`report --svg reliability.svg`): one reliability panel per question. Bars under the diagonal mean overconfidence.

A Score answer's `score` is read as a level index (0..N-1, like jev-rerank). If the answer has a numeric `legend` (`{label: position}`), the nearest legend position wins instead.

## Thresholds

```console
jev-eval thresholds --cache run.jsonl (--target-accuracy 0.95 | --max-escalation 0.30)
                    [--bootstrap N | --holdout FRACTION | --no-guard] [--tolerance N] [--seed S] [--out thresholds.json]
```

- `--target-accuracy A`: the lowest gate whose selective accuracy reaches `A` (the most coverage at that accuracy).
- `--max-escalation E`: the strictest gate that still escalates at most `E` of rows (the best accuracy within that budget).
- `--tolerance N`: for Score, count within-N levels of gold as correct. Recorded in `thresholds.json`.

Candidate gates are the confidences actually observed, so every recommended `t` can be reached. If no gate meets the goal, jev-eval says so instead of picking the closest.

**Honesty guard.** If you pick a cut-point using the same rows you score it on, the reported accuracy comes out too high, because you picked the cut that happened to look best on those rows. The default guard is **bootstrap with out-of-bag scoring** (`--bootstrap 1000`). It resamples your rows 1000 times. Each resample picks a gate on the rows it drew and scores that gate on the rows it left out. You get:

- the gate picked on all rows,
- an honest accuracy and coverage: the out-of-bag mean, measured on rows each pick never saw (slightly pessimistic, the safe direction),
- a 95% range for the gate and its accuracy,
- the optimistic same-rows number, shown next to it so you can see the gap.

On small datasets that gap can be large: a gate that looks like 60% in-sample on uninformative confidences scores under 50% out-of-bag. Use `--holdout 0.3` on large datasets (pick on 70%, report on the unseen 30%). Use `--no-guard` only when you accept the optimism. Resampling is seeded (`--seed`), so output is reproducible.

**Minimum N.** Below 200 labeled rows per question the recommendation is marked provisional. Below 50 it is refused with "gather more labels" and left out of `thresholds.json`.

### `thresholds.json` (contract version 1)

A small, versioned file the rest of the family can load:

```json
{
  "version": 1,
  "model": "jev-latest",
  "generatedAt": "2026-10-04T12:00:00Z",
  "guard": {"method": "bootstrap", "resamples": 1000, "seed": 0},
  "tolerance": 0,
  "questions": {
    "team": {"type": "choice", "threshold": 0.82, "accuracy": 0.946, "coverage": 0.587, "n": 412},
    "urgent": {"type": "noul", "threshold": 0.7, "accuracy": 0.951, "coverage": 0.55, "n": 412}
  },
  "composite": {"threshold": 0.8, "accuracy": 0.92, "coverage": 0.48, "n": 412, "questions": ["sentiment", "team"]}
}
```

| Field | Meaning |
|---|---|
| `model` | The model the run used. Thresholds are model-specific: re-measure when you change the pin. |
| `guard` | How `accuracy`/`coverage` were measured: `bootstrap` (out-of-bag mean), `holdout` (held-out split, with `fraction`), or `none` (same rows, optimistic). |
| `tolerance` | `0` means exact-match correctness. `N > 0` means Score answers within N levels of gold counted as correct. |
| `questions.<id>.threshold` | Auto-decide when the gate confidence is `>=` this. For **Noul** it applies to `abs(noul - 0.5) * 2`, never to the raw P(yes): 0.7 means "auto-decide when `noul >= 0.85` or `noul <= 0.15`". |
| `questions.<id>.accuracy`, `coverage`, `n` | What the gate buys, measured per `guard`, and the labeled rows behind it. |
| `composite` | Present only when two or more questions are gated (Choice/Score). One cut-point for the *minimum* of those confidences per row: the number for jev-sort `--escalate`. |

Questions that were refused, unstable under the guard, or had no reachable gate are left out, never written as `null`. Thresholds are on Jev's *raw* confidences.

## Compare

```console
jev-eval compare baseline.jsonl candidate.jsonl
```

Pairs the two caches on (state, question id), so the model or the question wording may differ, and that difference is the point. Per question it shows accuracy and ECE for each run, how often they agree, and what B **fixes** (A wrong, B right) and **breaks** (A right, B wrong). Use it before moving a model pin, or to A/B two wordings of a question. Rows only one run answered are counted as unpaired, never dropped silently.

## Recalibration (measured, not applied)

`report` and `calibrate` answer one question: if you remapped Jev's raw confidence with an isotonic (monotone) fit learned from your labels, how calibrated would it be on rows the fit never saw (5-fold cross-validation)?

- **For gating it changes nothing for Choice and Score.** The remap keeps the ordering, so "gate at raw 0.82" and "gate at calibrated 0.95" auto-handle exactly the same rows. `thresholds.json` already captures that, on raw values.
- **It matters when the confidence is used as a probability:** showing "% sure" to people, expected-cost decisions, or comparing confidences across questions that are calibrated differently.
- **Noul is the exception.** A remap of P(yes) can move the 0.5 crossing, so the report says where calibrated 0.5 sits in raw terms ("Jev's 0.5 is really 0.42").

`jev-eval calibrate --cache run.jsonl` also writes an optional `calibration.json`: per question, `input` (`confidence` or `noul`), `eceRaw`, `eceRecalibrated` (cross-validated), and `steps` as `[x_lo, x_hi, calibrated]`. A raw value maps to the first step whose `x_hi` is `>=` it, or to the last step above that. Nothing in the family is required to load it: jev-eval measures Jev, it does not wrap it.

## Library

```python
from jev_eval import load_rows, provider_from_env, recommend_threshold, report, run, score_records

records = run(load_rows("labeled.jsonl"), questions, provider_from_env(), cache_path="run.jsonl")
summary = report(records, bins=10)                  # accuracy, ece, reliability, risk-coverage, recalibration
cut = recommend_threshold(score_records(records).items["team"], target_accuracy=0.95)
```

Also exported: `compare`, `slice_report`, `recommend_all`, `thresholds_document`, `recalibration`, `calibration_document`, `isotonic`, and the metric primitives (`bin_by_confidence`, `ece`, `mce`, `selective_accuracy`, `risk_coverage`).

`Provider` is a protocol (`evaluate(state, questions) -> dict`). Pass any object that implements it, for example a fake in tests. `provider_from_env` reads `JEV_API_KEY` and honors `TYPESAFE_AI_BASE_URL` for a self-host, proxy or mock. The default model is `jev-latest`.

The cache is JSONL, one line per (row, question): model, state and question hashes, the question definition, the raw answer, the gold label, `meta` and the state size. The state itself is not stored. A rerun reuses cached answers (keyed by model, state hash and question hash). Progress streams to `<cache>.partial`, so an interrupted run resumes, and a repeated state is paid for once.

## Honest limitations

- **No labels, no signal.** Label quality caps everything here. ECE measured on noisy gold is an upper bound on the true error, not the true error.
- **It measures Jev. It does not improve Jev.** ~68% raw accuracy is a property of the model. jev-eval tells you the honest number and the best gate given it.
- **Score correctness is a modeling choice** (nearest label). The report states this and shows within-1 and mean distance next to it.
- **Small samples lie**, which is why the min-N guard and out-of-bag scoring exist. A gate that covers a handful of rows can clear any target by luck; the out-of-bag numbers and the range expose it.
- **Not a gate.** It recommends a cut-point for a human to review. It never changes a production threshold itself.
- Text only, ~32k context per state, like the rest of the family.

## Status

**1.0**: `run`, `report` (with slices and an SVG chart), `thresholds` with the out-of-bag bootstrap guard, `compare`, measure-only `calibrate`, and the `thresholds.json` v1 contract, for all three question types. Tested against a fake provider at 99% coverage. Gated by the [Foundry](https://github.com/CMaintz/foundry) Python stack (ruff, mypy strict, pytest, pip-audit, structural smells).

MIT (c) Christoffer Maintz
