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
team (choice, n=412): gate at 0.82, plausibly 0.76-0.88 (accuracy 93.1%-97.4%), auto-handle 61.2% at 95.3% accuracy, escalate 38.8%
wrote thresholds.json (3 questions)
```

Only `run` calls Jev (and costs money). `report` and `thresholds` are pure functions over the local cache, so you can re-slice with different bins, targets or guards for free.

## Inputs

**Labeled data** (`--data`, JSONL): a `state` (string or object, sent to Jev verbatim) and a `labels` map from question id to gold. Gold is a bool for Noul, a criterion key for Choice and a criterion label for Score. A row without a label for a question is skipped for that question only, so partial labels are fine. A gold value that does not fit its question (for example a typo in a Choice key) fails the run up front.

```json
{"state": {"subject": "Charged twice", "text": "billed two times, refund please"}, "labels": {"team": "billing", "urgent": true, "sentiment": "negative"}}
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

A Score answer's `score` is read as a level index (0..N-1, like jev-rerank). If the answer has a numeric `legend` (`{label: position}`), the nearest legend position wins instead.

## Thresholds

```console
jev-eval thresholds --cache run.jsonl (--target-accuracy 0.95 | --max-escalation 0.30)
                    [--bootstrap N | --holdout FRACTION | --no-guard] [--tolerance N] [--seed S] [--out thresholds.json]
```

- `--target-accuracy A`: the lowest gate whose selective accuracy reaches `A` (the most coverage at that accuracy).
- `--max-escalation E`: the strictest gate that still escalates at most `E` of rows (the best accuracy within that budget).
- `--tolerance N`: for Score, count within-N levels of gold as correct.

Candidate gates are the confidences actually observed, so every recommended `t` can be reached. If no gate meets the goal, jev-eval says so instead of picking the closest.

**Honesty guard.** A threshold picked on the same rows it is scored on is optimistic. By default jev-eval uses **`--bootstrap 1000`**: it picks on all rows and reports a 95% range for the gate and its accuracy over 1000 resamples ("gate at 0.82, plausibly 0.76-0.88"). Use `--holdout 0.3` on large datasets (pick on 70%, report on the unseen 30%). Use `--no-guard` only when you accept the optimism. Resampling is seeded (`--seed`), so the output is reproducible.

**Minimum N.** Below 200 labeled rows per question the recommendation is marked provisional. Below 50 it is refused with "gather more labels" and left out of `thresholds.json`.

### `thresholds.json` (contract version 1)

A small, versioned file the rest of the family can load:

```json
{
  "version": 1,
  "model": "jev-latest",
  "generatedAt": "2026-10-04T12:00:00Z",
  "questions": {
    "team": {"type": "choice", "threshold": 0.82, "accuracy": 0.953, "coverage": 0.612, "n": 412},
    "urgent": {"type": "noul", "threshold": 0.7, "accuracy": 0.961, "coverage": 0.55, "n": 412}
  },
  "composite": {"threshold": 0.8, "accuracy": 0.93, "coverage": 0.48, "n": 412, "questions": ["sentiment", "team"]}
}
```

- `model` is recorded because thresholds are model-specific. Re-measure when you change the model pin.
- A **Noul** threshold applies to `abs(noul - 0.5) * 2`, never to the raw P(yes). A gate of 0.7 means "auto-decide when `noul >= 0.85` or `noul <= 0.15`".
- With `--holdout`, `accuracy`/`coverage` are the held-out numbers. Otherwise they are on all rows, and the bootstrap range is printed with the recommendation.
- Questions that were refused or had no reachable gate are left out, never written as `null`.
- `composite` is present only when two or more questions are gated (Choice/Score). It is the single cut-point for jev-sort `--escalate`.

## Library

```python
from jev_eval import load_rows, provider_from_env, recommend_threshold, report, run, score_records

records = run(load_rows("labeled.jsonl"), questions, provider_from_env(), cache_path="run.jsonl")
summary = report(records, bins=10)                  # accuracy, ece, reliability, risk-coverage
cut = recommend_threshold(score_records(records).items["team"], target_accuracy=0.95)
```

`Provider` is a protocol (`evaluate(state, questions) -> dict`). Pass any object that implements it, for example a fake in tests. `provider_from_env` reads `JEV_API_KEY` and honors `TYPESAFE_AI_BASE_URL` for a self-host, proxy or mock. The default model is `jev-latest`.

The cache is JSONL, one line per (row, question), with the model, state and question hashes, the question definition, the raw answer and the gold label. A rerun reuses cached answers (keyed by model, state hash and question hash), so an interrupted run resumes and a repeated state is paid for once.

## Honest limitations

- **No labels, no signal.** Label quality caps everything here. ECE measured on noisy gold is an upper bound on the true error, not the true error.
- **It measures Jev. It does not improve Jev.** ~68% raw accuracy is a property of the model. jev-eval tells you the honest number and the best gate given it.
- **Score correctness is a modeling choice** (nearest label). The report states this and shows within-1 and mean distance next to it.
- **Small samples lie**, which is why the min-N guard exists.
- **Not a gate.** It recommends a cut-point for a human to review. It never changes a production threshold itself.
- **No recalibration.** It reports calibration and does not rewrite Jev's confidences (see roadmap).
- Text only, ~32k context per state, like the rest of the family.

## Status

**v0.2**: `run`, `report`, `thresholds` and the `thresholds.json` contract, for all three question types, tested against a fake provider. Gated by the [Foundry](https://github.com/CMaintz/foundry) Python stack (ruff, mypy strict, pytest at 90% coverage, pip-audit, structural smells).

**Roadmap:** v0.3 adds `compare` (model-pin drift, question-wording A/B), an optional stdlib SVG reliability chart and per-slice reports. Post-hoc probability recalibration (histogram binning or Platt scaling emitting a mapping the SDKs could apply) is roadmap only. Empirical thresholds already give the right gate, and rewriting confidences would turn jev-eval from "measures Jev" into "wraps Jev".

MIT (c) Christoffer Maintz
