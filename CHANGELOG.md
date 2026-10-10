# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-10

### Added

- `run` and `report` print the input tokens Jev metered and an estimated cost. Each cache record now carries the `call` that answered it and that call's `input_tokens`, so a repeated state or a resumed run is not counted twice.
- `run` warns before sending a row estimated within 10% of Jev's 32k state plus longest question budget, and `report` counts those rows.

### Changed

- A row Jev rejects (HTTP 400, 413 or 422, for example a state over the context limit) is recorded with its `error` and counted as unanswered instead of aborting the run. A rerun asks it again. `TypeSafeProvider` raises the new `RequestRejected` for these.

## [1.0.0] - 2026-10-04

First release. The `thresholds.json` contract (version 1) is stable from here.

### Added

- `jev-eval run`: one batched Jev call per labeled row, asking only that row's labeled questions. Raw answers go to a JSONL cache keyed by (model, state hash, question hash). A rerun resumes, progress streams to `<cache>.partial`, and a repeated state is paid for once. Rows may carry an optional `meta` object (never sent to Jev) for slicing.
- `jev-eval report`: per question and for the row-level composite gate. It covers accuracy, confusion, ECE, MCE, a reliability table (Noul bins raw P(yes) against the yes-rate) and a risk-coverage table. Noul also gets yes precision/recall, and Score gets within-1 accuracy and mean level distance. Also included: measure-only recalibration (isotonic, 5-fold cross-validated ECE, and the Noul 0.5 shift), `--slice length|meta.<field>`, and `--svg` (a stdlib reliability chart).
- `jev-eval thresholds`: `--target-accuracy` / `--max-escalation`, with `--tolerance` for Score. The default guard is bootstrap with out-of-bag scoring (`--bootstrap 1000`): it reports accuracy and coverage on rows each pick never saw, a 95% range, and the optimistic same-rows number alongside. `--holdout` and `--no-guard` are opt-in. The min-N guard warns below 200 rows and refuses below 50.
- `thresholds.json` contract v1: `{version, model, generatedAt, guard, tolerance, questions: {id: {type, threshold, accuracy, coverage, n}}, composite?}`.
- `jev-eval compare`: pairs two runs on (state, question id) and reports accuracy, ECE, agreement and what B fixes or breaks. It flags reworded questions and counts unpaired rows.
- `jev-eval calibrate`: measures what an isotonic recalibration would fix and writes an optional `calibration.json` mapping. Nothing in the family is required to load it.
- Questions from a JSON config (jev-sort's question schema or the Jev wire shape) and/or inline `-q` flags; gold labels are validated up front.
- Zero-dependency `TypeSafeProvider` (stdlib `urllib`, 429/529 backoff), the `Provider` protocol and `provider_from_env` (`JEV_API_KEY`, `TYPESAFE_AI_BASE_URL`).
- Foundry v2 Python stack: `mise run gate` (ruff, mypy strict, pytest with a 90% coverage floor, pip-audit, habit-hooks smells) and the Foundry gate/security/ratchet/bootstrap facades at `@v2`.

### Not in scope

- Applying a recalibration at runtime: jev-eval measures Jev, it does not wrap it. `calibration.json` exists for whoever wants it.
