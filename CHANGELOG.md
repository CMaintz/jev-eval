# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **v0.1:** `jev-eval run` - one batched Jev call per labeled row (labeled questions only), raw answers to a resumable JSONL cache keyed by (model, state hash, question hash), streamed to `<cache>.partial` so an interrupted run keeps its progress.
- **v0.1:** `jev-eval report` - per question: accuracy, confusion, ECE, MCE, reliability table (Noul bins raw P(yes) against the yes-rate), risk-coverage table, Noul precision/recall on yes, Score within-1 accuracy and mean level distance, plus the row-level composite gate (min of Choice/Score confidences).
- **v0.2:** `jev-eval thresholds` - `--target-accuracy` / `--max-escalation`, bootstrap honesty guard by default (`--bootstrap 1000`), opt-in `--holdout`, `--tolerance` for Score, min-N warning (< 200) and refusal (< 50).
- **v0.2:** the `thresholds.json` contract, version 1: `{version, model, generatedAt, questions: {id: {type, threshold, accuracy, coverage, n}}, composite?}`.
- Questions from a JSON config (jev-sort's question schema or the Jev wire shape) and/or inline `-q` flags; gold labels validated up front.
- Zero-dependency `TypeSafeProvider` (stdlib `urllib`, 429/529 backoff), `Provider` protocol and `provider_from_env` (`JEV_API_KEY`, `TYPESAFE_AI_BASE_URL`), from jev-rerank.
- Foundry Python stack: `mise run gate` (ruff, mypy strict, pytest with a 90% coverage floor, pip-audit, habit-hooks smells) and the Foundry gate/security/ratchet/bootstrap facades in CI.

### Roadmap

- v0.3: `compare` two runs (model drift, wording A/B), optional stdlib SVG reliability chart, per-slice reports.
- Post-hoc probability recalibration: roadmap only, not in the v0.x core.
