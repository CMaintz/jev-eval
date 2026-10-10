"""The subcommand bodies behind the CLI. Only `cmd_run` touches the network."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import thresholds as th
from .analysis import COMPOSITE, report, score_records
from .budget import STATE_BUDGET, estimate_tokens, near_limit, usage
from .cache import Record, load_cache, run, state_chars
from .calibrate import calibration_document, recalibration
from .check import check, load_thresholds, warnings
from .compare import compare
from .contract import GuardOptions, recommend_all, thresholds_document
from .dataset import LabeledRow, Questions, check_labels, load_config, load_rows, parse_inline
from .provider import DEFAULT_MODEL, provider_from_env
from .render import render_report, render_thresholds, render_usage
from .render_more import render_calibration, render_check, render_compare, render_slices, render_yes
from .reportsvg import reliability_svg
from .slices import slice_report
from .yes import yes_cuts, yes_field


def load(path: str) -> list[Record]:
    records = load_cache(path)
    if not records:
        raise ValueError(f"{path}: no cached answers (run `jev-eval run` first)")
    return records


def emit_json(doc: dict[str, Any], out: str, what: str) -> None:
    """Write a JSON document to `out`, or to stdout when `out` is '-'."""
    text = json.dumps(doc, indent=2) + "\n"
    if out == "-":
        sys.stdout.write(text)
        return
    Path(out).write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(doc['questions'])} {what})", file=sys.stderr)


def _questions(args: argparse.Namespace) -> tuple[Questions, str | None]:
    questions, model = load_config(args.config) if args.config else ({}, None)
    questions.update(dict(parse_inline(spec) for spec in args.question))
    if not questions:
        raise ValueError("no questions: pass --config or -q name:type(...)")
    return questions, model


def cmd_run(args: argparse.Namespace) -> int:
    questions, config_model = _questions(args)
    rows = load_rows(args.data)
    check_labels(rows, questions)
    model = args.model or config_model or DEFAULT_MODEL
    warn_near_limit(rows, questions)
    records = run(rows, questions, provider_from_env(model), model=model, cache_path=args.cache)
    answered = sum(1 for r in records if r["answer"] is not None)
    print(f"{len(rows)} rows, {len(records)} answers ({answered} usable) -> {args.cache}", file=sys.stderr)
    print(render_usage(usage(records)), file=sys.stderr)
    return 0


def warn_near_limit(rows: Sequence[LabeledRow], questions: Questions) -> None:
    """Before spending: flag rows whose state plus longest question is close to Jev's budget."""
    for i, row in enumerate(rows):
        asked = [q for qid, q in questions.items() if qid in row.labels]
        tokens = estimate_tokens(state_chars(row.state), asked)
        if near_limit(tokens):
            print(f"warning: row {i} is ~{tokens} tokens, near Jev's {STATE_BUDGET} budget", file=sys.stderr)


def cmd_report(args: argparse.Namespace) -> int:
    records = load(args.cache)
    summary = report(records, bins=args.bins)
    sys.stdout.write(render_report(summary))
    if args.slice:
        sys.stdout.write("\n" + render_slices(slice_report(records, args.slice, args.bins), args.slice))
    if args.svg:
        panels = list(summary["questions"].values()) + ([summary["composite"]] if summary["composite"] else [])
        Path(args.svg).write_text(reliability_svg(panels), encoding="utf-8")
        print(f"wrote {args.svg}", file=sys.stderr)
    return 0


def guard_options(args: argparse.Namespace) -> GuardOptions:
    if args.no_guard:
        return GuardOptions("none", 0, 0.0, args.seed)
    if args.holdout is not None:
        return GuardOptions("holdout", 0, args.holdout, args.seed)
    if args.bootstrap is not None:
        return GuardOptions("bootstrap", args.bootstrap, 0.0, args.seed)
    return GuardOptions(th.DEFAULT_GUARD, th.DEFAULT_BOOTSTRAP, th.DEFAULT_HOLDOUT, args.seed)


def cmd_thresholds(args: argparse.Namespace) -> int:
    goal = args.target_accuracy is not None or args.max_escalation is not None
    if not goal and not args.yes_precision:
        raise ValueError("pass --target-accuracy, --max-escalation and/or --yes-precision")
    scoring = score_records(load(args.cache), tolerance=args.tolerance)
    opts = guard_options(args)
    recs = recommend_all(scoring, th.picker(args.target_accuracy, args.max_escalation), opts) if goal else []
    cuts = yes_cuts(scoring, args.yes_precision, opts) if args.yes_precision else []
    sys.stdout.write(render_thresholds(recs, opts.guard) + render_yes(cuts, opts.guard))
    emit_json(thresholds_document(recs, scoring, opts, yes_field(cuts)), args.out, "questions")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Exit 1 when a recorded gate has drifted on the new rows, so a scheduled job can alert."""
    doc = load_thresholds(args.thresholds)
    scoring = score_records(load(args.cache), tolerance=int(doc.get("tolerance") or 0))
    checks = check(doc, scoring)
    sys.stdout.write(render_check(checks, warnings(doc, scoring)))
    return 1 if any(c.status == "drifted" for c in checks) else 0


def cmd_compare(args: argparse.Namespace) -> int:
    sys.stdout.write(render_compare(compare(load(args.run_a), load(args.run_b), args.tolerance)))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Measure-only: report what an isotonic remap would do, and write the optional mapping."""
    scoring = score_records(load(args.cache))
    usable = {q: items for q, items in scoring.items.items() if len(items) >= th.MIN_REFUSE and q != COMPOSITE}
    skipped = sorted(set(scoring.kinds) - set(usable))
    recs = [recalibration(q, scoring.kinds[q], usable[q], args.bins, args.folds) for q in sorted(usable)]
    sys.stdout.write(render_calibration(recs))
    if skipped:
        print(f"skipped (fewer than {th.MIN_REFUSE} labeled rows): {', '.join(skipped)}", file=sys.stderr)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    emit_json(calibration_document(recs, scoring.model, stamp), args.out, "questions")
    return 0


COMMANDS = {
    "run": cmd_run,
    "report": cmd_report,
    "thresholds": cmd_thresholds,
    "compare": cmd_compare,
    "calibrate": cmd_calibrate,
    "check": cmd_check,
}
