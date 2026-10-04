"""jev-eval CLI: run | report | thresholds. Only `run` touches the network."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import thresholds as th
from .analysis import gated_ids, report, score_records
from .cache import load_cache, run
from .contract import GuardOptions, recommend_all, thresholds_document
from .dataset import Questions, check_labels, load_config, load_rows, parse_inline
from .provider import DEFAULT_MODEL, provider_from_env
from .render import render_report, render_thresholds


def _add_run(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("run", help="call Jev once per labeled row and write the answer cache")
    p.add_argument("--data", required=True, help="labeled JSONL: {state, labels}")
    p.add_argument("--config", help="JSON question config (jev-sort's question schema)")
    p.add_argument("-q", "--question", action="append", default=[], help="inline name:choice(a,b) | name:noul | ...")
    p.add_argument("--cache", required=True, help="JSONL answer cache to write (reruns resume)")
    p.add_argument("--model", help=f"Jev model (default: config's, else {DEFAULT_MODEL})")


def _add_report(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("report", help="accuracy, calibration and risk-coverage over a cache (offline)")
    p.add_argument("--cache", required=True)
    p.add_argument("--bins", type=int, default=10)


def _add_thresholds(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("thresholds", help="recommend cut-points and write thresholds.json (offline)")
    p.add_argument("--cache", required=True)
    goal = p.add_mutually_exclusive_group(required=True)
    goal.add_argument("--target-accuracy", type=float, help="smallest gate whose accuracy clears this")
    goal.add_argument("--max-escalation", type=float, help="strictest gate escalating at most this fraction")
    guard = p.add_mutually_exclusive_group()
    guard.add_argument("--bootstrap", type=int, metavar="N", help="bootstrap range over N resamples")
    guard.add_argument("--holdout", type=float, metavar="FRACTION", help="pick on train, report on held-out fraction")
    guard.add_argument("--no-guard", action="store_true", help="pick and score on the same rows (optimistic)")
    p.add_argument("--seed", type=int, default=th.DEFAULT_SEED)
    p.add_argument("--tolerance", type=int, default=0, help="Score: count within-N levels of gold as correct")
    p.add_argument("--out", default="thresholds.json", help="contract file to write ('-' for stdout only)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev-eval", description="Measure and calibrate Jev on your labeled data.")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run(sub)
    _add_report(sub)
    _add_thresholds(sub)
    return parser


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
    records = run(rows, questions, provider_from_env(model), model=model, cache_path=args.cache)
    answered = sum(1 for r in records if r["answer"] is not None)
    print(f"{len(rows)} rows, {len(records)} answers ({answered} usable) -> {args.cache}", file=sys.stderr)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    records = _cache(args.cache)
    sys.stdout.write(render_report(report(records, bins=args.bins)))
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
    scoring = score_records(_cache(args.cache), tolerance=args.tolerance)
    opts = guard_options(args)
    recs = recommend_all(scoring, th.picker(args.target_accuracy, args.max_escalation), opts)
    sys.stdout.write(render_thresholds(recs, opts.guard))
    doc = thresholds_document(recs, scoring.model, gated_ids(scoring))
    text = json.dumps(doc, indent=2) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(doc['questions'])} questions)", file=sys.stderr)
    return 0


def _cache(path: str) -> list[dict[str, object]]:
    records = load_cache(path)
    if not records:
        raise ValueError(f"{path}: no cached answers (run `jev-eval run` first)")
    return records


COMMANDS = {"run": cmd_run, "report": cmd_report, "thresholds": cmd_thresholds}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (ValueError, RuntimeError, OSError) as err:
        print(f"jev-eval: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
