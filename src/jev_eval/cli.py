"""jev-eval CLI: run | report | thresholds | compare | calibrate. Only `run` touches the network."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TypeAlias

from . import __version__
from . import thresholds as th
from .calibrate import DEFAULT_FOLDS
from .commands import COMMANDS
from .provider import DEFAULT_MODEL

Sub: TypeAlias = "argparse._SubParsersAction[argparse.ArgumentParser]"


def _add_run(sub: Sub) -> None:
    p = sub.add_parser("run", help="call Jev once per labeled row and write the answer cache")
    p.add_argument("--data", required=True, help="labeled JSONL: {state, labels, meta?}")
    p.add_argument("--config", help="JSON question config (jev-sort's question schema)")
    p.add_argument("-q", "--question", action="append", default=[], help="inline name:choice(a,b) | name:noul | ...")
    p.add_argument("--cache", required=True, help="JSONL answer cache to write (reruns resume)")
    p.add_argument("--model", help=f"Jev model (default: config's, else {DEFAULT_MODEL})")


def _add_report(sub: Sub) -> None:
    p = sub.add_parser("report", help="accuracy, calibration and risk-coverage over a cache (offline)")
    p.add_argument("--cache", required=True)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--slice", metavar="SPEC", help="also break down by 'length' or 'meta.<field>'")
    p.add_argument("--svg", metavar="FILE", help="also write a reliability chart (SVG)")


def _add_thresholds(sub: Sub) -> None:
    p = sub.add_parser("thresholds", help="recommend cut-points and write thresholds.json (offline)")
    p.add_argument("--cache", required=True)
    goal = p.add_mutually_exclusive_group(required=True)
    goal.add_argument("--target-accuracy", type=float, help="smallest gate whose accuracy clears this")
    goal.add_argument("--max-escalation", type=float, help="strictest gate escalating at most this fraction")
    guard = p.add_mutually_exclusive_group()
    guard.add_argument(
        "--bootstrap",
        type=int,
        metavar="N",
        help=f"bootstrap with out-of-bag scoring (default guard, N={th.DEFAULT_BOOTSTRAP})",
    )
    guard.add_argument("--holdout", type=float, metavar="FRACTION", help="pick on train, report on held-out fraction")
    guard.add_argument("--no-guard", action="store_true", help="pick and score on the same rows (optimistic)")
    p.add_argument("--seed", type=int, default=th.DEFAULT_SEED)
    p.add_argument("--tolerance", type=int, default=0, help="Score: count within-N levels of gold as correct")
    p.add_argument("--out", default="thresholds.json", help="contract file to write ('-' for stdout only)")


def _add_compare(sub: Sub) -> None:
    p = sub.add_parser("compare", help="two runs over the same rows: model drift or question wording (offline)")
    p.add_argument("run_a", help="baseline cache (A)")
    p.add_argument("run_b", help="candidate cache (B)")
    p.add_argument("--tolerance", type=int, default=0, help="Score: count within-N levels of gold as correct")


def _add_calibrate(sub: Sub) -> None:
    p = sub.add_parser("calibrate", help="measure what a recalibration would fix; optional mapping file (offline)")
    p.add_argument("--cache", required=True)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--out", default="calibration.json", help="mapping file to write ('-' for stdout only)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev-eval", description="Measure and calibrate Jev on your labeled data.")
    parser.add_argument("--version", action="version", version=f"jev-eval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for add in (_add_run, _add_report, _add_thresholds, _add_compare, _add_calibrate):
        add(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (ValueError, RuntimeError, OSError) as err:
        print(f"jev-eval: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
