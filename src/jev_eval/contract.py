"""Per-question recommendations and the family-wide `thresholds.json` contract (version 1).

    {"version": 1, "model": str, "generatedAt": ISO-8601 UTC,
     "guard": {"method": "bootstrap" | "holdout" | "none", ...}, "tolerance": int,
     "questions": {id: {"type", "threshold", "accuracy", "coverage", "n"}},
     "composite"?: {"threshold", "accuracy", "coverage", "n", "questions": [id, ...]}}

`accuracy`/`coverage` are what the gate buys on rows the pick did not see: the mean
out-of-bag estimate under bootstrap, the held-out split under holdout, and the (optimistic)
same-rows numbers only under "none". `tolerance` > 0 means Score answers within that many
levels of gold counted as correct. A Noul threshold applies to the Noul confidence
|noul - 0.5| * 2 (a two-sided band around 0.5), never to the raw P(yes). Questions refused
for too few labels, unstable under the guard, or with no gate that meets the goal are
omitted rather than written as null.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .analysis import COMPOSITE, Scoring, composite, gated_ids
from .metrics import Point, selective_accuracy
from .score import Scored
from .thresholds import MIN_REFUSE, MIN_WARN, Guard, Interval, Picker, bootstrap, holdout

CONTRACT_VERSION = 1


@dataclass(frozen=True)
class GuardOptions:
    guard: Guard
    resamples: int
    fraction: float
    seed: int


@dataclass(frozen=True)
class Recommendation:
    qid: str
    kind: str
    n: int
    status: str  # ok | warn (below MIN_WARN) | refused (below MIN_REFUSE) | unreachable | unstable
    point: Point | None = None  # the gate and what it buys under the guard
    interval: Interval | None = None
    in_sample: Point | None = None  # the same gate scored on the rows it was picked on

    @property
    def usable(self) -> bool:
        return self.point is not None and self.status in ("ok", "warn")


def _guarded(items: Sequence[Scored], pick: Picker, opts: GuardOptions) -> tuple[Point | None, Interval | None]:
    """(the reported gate, the bootstrap ranges). Under bootstrap the gate is the all-rows pick
    with its accuracy/coverage replaced by the out-of-bag means."""
    rng = random.Random(opts.seed)
    if opts.guard == "holdout":
        return holdout(items, pick, opts.fraction, rng), None
    point = pick(items)
    if opts.guard != "bootstrap" or point is None:
        return point, None
    interval = bootstrap(items, pick, opts.resamples, rng)
    if interval is None:
        return None, None
    covered = round(interval.oob_coverage * len(items))
    return Point(point.threshold, interval.oob_coverage, interval.oob_accuracy, covered), interval


def recommend(qid: str, kind: str, items: Sequence[Scored], pick: Picker, opts: GuardOptions) -> Recommendation:
    """Refuse below MIN_REFUSE labeled rows, warn below MIN_WARN, else pick under the guard."""
    if len(items) < MIN_REFUSE:
        return Recommendation(qid, kind, len(items), "refused")
    in_sample = pick(items)
    if in_sample is None:
        return Recommendation(qid, kind, len(items), "unreachable")
    point, interval = _guarded(items, pick, opts)
    if point is None:
        return Recommendation(qid, kind, len(items), "unstable", in_sample=in_sample)
    status = "warn" if len(items) < MIN_WARN else "ok"
    return Recommendation(qid, kind, len(items), status, point, interval, same_gate_all_rows(items, point))


def same_gate_all_rows(items: Sequence[Scored], point: Point) -> Point:
    """The reported gate scored on every row (optimistic): the contrast for the guarded numbers.

    Under holdout the reported gate was picked on the train split, so this re-scores that gate
    rather than reusing the all-rows pick, which may be a different threshold.
    """
    acc, cov = selective_accuracy(items, point.threshold)
    return Point(point.threshold, cov, acc, round(cov * len(items)))


def recommend_all(scoring: Scoring, pick: Picker, opts: GuardOptions) -> list[Recommendation]:
    """Every question, then the row-level composite gate when more than one question is gated."""
    recs = [recommend(q, scoring.kinds[q], scoring.items.get(q, []), pick, opts) for q in sorted(scoring.kinds)]
    if len(gated_ids(scoring)) > 1:
        recs.append(recommend(COMPOSITE, "composite", composite(scoring), pick, opts))
    return recs


def _entry(rec: Recommendation) -> dict[str, Any]:
    point = rec.point or Point(0.0, 0.0, 0.0, 0)  # only called for usable recommendations
    return {
        "threshold": round(point.threshold, 4),
        "accuracy": round(point.accuracy, 4),
        "coverage": round(point.coverage, 4),
        "n": rec.n,
    }


def guard_field(opts: GuardOptions) -> dict[str, Any]:
    """How `accuracy`/`coverage` were measured, so a consumer knows what they mean."""
    if opts.guard == "bootstrap":
        return {"method": "bootstrap", "resamples": opts.resamples, "seed": opts.seed}
    if opts.guard == "holdout":
        return {"method": "holdout", "fraction": opts.fraction, "seed": opts.seed}
    return {"method": "none"}


def thresholds_document(
    recs: Sequence[Recommendation],
    model: str,
    gated: Sequence[str],
    opts: GuardOptions,
    tolerance: int = 0,
    now: datetime | None = None,
) -> dict[str, Any]:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    usable = [rec for rec in recs if rec.usable]
    doc: dict[str, Any] = {
        "version": CONTRACT_VERSION,
        "model": model,
        "generatedAt": stamp,
        "guard": guard_field(opts),
        "tolerance": tolerance,
        "questions": {rec.qid: {"type": rec.kind, **_entry(rec)} for rec in usable if rec.qid != COMPOSITE},
    }
    for rec in usable:
        if rec.qid == COMPOSITE:
            doc["composite"] = {**_entry(rec), "questions": list(gated)}
    return doc
