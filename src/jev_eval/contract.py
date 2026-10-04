"""Per-question recommendations and the family-wide `thresholds.json` contract (version 1).

    {"version": 1, "model": str, "generatedAt": ISO-8601 UTC,
     "questions": {id: {"type", "threshold", "accuracy", "coverage", "n"}},
     "composite"?: {"threshold", "accuracy", "coverage", "n", "questions": [id, ...]}}

A Noul threshold applies to the Noul confidence |noul - 0.5| * 2 (a two-sided band around
0.5), never to the raw P(yes). Questions refused for too few labels, or with no threshold
that meets the goal, are omitted rather than written as null.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .analysis import COMPOSITE, Scoring, composite, gated_ids
from .metrics import Point
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
    status: str  # ok | warn (below MIN_WARN) | refused (below MIN_REFUSE) | unreachable
    point: Point | None = None
    interval: Interval | None = None

    @property
    def usable(self) -> bool:
        return self.point is not None and self.status in ("ok", "warn")


def _guarded(items: Sequence[Scored], pick: Picker, opts: GuardOptions) -> tuple[Point | None, Interval | None]:
    rng = random.Random(opts.seed)
    if opts.guard == "holdout":
        return holdout(items, pick, opts.fraction, rng), None
    point = pick(items)
    if opts.guard == "bootstrap" and point is not None:
        return point, bootstrap(items, pick, opts.resamples, rng)
    return point, None


def recommend(qid: str, kind: str, items: Sequence[Scored], pick: Picker, opts: GuardOptions) -> Recommendation:
    """Refuse below MIN_REFUSE labeled rows, warn below MIN_WARN, else pick under the guard."""
    if len(items) < MIN_REFUSE:
        return Recommendation(qid, kind, len(items), "refused")
    point, interval = _guarded(items, pick, opts)
    if point is None:
        return Recommendation(qid, kind, len(items), "unreachable")
    status = "warn" if len(items) < MIN_WARN else "ok"
    return Recommendation(qid, kind, len(items), status, point, interval)


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


def thresholds_document(
    recs: Sequence[Recommendation], model: str, gated: Sequence[str], now: datetime | None = None
) -> dict[str, Any]:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    usable = [rec for rec in recs if rec.usable]
    doc: dict[str, Any] = {
        "version": CONTRACT_VERSION,
        "model": model,
        "generatedAt": stamp,
        "questions": {rec.qid: {"type": rec.kind, **_entry(rec)} for rec in usable if rec.qid != COMPOSITE},
    }
    for rec in usable:
        if rec.qid == COMPOSITE:
            doc["composite"] = {**_entry(rec), "questions": list(gated)}
    return doc
