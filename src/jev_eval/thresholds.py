"""Recommend confidence cut-points, with an honesty guard and a min-N refusal.

The guard default is a single constant (DEFAULT_GUARD) so the pending bootstrap-vs-holdout
decision (spec section 14.2) is a one-line flip.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from .metrics import Point, risk_coverage_curve, selective_accuracy
from .score import Scored

Guard = Literal["bootstrap", "holdout", "none"]
DEFAULT_GUARD: Guard = "bootstrap"
DEFAULT_BOOTSTRAP = 1000
DEFAULT_HOLDOUT = 0.3
DEFAULT_SEED = 0
MIN_REFUSE = 50
MIN_WARN = 200

Picker = Callable[[Sequence[Scored]], "Point | None"]


def recommend_threshold(
    items: Sequence[Scored], *, target_accuracy: float | None = None, max_escalation: float | None = None
) -> Point | None:
    """The gate for exactly one goal; None when no achievable threshold meets it.

    target_accuracy: the smallest t whose selective accuracy clears the target (most coverage).
    max_escalation: the largest t whose escalation rate stays within budget (best accuracy).
    """
    if (target_accuracy is None) == (max_escalation is None):
        raise ValueError("pass exactly one of target_accuracy or max_escalation")
    curve = risk_coverage_curve(items)
    if target_accuracy is not None:
        hits = [p for p in curve if p.accuracy >= target_accuracy]
        return max(hits, key=lambda p: p.coverage) if hits else None
    floor = 1.0 - float(max_escalation or 0.0) - 1e-12
    return next((p for p in curve if p.coverage >= floor), None)


def picker(target_accuracy: float | None, max_escalation: float | None) -> Picker:
    def pick(items: Sequence[Scored]) -> Point | None:
        return recommend_threshold(items, target_accuracy=target_accuracy, max_escalation=max_escalation)

    return pick


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 1]) - enough for a reported range, no numpy."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


@dataclass(frozen=True)
class Interval:
    """Bootstrap result: 95% ranges plus the honest out-of-bag (OOB) estimate.

    Each resample picks a gate on its in-bag rows and scores it on the rows that resample
    left out, so `oob_accuracy`/`oob_coverage` are measured on rows the pick never saw. OOB
    is slightly pessimistic (each pick sees ~63% of the unique rows), the safe direction.
    """

    threshold: tuple[float, float]
    accuracy: tuple[float, float]
    oob_accuracy: float
    oob_coverage: float
    resamples: int
    misses: int


def _resample(items: Sequence[Scored], pick: Picker, rng: random.Random) -> tuple[Point | None, Point | None]:
    """One bootstrap draw: (the in-bag pick, that pick scored on the out-of-bag rows)."""
    drawn = [rng.randrange(len(items)) for _ in items]
    point = pick([items[i] for i in drawn])
    in_bag = set(drawn)
    oob = [item for i, item in enumerate(items) if i not in in_bag]
    if point is None or not oob:
        return point, None
    acc, cov = selective_accuracy(oob, point.threshold)
    return point, (Point(point.threshold, cov, acc, round(cov * len(oob))) if cov else None)


def bootstrap(items: Sequence[Scored], pick: Picker, resamples: int, rng: random.Random) -> Interval | None:
    """95% range on the gate and its out-of-bag accuracy, plus the mean OOB accuracy/coverage."""
    picks: list[Point] = []
    scored: list[Point] = []
    for _ in range(resamples):
        point, oob = _resample(items, pick, rng)
        picks += [point] if point else []
        scored += [oob] if oob else []
    if not picks or not scored:
        return None
    ts, accs = [p.threshold for p in picks], [p.accuracy for p in scored]
    return Interval(
        (percentile(ts, 0.025), percentile(ts, 0.975)),
        (percentile(accs, 0.025), percentile(accs, 0.975)),
        sum(accs) / len(accs),
        sum(p.coverage for p in scored) / len(scored),
        resamples,
        resamples - len(picks),
    )


def holdout(items: Sequence[Scored], pick: Picker, fraction: float, rng: random.Random) -> Point | None:
    """Pick t on a train split, then report what it buys on the unseen test split."""
    shuffled = list(items)
    rng.shuffle(shuffled)
    cut = max(1, round(len(shuffled) * fraction))
    test, train = shuffled[:cut], shuffled[cut:]
    chosen = pick(train)
    if chosen is None:
        return None
    acc, cov = selective_accuracy(test, chosen.threshold)
    return Point(chosen.threshold, cov, acc, round(cov * len(test)))
