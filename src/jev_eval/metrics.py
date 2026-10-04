"""Pure metrics over scored rows: calibration (ECE/MCE, reliability) and risk-coverage."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .score import Scored

DEFAULT_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    count: int
    mean_prob: float
    rate: float

    @property
    def gap(self) -> float:
        return abs(self.mean_prob - self.rate)


@dataclass(frozen=True)
class Point:
    """One row of a risk-coverage table: gate at `threshold`, auto-handle `coverage` at `accuracy`."""

    threshold: float
    coverage: float
    accuracy: float
    covered: int

    @property
    def escalation(self) -> float:
        return 1.0 - self.coverage


def _bin_index(prob: float, n: int) -> int:
    return min(max(int(prob * n), 0), n - 1)


def bin_by_confidence(items: Sequence[Scored], n: int = 10) -> list[Bin]:
    """Equal-width bins over [0, 1] of `prob` vs the empirical rate of `outcome` (empty bins kept)."""
    groups: list[list[Scored]] = [[] for _ in range(n)]
    for item in items:
        groups[_bin_index(item.prob, n)].append(item)
    return [_make_bin(i, n, group) for i, group in enumerate(groups)]


def _make_bin(i: int, n: int, group: list[Scored]) -> Bin:
    if not group:
        return Bin(i / n, (i + 1) / n, 0, 0.0, 0.0)
    mean = sum(item.prob for item in group) / len(group)
    rate = sum(1 for item in group if item.outcome) / len(group)
    return Bin(i / n, (i + 1) / n, len(group), mean, rate)


def ece(bins: Sequence[Bin]) -> float:
    """Expected Calibration Error: the count-weighted mean |confidence - empirical rate|."""
    total = sum(b.count for b in bins)
    return sum(b.count * b.gap for b in bins) / total if total else 0.0


def mce(bins: Sequence[Bin]) -> float:
    """Max Calibration Error: the worst non-empty bin's gap."""
    return max((b.gap for b in bins if b.count), default=0.0)


def accuracy(items: Sequence[Scored]) -> float:
    return sum(1 for item in items if item.correct) / len(items) if items else 0.0


def selective_accuracy(items: Sequence[Scored], t: float) -> tuple[float, float]:
    """Accuracy among rows at or above confidence t, and the coverage fraction."""
    covered = [item for item in items if item.confidence >= t]
    if not covered:
        return (0.0, 0.0)
    return (accuracy(covered), len(covered) / len(items))


def risk_coverage(items: Sequence[Scored], grid: Sequence[float] = DEFAULT_GRID) -> list[Point]:
    """The headline table at fixed candidate thresholds."""
    points = []
    for t in grid:
        acc, cov = selective_accuracy(items, t)
        points.append(Point(t, cov, acc, round(cov * len(items))))
    return points


def risk_coverage_curve(items: Sequence[Scored]) -> list[Point]:
    """Every achievable gate (each distinct confidence), highest first, in one sorted sweep."""
    ordered = sorted(items, key=lambda item: item.confidence, reverse=True)
    points: list[Point] = []
    correct = 0
    for i, item in enumerate(ordered, 1):
        correct += item.correct
        if i == len(ordered) or ordered[i].confidence != item.confidence:
            points.append(Point(item.confidence, i / len(ordered), correct / i, i))
    return points


def yes_precision_recall(items: Sequence[Scored], t: float) -> tuple[float, float]:
    """Noul precision and recall on "yes" among the rows auto-decided at threshold t."""
    covered = [item for item in items if item.confidence >= t]
    true_pos = sum(1 for item in covered if item.predicted and item.gold)
    said_yes = sum(1 for item in covered if item.predicted)
    were_yes = sum(1 for item in covered if item.gold)
    return (true_pos / said_yes if said_yes else 0.0, true_pos / were_yes if were_yes else 0.0)


def within(items: Sequence[Scored], levels: int) -> float:
    """Score: fraction landing within `levels` of the gold level."""
    scored = [item for item in items if item.distance is not None]
    return sum(1 for item in scored if (item.distance or 0) <= levels) / len(scored) if scored else 0.0


def mean_distance(items: Sequence[Scored]) -> float:
    """Score: mean ordinal distance from the gold level (0 = always exact)."""
    gaps = [item.distance for item in items if item.distance is not None]
    return sum(gaps) / len(gaps) if gaps else 0.0


def confusion(items: Sequence[Scored]) -> Counter[tuple[str, str]]:
    """(gold, predicted) pair counts - the confusion summary behind raw accuracy."""
    return Counter((str(item.gold), str(item.predicted)) for item in items)
