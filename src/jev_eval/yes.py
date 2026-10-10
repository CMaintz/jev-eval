"""Yes cut-points: the lowest raw P(yes) at which a Noul's "yes" is right often enough to act on.

Leash's repairAt/noteAt and jev-guard's Noul cut-offs act when P(yes) is high, which the
two-sided Noul gate (|p - 0.5| * 2) does not answer. Here a row is flagged when p >= t, and
the target is the precision of those flags (how often a flagged row really is a yes). Each
Noul item is re-read as confidence = p and correct = gold is yes, so precision is selective
accuracy, the flag rate is coverage, and the existing bootstrap/holdout guard applies as is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .analysis import Scoring
from .contract import GuardOptions, Recommendation, recommend
from .score import Scored
from .thresholds import picker


@dataclass(frozen=True)
class YesCut:
    """The flag-at-yes gate for one precision target."""

    qid: str
    target: float
    rec: Recommendation
    base_rate: float  # share of labeled rows that are yes

    @property
    def recall(self) -> float:
        """Share of the real yeses the flag catches (precision x flag rate / yes rate)."""
        point = self.rec.point
        return point.accuracy * point.coverage / self.base_rate if point and self.base_rate else 0.0


def as_yes_items(items: Sequence[Scored]) -> list[Scored]:
    return [replace(item, confidence=item.prob, correct=bool(item.gold)) for item in items]


def yes_cut(qid: str, items: Sequence[Scored], target: float, opts: GuardOptions) -> YesCut:
    flags = as_yes_items(items)
    base = sum(item.correct for item in flags) / len(flags) if flags else 0.0
    return YesCut(qid, target, recommend(qid, "noul", flags, picker(target, None), opts), base)


def yes_cuts(scoring: Scoring, targets: Sequence[float], opts: GuardOptions) -> list[YesCut]:
    """Every Noul question at every target, strictest target first."""
    nouls = sorted(qid for qid, kind in scoring.kinds.items() if kind == "noul")
    ordered = sorted(targets, reverse=True)
    return [yes_cut(qid, scoring.items.get(qid, []), t, opts) for qid in nouls for t in ordered]


def yes_field(cuts: Sequence[YesCut]) -> dict[str, list[dict[str, Any]]]:
    """The contract's `yesAt`: {id: [{target, threshold, precision, recall, flagged, n}]}."""
    out: dict[str, list[dict[str, Any]]] = {}
    for cut in cuts:
        point = cut.rec.point
        if cut.rec.usable and point is not None:
            out.setdefault(cut.qid, []).append(
                {
                    "target": cut.target,
                    "threshold": round(point.threshold, 4),
                    "precision": round(point.accuracy, 4),
                    "recall": round(cut.recall, 4),
                    "flagged": round(point.coverage, 4),
                    "n": cut.rec.n,
                }
            )
    return out
