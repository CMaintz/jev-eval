"""Measure-only recalibration: how much would a monotone remap of Jev's confidence fix ECE?

jev-eval does not wrap Jev or rewrite its answers. This module answers a question about the
measurement: if you remapped the raw confidence (Choice/Score) or P(yes) (Noul) with an
isotonic fit, how calibrated would it be on rows the fit never saw (k-fold)? It can also
emit the fitted mapping as an optional `calibration.json`; nothing in the family has to load it.

Because the fit is monotone, remapping a Choice/Score confidence never changes which rows a
gate auto-handles; the thresholds in `thresholds.json` stay on raw values. Noul differs: a
monotone map of P(yes) can move the 0.5 crossing, so `noul_midpoint` reports where calibrated
0.5 sits in raw terms ("Jev's 0.5 is really 0.42").
"""

from __future__ import annotations

import bisect
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .metrics import bin_by_confidence, ece
from .score import Scored

DEFAULT_FOLDS = 5


@dataclass(frozen=True)
class Step:
    """One block of the isotonic fit: raw values in [x_lo, x_hi] (observed range) map to `value`."""

    x_lo: float
    x_hi: float
    value: float


def _tie_groups(pairs: Sequence[tuple[float, bool]]) -> list[list[float]]:
    """[x, x, outcome_sum, count] per distinct raw value: tied values share one calibrated value."""
    groups: list[list[float]] = []
    for x, outcome in sorted(pairs):
        if groups and groups[-1][1] == x:
            groups[-1][2:] = [groups[-1][2] + outcome, groups[-1][3] + 1]
        else:
            groups.append([x, x, float(outcome), 1.0])
    return groups


def isotonic(pairs: Sequence[tuple[float, bool]]) -> list[Step]:
    """Pool-adjacent-violators: the non-decreasing step function closest to the outcomes.

    Tie groups are complete before pooling starts; pooling point by point would let each
    group's first-sorted outcomes look like a violation on their own.
    """
    blocks: list[list[float]] = []  # [x_lo, x_hi, outcome_sum, count]
    for group in _tie_groups(pairs):
        blocks.append(group)
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] >= blocks[-1][2] / blocks[-1][3]:
            last = blocks.pop()
            blocks[-1] = [blocks[-1][0], last[1], blocks[-1][2] + last[2], blocks[-1][3] + last[3]]
    return [Step(lo, hi, total / count) for lo, hi, total, count in blocks]


def apply(steps: Sequence[Step], x: float) -> float:
    """Map a raw value through the fit (above the last block: the last block's value)."""
    i = bisect.bisect_left([s.x_hi for s in steps], x)
    return steps[min(i, len(steps) - 1)].value


def _remapped(items: Sequence[Scored], steps: Sequence[Step]) -> list[Scored]:
    return [replace(item, prob=apply(steps, item.prob)) for item in items]


def cross_validated_ece(items: Sequence[Scored], bins: int = 10, folds: int = DEFAULT_FOLDS, seed: int = 0) -> float:
    """ECE after recalibration, each row remapped by a fit on the other folds only."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    held_out: list[Scored] = []
    for k in range(folds):
        test = {i for pos, i in enumerate(order) if pos % folds == k}
        steps = isotonic([(items[i].prob, items[i].outcome) for i in order if i not in test])
        held_out += _remapped([items[i] for i in sorted(test)], steps) if steps else []
    return ece(bin_by_confidence(held_out, bins))


def noul_midpoint(steps: Sequence[Step]) -> float | None:
    """The smallest observed raw P(yes) whose calibrated value reaches 0.5, or None if none does."""
    return next((s.x_lo for s in steps if s.value >= 0.5), None)


@dataclass(frozen=True)
class Recalibration:
    """What an isotonic remap would do for one question (measured, never applied)."""

    qid: str
    kind: str
    n: int
    ece_raw: float
    ece_cv: float
    steps: list[Step]
    midpoint: float | None = None  # Noul only: raw P(yes) where calibrated 0.5 begins


def recalibration(
    qid: str, kind: str, items: Sequence[Scored], bins: int = 10, folds: int = DEFAULT_FOLDS
) -> Recalibration:
    """Raw ECE, cross-validated recalibrated ECE, and the all-rows fit for the mapping file."""
    steps = isotonic([(item.prob, item.outcome) for item in items])
    raw = ece(bin_by_confidence(items, bins))
    midpoint = noul_midpoint(steps) if kind == "noul" else None
    return Recalibration(qid, kind, len(items), raw, cross_validated_ece(items, bins, folds), steps, midpoint)


def calibration_document(recs: Sequence[Recalibration], model: str, stamp: str) -> dict[str, Any]:
    """The optional mapping file. Each step is [x_lo, x_hi, calibrated]; a raw value maps to the
    first step whose x_hi is >= it (the last step above that). `input` names the raw value."""
    return {
        "version": 1,
        "model": model,
        "generatedAt": stamp,
        "method": "isotonic",
        "questions": {
            r.qid: {
                "type": r.kind,
                "input": "noul" if r.kind == "noul" else "confidence",
                "n": r.n,
                "eceRaw": round(r.ece_raw, 4),
                "eceRecalibrated": round(r.ece_cv, 4),
                "steps": [[round(s.x_lo, 6), round(s.x_hi, 6), round(s.value, 6)] for s in r.steps],
            }
            for r in recs
        },
    }
