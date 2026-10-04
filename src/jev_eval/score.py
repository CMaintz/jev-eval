"""Answer-vs-gold per question type, and the confidence each type gates on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Scored:
    """One (row, question) outcome.

    `confidence`/`correct` drive gating. `prob`/`outcome` drive the reliability table: for
    Choice/Score they equal confidence/correct (top-label calibration); for Noul they are the
    raw P(yes) against the gold yes, so a Noul reliability bin compares p to the yes-rate.
    `distance` is the ordinal level gap for Score (None for the other types).
    """

    row: int
    kind: str
    confidence: float
    correct: bool
    prob: float
    outcome: bool
    predicted: Any
    gold: Any
    distance: int | None = None


def _argmax(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=lambda key: float(probabilities[key]))


def choice_prediction(answer: dict[str, Any]) -> tuple[Any, float]:
    """(argmax label, confidence) - probabilities win over the bare `choice` field."""
    probs = answer.get("probabilities")
    label = _argmax(probs) if isinstance(probs, dict) and probs else answer.get("choice")
    conf = answer.get("confidence")
    if conf is None and isinstance(probs, dict) and probs:
        conf = max(float(p) for p in probs.values())
    return label, float(conf or 0.0)


def nearest_level(value: float, levels: list[str], legend: Any = None) -> int:
    """Index of the criterion label nearest the score.

    With a numeric `legend` ({label: position}) the nearest legend position wins; otherwise
    the score is read as a level index (0..N-1), as jev-rerank does.
    """
    if isinstance(legend, dict) and all(isinstance(legend.get(lv), (int, float)) for lv in levels):
        return min(range(len(levels)), key=lambda i: abs(float(legend[levels[i]]) - value))
    return min(max(round(value), 0), len(levels) - 1)


def score_choice(row: int, answer: dict[str, Any], gold: Any) -> Scored:
    label, conf = choice_prediction(answer)
    ok = label == gold
    return Scored(row, "choice", conf, ok, conf, ok, label, gold)


def score_noul(row: int, answer: dict[str, Any], gold: bool) -> Scored:
    p = float(answer["noul"])
    predicted = p >= 0.5
    return Scored(row, "noul", abs(p - 0.5) * 2, predicted == gold, p, bool(gold), predicted, gold)


def score_score(row: int, answer: dict[str, Any], gold: Any, levels: list[str]) -> Scored:
    index = nearest_level(float(answer["score"]), levels, answer.get("legend"))
    distance = abs(index - levels.index(gold))
    conf = float(answer.get("confidence") or 0.0)
    return Scored(row, "score", conf, distance == 0, conf, distance == 0, levels[index], gold, distance)


def score_answer(row: int, question: dict[str, Any], answer: Any, gold: Any) -> Scored | None:
    """Compare one answer to its gold; None when Jev returned no usable answer."""
    if not isinstance(answer, dict):
        return None
    kind = question["type"]
    try:
        if kind == "choice":
            return score_choice(row, answer, gold)
        if kind == "noul":
            return score_noul(row, answer, gold)
        return score_score(row, answer, gold, list(question["criteria"]))
    except (KeyError, TypeError, ValueError):
        return None


def with_tolerance(item: Scored, tolerance: int) -> Scored:
    """Widen Score correctness to within-N levels of gold (thresholds' --tolerance)."""
    if item.distance is None or tolerance <= 0:
        return item
    ok = item.distance <= tolerance
    return Scored(item.row, item.kind, item.confidence, ok, item.prob, ok, item.predicted, item.gold, item.distance)
