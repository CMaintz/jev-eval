"""A deterministic fake Jev and a synthetic labeled dataset, roughly calibrated by construction."""

from __future__ import annotations

from typing import Any

from jev_eval import LabeledRow

TEAMS = ["billing", "tech", "sales"]
LEVELS = ["negative", "neutral", "positive"]
QUESTIONS: dict[str, dict[str, Any]] = {
    "team": {"type": "choice", "instructions": "team", "criteria": {t: t for t in TEAMS}},
    "urgent": {"type": "noul", "instructions": "urgent"},
    "sentiment": {"type": "score", "instructions": "sentiment", "criteria": LEVELS},
}


def confidence(i: int) -> float:
    """Spread confidences over [0.35, 0.99]."""
    return 0.35 + (i * 37 % 65) / 100


def hit(i: int, p: float) -> bool:
    """Correct with probability ~p, deterministically (a low-discrepancy draw)."""
    return (i * 0.6180339887) % 1.0 < p


def labels(i: int) -> dict[str, Any]:
    return {"team": TEAMS[i % 3], "urgent": i % 2 == 0, "sentiment": LEVELS[i % 3]}


def make_rows(n: int) -> list[LabeledRow]:
    return [LabeledRow({"id": i, "text": f"ticket {i}"}, labels(i)) for i in range(n)]


def _team(i: int) -> dict[str, Any]:
    c = confidence(i)
    gold = TEAMS[i % 3]
    pick = gold if hit(i, c) else TEAMS[(i + 1) % 3]
    rest = (1 - c) / 2
    return {"choice": pick, "confidence": c, "probabilities": {t: (c if t == pick else rest) for t in TEAMS}}


def _urgent(i: int) -> dict[str, Any]:
    p = confidence(i)
    yes = i % 2 == 0
    said_yes = yes if hit(i, p) else not yes
    return {"noul": (0.5 + p / 2) if said_yes else (0.5 - p / 2)}


def _sentiment(i: int) -> dict[str, Any]:
    c = confidence(i)
    gold = i % 3
    level = gold if hit(i, c) else (gold + 1) % 3
    return {"score": float(level), "confidence": c}


ANSWERS = {"team": _team, "urgent": _urgent, "sentiment": _sentiment}


class FakeProvider:
    """Answers each asked question from the state's id; records every call."""

    model = "jev-fake"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"state": state, "questions": dict(questions)})
        answers = {qid: ANSWERS[qid](int(state["id"])) for qid in questions}
        return {"model": self.model, "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 2}}
