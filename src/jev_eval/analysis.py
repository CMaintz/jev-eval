"""Pure analysis over a run cache: score every record, then summarize per question."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from . import metrics
from .cache import Record
from .calibrate import Recalibration, recalibration
from .score import Scored, score_answer, with_tolerance

COMPOSITE = "_row"
RECAL_MIN = 50  # below this a cross-validated remap is noise; matches the thresholds refusal floor


@dataclass(frozen=True)
class Scoring:
    """Scored items per question id, plus how many records had no usable answer."""

    items: dict[str, list[Scored]]
    kinds: dict[str, str]
    unanswered: dict[str, int]
    model: str


def score_records(records: Sequence[Record], tolerance: int = 0) -> Scoring:
    items: dict[str, list[Scored]] = defaultdict(list)
    kinds: dict[str, str] = {}
    unanswered: dict[str, int] = defaultdict(int)
    for rec in records:
        kinds[rec["id"]] = rec["type"]
        scored = score_answer(rec["row"], rec["question"], rec["answer"], rec["gold"])
        if scored is None:
            unanswered[rec["id"]] += 1
        else:
            items[rec["id"]].append(with_tolerance(scored, tolerance))
    models = sorted({str(rec["model"]) for rec in records})
    return Scoring(dict(items), kinds, dict(unanswered), ", ".join(models) or "unknown")


def gated_ids(scoring: Scoring) -> list[str]:
    """The questions jev-sort gates a row on: Choice and Score (Noul carries no gate confidence)."""
    return sorted(qid for qid, kind in scoring.kinds.items() if kind in ("choice", "score"))


def composite(scoring: Scoring) -> list[Scored]:
    """Row-level gate: min gated confidence per row; correct only if every gated answer is.

    Only rows scored on every gated question count, so the gate is measured like for like.
    """
    ids = gated_ids(scoring)
    per_row: dict[int, list[Scored]] = defaultdict(list)
    for qid in ids:
        for item in scoring.items.get(qid, []):
            per_row[item.row].append(item)
    return [_combine(row, group) for row, group in sorted(per_row.items()) if len(group) == len(ids)]


def _combine(row: int, group: list[Scored]) -> Scored:
    conf = min(item.confidence for item in group)
    ok = all(item.correct for item in group)
    return Scored(row, "composite", conf, ok, conf, ok, None, None)


@dataclass(frozen=True)
class QuestionReport:
    qid: str
    kind: str
    n: int
    unanswered: int
    accuracy: float
    ece: float
    mce: float
    reliability: list[metrics.Bin]
    risk_coverage: list[metrics.Point]
    extras: dict[str, float] = field(default_factory=dict)
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    recalibration: Recalibration | None = None


def _extras(kind: str, items: Sequence[Scored]) -> dict[str, float]:
    if kind == "score":
        return {"within_1": metrics.within(items, 1), "mean_distance": metrics.mean_distance(items)}
    if kind == "noul":
        precision, recall = metrics.yes_precision_recall(items, 0.0)
        return {"yes_precision": precision, "yes_recall": recall}
    return {}


def question_report(qid: str, kind: str, items: Sequence[Scored], unanswered: int, bins: int) -> QuestionReport:
    reliability = metrics.bin_by_confidence(items, bins)
    return QuestionReport(
        qid=qid,
        kind=kind,
        n=len(items),
        unanswered=unanswered,
        accuracy=metrics.accuracy(items),
        ece=metrics.ece(reliability),
        mce=metrics.mce(reliability),
        reliability=reliability,
        risk_coverage=metrics.risk_coverage(items),
        extras=_extras(kind, items),
        confusion=dict(metrics.confusion(items)) if kind != "composite" else {},
        recalibration=recalibration(qid, kind, items, bins)
        if kind != "composite" and len(items) >= RECAL_MIN
        else None,
    )


def report(records: Sequence[Record], bins: int = 10) -> dict[str, Any]:
    """Accuracy, ECE/MCE, reliability and risk-coverage per question, plus the row-level gate."""
    scoring = score_records(records)
    questions = {
        qid: question_report(qid, scoring.kinds[qid], scoring.items.get(qid, []), scoring.unanswered.get(qid, 0), bins)
        for qid in sorted(scoring.kinds)
    }
    rows = composite(scoring)
    gate = question_report(COMPOSITE, "composite", rows, 0, bins) if rows and len(gated_ids(scoring)) > 1 else None
    return {"model": scoring.model, "questions": questions, "composite": gate}
