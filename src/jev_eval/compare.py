"""Compare two runs over the same labeled rows: model-pin drift, or two question wordings.

Records pair on (state hash, question id), so the model and the question text may differ
between runs - that difference is the point. Rows present in only one run are counted as
unpaired, never silently dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .analysis import score_records
from .cache import Record
from .metrics import accuracy, bin_by_confidence, ece
from .score import Scored

PairKey = tuple[str, str]


@dataclass(frozen=True)
class QuestionDiff:
    """Run A vs run B on one question, over the rows both runs answered."""

    qid: str
    kind: str
    paired: int
    unpaired: int
    reworded: bool
    accuracy: tuple[float, float]
    ece: tuple[float, float]
    agreement: float
    both_right: int
    fixes: int  # A wrong, B right
    breaks: int  # A right, B wrong
    both_wrong: int


@dataclass(frozen=True)
class Comparison:
    models: tuple[str, str]
    questions: list[QuestionDiff]


def _by_pair(records: Sequence[Record], tolerance: int) -> dict[PairKey, Scored]:
    """Scored items keyed by (state hash, question id); the first answer wins on duplicates."""
    scoring = score_records(records, tolerance)
    hashes = {(rec["id"], rec["row"]): rec["state_hash"] for rec in records}
    out: dict[PairKey, Scored] = {}
    for qid, items in scoring.items.items():
        for item in items:
            out.setdefault((hashes[(qid, item.row)], qid), item)
    return out


def _counts(pairs: Sequence[tuple[Scored, Scored]]) -> tuple[int, int, int, int]:
    both = sum(1 for a, b in pairs if a.correct and b.correct)
    fixes = sum(1 for a, b in pairs if not a.correct and b.correct)
    breaks = sum(1 for a, b in pairs if a.correct and not b.correct)
    return both, fixes, breaks, len(pairs) - both - fixes - breaks


def _wording(records: Sequence[Record]) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = {}
    for rec in records:
        seen.setdefault(rec["id"], set()).add(rec["question_hash"])
    return seen


def diff_question(
    qid: str, kind: str, a: dict[PairKey, Scored], b: dict[PairKey, Scored], reworded: bool
) -> QuestionDiff:
    keys_a = {k for k in a if k[1] == qid}
    keys_b = {k for k in b if k[1] == qid}
    pairs = [(a[k], b[k]) for k in sorted(keys_a & keys_b)]
    left, right = [p[0] for p in pairs], [p[1] for p in pairs]
    agree = sum(1 for x, y in pairs if x.predicted == y.predicted) / len(pairs) if pairs else 0.0
    return QuestionDiff(
        qid,
        kind,
        len(pairs),
        len(keys_a ^ keys_b),
        reworded,
        (accuracy(left), accuracy(right)),
        (ece(bin_by_confidence(left)), ece(bin_by_confidence(right))),
        agree,
        *_counts(pairs),
    )


def compare(run_a: Sequence[Record], run_b: Sequence[Record], tolerance: int = 0) -> Comparison:
    """Per question: paired accuracy and ECE for each run, agreement, and what B fixes or breaks."""
    a, b = _by_pair(run_a, tolerance), _by_pair(run_b, tolerance)
    kinds = {rec["id"]: rec["type"] for rec in [*run_a, *run_b]}
    words_a, words_b = _wording(run_a), _wording(run_b)
    diffs = [diff_question(q, kinds[q], a, b, words_a.get(q) != words_b.get(q)) for q in sorted(kinds)]
    models = (score_records(run_a).model, score_records(run_b).model)
    return Comparison(models, diffs)
