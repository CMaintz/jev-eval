"""Jev's request budget and what a run spent.

Jev takes 32k tokens for the state plus the longest question (64k for the whole request,
docs.typesafe.ai/models). The docs do not say what happens past that, so jev-eval warns
before sending a row that is close, and a row Jev rejects is recorded with its `error`
instead of aborting the run. Token counts here are estimates at ~4 characters per token;
the spend uses the real `usage.input_tokens` Jev returns.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

STATE_BUDGET = 32_000  # tokens: state + the longest question
NEAR = 0.9  # warn at 90% of the budget
CHARS_PER_TOKEN = 4
PRICE_PER_MILLION = 0.042  # TypeSafe list price per million input tokens, output free


def estimate_tokens(state_size: int, questions: Sequence[Any]) -> int:
    """Rough tokens for the part Jev budgets: the state plus the longest question."""
    longest = max((len(json.dumps(q, ensure_ascii=False)) for q in questions), default=0)
    return (state_size + longest) // CHARS_PER_TOKEN


def near_limit(tokens: int) -> bool:
    return tokens >= STATE_BUDGET * NEAR


@dataclass(frozen=True)
class Usage:
    """What a cache cost and how close its rows came to the budget."""

    input_tokens: int
    calls: int
    unmetered: int  # calls with no usage recorded (an older cache)
    near_limit: int  # rows estimated within 10% of the budget
    rejected: int  # rows Jev refused

    @property
    def cost(self) -> float:
        return self.input_tokens / 1_000_000 * PRICE_PER_MILLION


def _calls(records: Sequence[dict[str, Any]]) -> dict[Any, int | None]:
    """Input tokens per distinct Jev call (a call answers several questions)."""
    calls: dict[Any, int | None] = {}
    for rec in records:
        if rec.get("answer") is not None:
            calls[rec.get("call") or (rec["model"], rec["state_hash"], rec["row"])] = rec.get("input_tokens")
    return calls


def _rows(records: Sequence[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
    rows: dict[Any, list[dict[str, Any]]] = {}
    for rec in records:
        rows.setdefault((rec["model"], rec["row"]), []).append(rec)
    return rows


def usage(records: Sequence[dict[str, Any]]) -> Usage:
    calls = _calls(records)
    rows = _rows(records).values()
    near = sum(
        1 for row in rows if near_limit(estimate_tokens(row[0].get("state_chars") or 0, [r["question"] for r in row]))
    )
    return Usage(
        input_tokens=sum(t for t in calls.values() if t),
        calls=len(calls),
        unmetered=sum(1 for t in calls.values() if t is None),
        near_limit=near,
        rejected=sum(1 for row in rows if any(r.get("error") for r in row)),
    )
