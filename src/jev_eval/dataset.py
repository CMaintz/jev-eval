"""Labeled JSONL rows and question config (jev-sort's question schema, as JSON)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Questions = dict[str, dict[str, Any]]
_INLINE = re.compile(r"^(\w+)(?:\(([^)]*)\))?$")


@dataclass(frozen=True)
class LabeledRow:
    state: Any
    labels: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


def parse_row(line: str, where: str) -> LabeledRow:
    """One JSONL line: {"state": str | object, "labels": {question_id: gold}, "meta"?: {...}}.

    `meta` is never sent to Jev; it rides along into the cache for per-slice reports.
    """
    doc = json.loads(line)
    if not isinstance(doc, dict) or "state" not in doc or not isinstance(doc.get("labels"), dict):
        raise ValueError(f"{where}: expected an object with 'state' and a 'labels' map")
    meta = doc.get("meta")
    if meta is not None and not isinstance(meta, dict):
        raise ValueError(f"{where}: 'meta' must be an object")
    return LabeledRow(doc["state"], doc["labels"], meta or {})


def load_rows(path: str | Path) -> list[LabeledRow]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [parse_row(line, f"{path}:{n}") for n, line in enumerate(lines, 1) if line.strip()]


def choice(options: dict[str, str], instructions: str) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def noul(instructions: str) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def score(levels: list[str], instructions: str) -> dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def parse_inline(spec: str) -> tuple[str, dict[str, Any]]:
    """`name:choice(a,b,c)` | `name:noul` | `name:score(low,mid,high)` - jev-sort's -q syntax."""
    name, _, body = spec.partition(":")
    match = _INLINE.match(body.strip())
    if not name.strip() or not match:
        raise ValueError(f'bad -q "{spec}": expected name:type(...)')
    kind, args = match.group(1), [a.strip() for a in (match.group(2) or "").split(",") if a.strip()]
    if kind == "noul":
        return name.strip(), noul(name.strip())
    if kind in ("choice", "score") and len(args) < 2:
        raise ValueError(f'{kind} "{name}" needs at least 2 options')
    if kind == "choice":
        return name.strip(), choice({a: a for a in args}, name.strip())
    if kind == "score":
        return name.strip(), score(args, name.strip())
    raise ValueError(f'unknown question type "{kind}" in "{spec}"')


def normalize_question(name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Accept jev-sort's config shape (kind/options/levels) or the wire shape (type/criteria)."""
    kind = raw.get("type") or raw.get("kind")
    instructions = str(raw.get("instructions") or name)
    if kind == "noul":
        return noul(instructions)
    if kind == "choice":
        options = raw.get("criteria") or raw.get("options") or {}
        return choice(options if isinstance(options, dict) else {o: o for o in options}, instructions)
    if kind == "score":
        return score(list(raw.get("criteria") or raw.get("levels") or []), instructions)
    raise ValueError(f'question "{name}": unknown type {kind!r}')


def load_config(path: str | Path) -> tuple[Questions, str | None]:
    """A JSON config: {"model"?: str, "questions": {id: question}}. Returns (questions, model)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = doc.get("questions") if isinstance(doc, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: config needs a non-empty 'questions' map")
    return {name: normalize_question(name, q) for name, q in raw.items()}, doc.get("model")


def gold_problem(question: dict[str, Any], gold: Any) -> str | None:
    """Why a gold label cannot be scored against this question, or None if it is fine."""
    kind = question["type"]
    if kind == "noul" and not isinstance(gold, bool):
        return f"noul gold must be true/false, got {gold!r}"
    if kind in ("choice", "score") and gold not in question["criteria"]:
        return f"{kind} gold {gold!r} is not one of {list(question['criteria'])}"
    return None


def check_labels(rows: list[LabeledRow], questions: Questions) -> None:
    """Fail fast on a gold label that does not fit its question (a typo would skew every metric)."""
    for n, row in enumerate(rows, 1):
        for qid, gold in row.labels.items():
            problem = gold_problem(questions[qid], gold) if qid in questions else None
            if problem:
                raise ValueError(f"row {n}, question {qid!r}: {problem}")
