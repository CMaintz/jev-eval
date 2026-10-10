"""`check`: does a published thresholds.json still hold on newly labeled rows?

Each recorded gate is applied as is (no re-picking) to the new cache, and its accuracy
there is compared with what the file recorded. A gate has drifted when the new accuracy
falls below the recorded one by more than sampling noise explains: two standard errors of
a proportion at the recorded accuracy over the rows the gate covers now. Fewer covered
rows widen that margin, so a small new sample flags only a large drop.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .analysis import COMPOSITE, Scoring, composite
from .metrics import selective_accuracy
from .score import Scored
from .yes import as_yes_items

MIN_COVERED = 30  # below this many covered rows the new accuracy is too noisy to call


@dataclass(frozen=True)
class GateCheck:
    name: str
    threshold: float
    recorded: float  # accuracy (or yes precision) the file recorded
    accuracy: float  # the same gate on the new rows
    coverage: float
    covered: int
    status: str  # holds | drifted | too few rows | no data


def margin(recorded: float, covered: int) -> float:
    return 2 * math.sqrt(recorded * (1 - recorded) / covered) if covered else 1.0


def _status(recorded: float, accuracy: float, covered: int) -> str:
    if covered == 0:
        return "no data"
    if accuracy >= recorded - margin(recorded, covered):
        return "holds"
    return "drifted" if covered >= MIN_COVERED else "too few rows"


def check_gate(name: str, gate: dict[str, Any], items: Sequence[Scored], key: str = "accuracy") -> GateCheck:
    t, recorded = float(gate["threshold"]), float(gate[key])
    acc, cov = selective_accuracy(items, t) if items else (0.0, 0.0)
    covered = round(cov * len(items))
    return GateCheck(name, t, recorded, acc, cov, covered, _status(recorded, acc, covered))


def _yes_checks(doc: dict[str, Any], scoring: Scoring) -> list[GateCheck]:
    return [
        check_gate(f"{qid} yes@{cut['target']}", cut, as_yes_items(scoring.items.get(qid, [])), "precision")
        for qid, cuts in sorted((doc.get("yesAt") or {}).items())
        for cut in cuts
    ]


def check(doc: dict[str, Any], scoring: Scoring) -> list[GateCheck]:
    """Every gate in the file (questions, composite, yes cut-points) applied to `scoring`."""
    checks = [check_gate(qid, gate, scoring.items.get(qid, [])) for qid, gate in sorted(doc["questions"].items())]
    if doc.get("composite"):
        checks.append(check_gate(COMPOSITE, doc["composite"], composite(scoring)))
    return checks + _yes_checks(doc, scoring)


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def warnings(doc: dict[str, Any], scoring: Scoring) -> list[str]:
    """Model changes and rewordings: the gate may still hold, but it was measured on something else."""
    out = []
    if doc.get("model") != scoring.model:
        out.append(f"thresholds were measured on {doc.get('model')}, these rows on {scoring.model}")
    for qid, definition in sorted((doc.get("definitions") or {}).items()):
        if qid in scoring.definitions and not _same(definition, scoring.definitions[qid]):
            out.append(f"{qid} was reworded since the thresholds were measured")
    return out


def load_thresholds(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("questions"), dict):
        raise ValueError(f"{path}: not a thresholds.json (contract version 1)")
    return doc
