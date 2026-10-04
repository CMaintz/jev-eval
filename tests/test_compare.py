from __future__ import annotations

from typing import Any

from jev_eval import compare, run
from jev_eval.render_more import render_compare
from tests.fakes import QUESTIONS, FakeProvider, make_rows


class Flipped(FakeProvider):
    """Run B: a model that answers `urgent` the other way on every third row."""

    model = "jev-next"

    def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        out = super().evaluate(state, questions)
        if "urgent" in out["answers"] and state["id"] % 3 == 0:
            out["answers"]["urgent"] = {"noul": 1 - out["answers"]["urgent"]["noul"]}
        return out


def test_identical_runs_agree_fully() -> None:
    records = run(make_rows(60), QUESTIONS, FakeProvider())
    result = compare(records, records)
    assert result.models == ("jev-fake", "jev-fake")
    team = next(d for d in result.questions if d.qid == "team")
    assert team.paired == 60 and team.unpaired == 0 and not team.reworded
    assert team.agreement == 1.0 and team.fixes == 0 and team.breaks == 0
    assert team.accuracy[0] == team.accuracy[1]


def test_drift_counts_fixes_and_breaks_per_paired_row() -> None:
    a = run(make_rows(60), QUESTIONS, FakeProvider())
    b = run(make_rows(60), QUESTIONS, Flipped())
    urgent = next(d for d in compare(a, b).questions if d.qid == "urgent")
    assert urgent.paired == 60
    assert urgent.fixes + urgent.breaks == 20  # every third row flipped
    assert abs(urgent.agreement - 40 / 60) < 1e-9
    assert urgent.both_right + urgent.both_wrong == 40
    assert "B fixes" in render_compare(compare(a, b))


def test_rewording_and_unpaired_rows_are_reported() -> None:
    a = run(make_rows(10), QUESTIONS, FakeProvider())
    reworded = {**QUESTIONS, "team": {**QUESTIONS["team"], "instructions": "Which team owns this?"}}
    b = run(make_rows(8), reworded, FakeProvider())
    team = next(d for d in compare(a, b).questions if d.qid == "team")
    assert team.reworded
    assert team.paired == 8 and team.unpaired == 2
    assert "wording differs" in render_compare(compare(a, b))
