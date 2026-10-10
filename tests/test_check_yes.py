from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pytest import CaptureFixture

from jev_eval import cli, run, score_records
from jev_eval.check import GateCheck, check, check_gate, margin, warnings
from jev_eval.contract import GuardOptions
from jev_eval.render_more import render_check
from jev_eval.yes import as_yes_items, yes_cuts, yes_field
from tests.fakes import QUESTIONS, TEAMS, FakeProvider, make_rows

NO_GUARD = GuardOptions("none", 0, 0.0, 0)


class Drifted(FakeProvider):
    """Still confident about `team`, but now wrong about it far more often."""

    model = "jev-drifted"

    def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        out = super().evaluate(state, questions)
        team = out["answers"].get("team")
        if team and int(state["id"]) % 3:
            team["choice"] = TEAMS[(TEAMS.index(team["choice"]) + 1) % 3]
            team.pop("probabilities")
        return out


def cache(tmp_path: Path, name: str, provider: FakeProvider, questions: dict[str, Any] = QUESTIONS) -> Path:
    path = tmp_path / name
    run(make_rows(400), questions, provider, cache_path=path)
    return path


def thresholds(tmp_path: Path, cache_path: Path, *extra: str) -> dict[str, Any]:
    out = tmp_path / "thresholds.json"
    argv = ["thresholds", "--cache", str(cache_path), "--out", str(out), "--no-guard", *extra]
    assert cli.main(argv) == 0
    doc: dict[str, Any] = json.loads(out.read_text())
    return doc


def test_yes_items_read_p_as_confidence_and_a_gold_yes_as_correct() -> None:
    items = score_records(run(make_rows(10), {"urgent": QUESTIONS["urgent"]}, FakeProvider())).items["urgent"]
    flags = as_yes_items(items)
    assert [f.confidence for f in flags] == [i.prob for i in items]
    assert [f.correct for f in flags] == [bool(i.gold) for i in items]


def test_a_stricter_precision_target_flags_at_a_higher_p() -> None:
    scoring = score_records(run(make_rows(400), QUESTIONS, FakeProvider()))
    strict, loose = yes_cuts(scoring, [0.7, 0.9], NO_GUARD)
    assert (strict.target, loose.target) == (0.9, 0.7)
    assert strict.rec.point and loose.rec.point
    assert strict.rec.point.threshold > loose.rec.point.threshold > 0.5
    assert strict.rec.point.accuracy >= 0.9 and 0 < loose.recall <= 1
    entry = yes_field([strict, loose])["urgent"][0]
    assert list(entry) == ["target", "threshold", "precision", "recall", "flagged", "n"]


def test_unreachable_yes_targets_are_left_out_of_the_contract() -> None:
    scoring = score_records(run(make_rows(40), QUESTIONS, FakeProvider()))
    assert yes_field(yes_cuts(scoring, [0.9], NO_GUARD)) == {}  # under 50 rows: refused


def test_thresholds_can_write_only_yes_cut_points(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    doc = thresholds(tmp_path, cache(tmp_path, "a.jsonl", FakeProvider()), "--yes-precision", "0.9,0.7")
    assert doc["questions"] == {} and "composite" not in doc
    assert [c["target"] for c in doc["yesAt"]["urgent"]] == [0.9, 0.7]
    assert "flag yes at P(yes) >=" in capsys.readouterr().out


def test_thresholds_needs_some_goal(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    assert cli.main(["thresholds", "--cache", str(cache(tmp_path, "a.jsonl", FakeProvider()))]) == 2
    assert "--yes-precision" in capsys.readouterr().err


def test_bad_yes_targets_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["thresholds", "--cache", "x", "--yes-precision", "1.5"])


def test_gates_hold_on_rows_like_the_ones_they_were_measured_on(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    a = cache(tmp_path, "a.jsonl", FakeProvider())
    thresholds(tmp_path, a, "--target-accuracy", "0.8", "--yes-precision", "0.9")
    argv = ["check", "--cache", str(a), "--thresholds", str(tmp_path / "thresholds.json")]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "0 of 5 gates drifted" in out and "urgent yes@0.9" in out


def test_a_drifted_gate_fails_the_check(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    thresholds(tmp_path, cache(tmp_path, "a.jsonl", FakeProvider()), "--target-accuracy", "0.8")
    b = cache(tmp_path, "b.jsonl", Drifted())
    assert cli.main(["check", "--cache", str(b), "--thresholds", str(tmp_path / "thresholds.json")]) == 1
    out = capsys.readouterr().out
    assert "! team" in out and "drifted" in out
    assert "measured on jev-fake, these rows on jev-drifted" in out


def test_a_reworded_question_is_named() -> None:
    reworded = {**QUESTIONS, "team": {**QUESTIONS["team"], "instructions": "which team owns this?"}}
    doc = {"model": "jev-fake", "questions": {}, "definitions": QUESTIONS}
    scoring = score_records(run(make_rows(5), reworded, FakeProvider()))
    assert warnings(doc, scoring) == ["team was reworded since the thresholds were measured"]


def test_noise_decides_small_drops_and_empty_gates() -> None:
    assert margin(0.9, 0) == 1.0 and margin(0.9, 100) == pytest.approx(0.06)
    items = score_records(run(make_rows(60), QUESTIONS, FakeProvider())).items["team"]
    assert check_gate("team", {"threshold": 1.1, "accuracy": 0.9}, items).status == "no data"
    assert check_gate("team", {"threshold": 0.95, "accuracy": 1.0}, items).status in ("holds", "too few rows")


def test_check_covers_every_gate_and_render_counts_drift() -> None:
    scoring = score_records(run(make_rows(60), QUESTIONS, FakeProvider()))
    gate = {"threshold": 0.5, "accuracy": 0.9}
    yes = {"urgent": [{**gate, "precision": 0.9, "target": 0.9}]}
    doc = {"questions": {"team": gate}, "composite": gate, "yesAt": yes}
    names = [c.name for c in check(doc, scoring)]
    assert names == ["team", "_row", "urgent yes@0.9"]
    drifted = GateCheck("team", 0.5, 0.9, 0.5, 0.8, 48, "drifted")
    assert "1 of 1 gates drifted; re-run thresholds" in render_check([drifted], [])
