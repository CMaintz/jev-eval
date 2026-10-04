from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jev_eval import composite, report, run, score_records, thresholds_document
from jev_eval.analysis import COMPOSITE, gated_ids
from jev_eval.contract import GuardOptions, recommend, recommend_all
from jev_eval.thresholds import picker
from tests.fakes import QUESTIONS, FakeProvider, make_rows

NO_GUARD = GuardOptions("none", 0, 0.0, 0)


def records(n: int) -> list[dict[str, object]]:
    return run(make_rows(n), QUESTIONS, FakeProvider())


def test_report_covers_every_type_and_the_row_gate() -> None:
    summary = report(records(300), bins=5)
    assert summary["model"] == "jev-fake"
    team, urgent, sentiment = (summary["questions"][q] for q in ("team", "urgent", "sentiment"))
    assert team.n == 300 and len(team.reliability) == 5
    assert 0.5 < team.accuracy < 0.9
    assert team.ece < 0.15  # the fake is calibrated by construction
    assert set(urgent.extras) == {"yes_precision", "yes_recall"}
    assert set(sentiment.extras) == {"within_1", "mean_distance"}
    assert sentiment.extras["within_1"] >= sentiment.accuracy
    assert summary["composite"] is not None and summary["composite"].kind == "composite"


def test_unanswered_records_are_counted_not_scored() -> None:
    recs = records(3)
    recs[0] = {**recs[0], "answer": None}
    scoring = score_records(recs)
    assert scoring.unanswered == {"team": 1}
    assert len(scoring.items["team"]) == 2


def test_composite_is_min_confidence_and_all_correct() -> None:
    scoring = score_records(records(30))
    rows = composite(scoring)
    assert gated_ids(scoring) == ["sentiment", "team"]
    assert len(rows) == 30
    team = {i.row: i for i in scoring.items["team"]}
    sent = {i.row: i for i in scoring.items["sentiment"]}
    for r in rows:
        assert r.confidence == min(team[r.row].confidence, sent[r.row].confidence)
        assert r.correct == (team[r.row].correct and sent[r.row].correct)


def test_min_n_guard_refuses_warns_and_passes() -> None:
    items = score_records(records(300)).items["team"]
    pick = picker(0.8, None)
    assert recommend("team", "choice", items[:49], pick, NO_GUARD).status == "refused"
    assert recommend("team", "choice", items[:120], pick, NO_GUARD).status == "warn"
    assert recommend("team", "choice", items, picker(1.01, None), NO_GUARD).status == "unreachable"


def test_bootstrap_reports_out_of_bag_numbers_not_in_sample() -> None:
    items = score_records(records(300)).items["team"]
    rec = recommend("team", "choice", items, picker(0.8, None), GuardOptions("bootstrap", 60, 0.0, 0))
    assert rec.status == "ok" and rec.usable
    assert rec.point is not None and rec.interval is not None and rec.in_sample is not None
    assert rec.point.threshold == rec.in_sample.threshold
    assert rec.point.accuracy == rec.interval.oob_accuracy
    assert rec.point.accuracy <= rec.in_sample.accuracy + 0.05  # OOB is not the optimistic number


def test_unstable_when_the_guard_cannot_score_any_pick() -> None:
    items = score_records(records(300)).items["team"]
    rec = recommend("team", "choice", items, picker(0.99, None), GuardOptions("bootstrap", 0, 0.0, 0))
    assert rec.status in ("unstable", "unreachable")
    assert not rec.usable


def test_holdout_guard_reports_on_held_out_rows() -> None:
    items = score_records(records(300)).items["team"]
    rec = recommend("team", "choice", items, picker(0.8, None), GuardOptions("holdout", 0, 0.3, 0))
    assert rec.point is not None and rec.interval is None
    assert rec.point.covered <= 90


def test_thresholds_document_is_the_v1_contract() -> None:
    scoring = score_records(records(300), tolerance=1)
    recs = recommend_all(scoring, picker(0.8, None), NO_GUARD)
    assert [r.qid for r in recs] == ["sentiment", "team", "urgent", COMPOSITE]
    when = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
    doc = thresholds_document(recs, scoring.model, gated_ids(scoring), NO_GUARD, 1, when)
    assert list(doc) == ["version", "model", "generatedAt", "guard", "tolerance", "questions", "composite"]
    assert doc["version"] == 1 and doc["model"] == "jev-fake" and doc["generatedAt"] == "2026-10-04T12:00:00Z"
    assert doc["guard"] == {"method": "none"} and doc["tolerance"] == 1
    team = doc["questions"]["team"]
    assert list(team) == ["type", "threshold", "accuracy", "coverage", "n"]
    assert team["type"] == "choice" and team["n"] == 300 and team["accuracy"] >= 0.8
    assert doc["composite"]["questions"] == ["sentiment", "team"]


@pytest.mark.parametrize(
    ("opts", "expected"),
    [
        (GuardOptions("bootstrap", 1000, 0.0, 7), {"method": "bootstrap", "resamples": 1000, "seed": 7}),
        (GuardOptions("holdout", 0, 0.3, 0), {"method": "holdout", "fraction": 0.3, "seed": 0}),
    ],
)
def test_guard_field_says_what_accuracy_means(opts: GuardOptions, expected: dict[str, object]) -> None:
    assert thresholds_document([], "m", [], opts)["guard"] == expected


def test_refused_questions_are_omitted_not_null() -> None:
    scoring = score_records(records(40))
    doc = thresholds_document(recommend_all(scoring, picker(0.8, None), NO_GUARD), "m", [], NO_GUARD)
    assert doc["questions"] == {}
    assert "composite" not in doc


def test_report_measures_recalibration_but_not_for_small_or_composite() -> None:
    summary = report(records(300))
    team = summary["questions"]["team"].recalibration
    assert team is not None and team.ece_cv >= 0 and team.steps
    assert summary["composite"].recalibration is None
    assert report(records(30))["questions"]["team"].recalibration is None
