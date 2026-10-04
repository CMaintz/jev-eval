from __future__ import annotations

from datetime import datetime, timezone

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
    ok = recommend("team", "choice", items, pick, GuardOptions("bootstrap", 50, 0.0, 0))
    assert ok.status == "ok" and ok.interval is not None and ok.usable
    assert recommend("team", "choice", items, picker(1.01, None), NO_GUARD).status == "unreachable"


def test_holdout_guard_reports_on_held_out_rows() -> None:
    items = score_records(records(300)).items["team"]
    rec = recommend("team", "choice", items, picker(0.8, None), GuardOptions("holdout", 0, 0.3, 0))
    assert rec.point is not None and rec.interval is None
    assert rec.point.covered <= 90


def test_thresholds_document_is_the_v1_contract() -> None:
    scoring = score_records(records(300))
    recs = recommend_all(scoring, picker(0.8, None), NO_GUARD)
    assert [r.qid for r in recs] == ["sentiment", "team", "urgent", COMPOSITE]
    when = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
    doc = thresholds_document(recs, scoring.model, gated_ids(scoring), when)
    assert list(doc) == ["version", "model", "generatedAt", "questions", "composite"]
    assert doc["version"] == 1 and doc["model"] == "jev-fake" and doc["generatedAt"] == "2026-10-04T12:00:00Z"
    team = doc["questions"]["team"]
    assert list(team) == ["type", "threshold", "accuracy", "coverage", "n"]
    assert team["type"] == "choice" and team["n"] == 300 and team["accuracy"] >= 0.8
    assert doc["composite"]["questions"] == ["sentiment", "team"]


def test_refused_questions_are_omitted_not_null() -> None:
    scoring = score_records(records(40))
    doc = thresholds_document(recommend_all(scoring, picker(0.8, None), NO_GUARD), "m", [])
    assert doc["questions"] == {}
    assert "composite" not in doc
