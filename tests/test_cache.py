from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jev_eval import LabeledRow, load_cache, run
from jev_eval.cache import stable_hash
from tests.fakes import QUESTIONS, FakeProvider, make_rows


def test_one_batched_call_per_row_with_only_labeled_questions() -> None:
    rows = [LabeledRow({"id": 0}, {"team": "billing"}), LabeledRow({"id": 1}, {}), *make_rows(3)[1:]]
    provider = FakeProvider()
    records = run(rows, QUESTIONS, provider)
    assert len(provider.calls) == 3  # the unlabeled row costs nothing
    assert list(provider.calls[0]["questions"]) == ["team"]
    assert len(provider.calls[1]["questions"]) == 3
    assert {r["model"] for r in records} == {"jev-fake"}
    assert len(records) == 1 + 3 + 3


def test_records_carry_everything_the_offline_commands_need() -> None:
    record = run(make_rows(1), {"urgent": QUESTIONS["urgent"]}, FakeProvider(), model="jev-x")[0]
    assert record["model"] == "jev-x"
    assert record["gold"] is True
    assert record["question"] == QUESTIONS["urgent"]
    assert record["state_hash"] == stable_hash({"id": 0, "text": "ticket 0"})
    assert "noul" in record["answer"]


def test_rerun_resumes_from_the_cache_at_zero_cost(tmp_path: Path) -> None:
    cache = tmp_path / "run.jsonl"
    first = run(make_rows(4), QUESTIONS, FakeProvider(), cache_path=cache)
    again = FakeProvider()
    second = run(make_rows(4), QUESTIONS, again, cache_path=cache)
    assert again.calls == []
    assert second == first == load_cache(cache)
    assert not (tmp_path / "run.jsonl.partial").exists()


def test_interrupted_run_keeps_its_progress(tmp_path: Path) -> None:
    cache = tmp_path / "run.jsonl"

    class Dies(FakeProvider):
        def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
            if state["id"] == 2:
                raise RuntimeError("network down")
            return super().evaluate(state, questions)

    with pytest.raises(RuntimeError):
        run(make_rows(4), QUESTIONS, Dies(), cache_path=cache)
    resumed = FakeProvider()
    run(make_rows(4), QUESTIONS, resumed, cache_path=cache)
    assert [c["state"]["id"] for c in resumed.calls] == [2, 3]


def test_repeated_state_is_paid_for_once() -> None:
    rows = [LabeledRow({"id": 5}, {"urgent": True}), LabeledRow({"id": 5}, {"urgent": False})]
    provider = FakeProvider()
    records = run(rows, QUESTIONS, provider)
    assert len(provider.calls) == 1
    assert [r["row"] for r in records] == [0, 1]
    assert records[0]["answer"] == records[1]["answer"]


def test_missing_answers_are_recorded_and_retried() -> None:
    class Partial(FakeProvider):
        def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
            super().evaluate(state, questions)
            return {"answers": []}

    provider = Partial()
    rows = [LabeledRow({"id": 1}, {"urgent": True}), LabeledRow({"id": 1}, {"urgent": True})]
    records = run(rows, QUESTIONS, provider)
    assert [r["answer"] for r in records] == [None, None]
    assert len(provider.calls) == 2


def test_records_carry_meta_and_state_size_for_slices() -> None:
    rows = [LabeledRow("plain text", {"urgent": True}, {"source": "web"}), LabeledRow({"id": 1}, {"urgent": False})]

    class Echo(FakeProvider):
        def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
            return {"answers": {"urgent": {"noul": 0.9}}}

    first, second = run(rows, QUESTIONS, Echo())
    assert first["meta"] == {"source": "web"} and first["state_chars"] == len("plain text")
    assert second["meta"] == {} and second["state_chars"] == len('{"id": 1}')
