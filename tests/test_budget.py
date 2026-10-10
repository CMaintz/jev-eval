from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from pytest import CaptureFixture, MonkeyPatch

from jev_eval import LabeledRow, TypeSafeProvider, cli, commands, run
from jev_eval.budget import STATE_BUDGET, estimate_tokens, near_limit, usage
from jev_eval.provider import RequestRejected
from jev_eval.render import render_usage
from tests.fakes import QUESTIONS, FakeProvider, labels, make_rows

HUGE = "x" * (STATE_BUDGET * 4)


class Refuses(FakeProvider):
    """Rejects any state whose text is over the budget, like an over-limit request."""

    def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        if len(state["text"]) >= len(HUGE):
            raise RequestRejected("HTTP 422: state too long")
        return super().evaluate(state, questions)


def huge_rows() -> list[LabeledRow]:
    return [*make_rows(2), LabeledRow({"id": 2, "text": HUGE}, labels(2))]


def test_estimate_counts_the_state_and_only_the_longest_question() -> None:
    short, long = {"q": "a"}, {"q": "a" * 400}
    assert estimate_tokens(400, [short, long]) == (400 + len(json.dumps(long))) // 4
    assert estimate_tokens(0, []) == 0
    assert near_limit(STATE_BUDGET) and not near_limit(STATE_BUDGET // 2)


def test_usage_sums_each_jev_call_once() -> None:
    rows = make_rows(4)
    records = run([*rows, rows[0]], QUESTIONS, FakeProvider())  # the repeat is not paid for again
    use = usage(records)
    assert (use.calls, use.input_tokens, use.unmetered, use.rejected) == (4, 40, 0, 0)
    assert use.cost == pytest.approx(40 / 1_000_000 * 0.042)


def test_a_resumed_run_keeps_the_usage_of_reused_answers(tmp_path: Path) -> None:
    cache = tmp_path / "run.jsonl"
    first = usage(run(make_rows(3), QUESTIONS, FakeProvider(), cache_path=cache))
    again = FakeProvider()
    second = usage(run(make_rows(3), QUESTIONS, again, cache_path=cache))
    assert again.calls == [] and second == first


def test_a_rejected_row_is_recorded_and_the_run_carries_on(tmp_path: Path) -> None:
    cache = tmp_path / "run.jsonl"
    records = run(huge_rows(), QUESTIONS, Refuses(), cache_path=cache)
    refused = [r for r in records if r["row"] == 2]
    assert all(r["answer"] is None and "422" in r["error"] for r in refused)
    assert usage(records).rejected == 1 and usage(records).near_limit == 1
    retry = FakeProvider()
    run(huge_rows(), QUESTIONS, retry, cache_path=cache)
    assert [c["state"]["id"] for c in retry.calls] == [2]  # only the refused row is asked again


def test_an_older_cache_without_usage_is_unmetered() -> None:
    fresh = run(make_rows(2), QUESTIONS, FakeProvider())
    records = [{k: v for k, v in r.items() if k not in ("call", "input_tokens")} for r in fresh]
    assert usage(records).unmetered == 2
    assert "2 calls without usage" in render_usage(usage(records))


def test_render_usage_flags_near_limit_and_rejected_rows() -> None:
    text = render_usage(usage(run(huge_rows(), QUESTIONS, Refuses())))
    assert "1 rows estimated at >= 90%" in text and "1 rows rejected by Jev" in text


def test_provider_raises_request_rejected_on_422(monkeypatch: MonkeyPatch) -> None:
    def refuse(request: Any) -> Any:
        body = io.BytesIO(b'{"error": "state too long"}')
        raise urllib.error.HTTPError(request.full_url, 422, "unprocessable", {}, body)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(RequestRejected, match="HTTP 422: .*state too long"):
        TypeSafeProvider("k").evaluate({}, {})


def test_run_warns_before_spending_on_a_row_near_the_budget(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    monkeypatch.setattr(commands, "provider_from_env", lambda model: Refuses())
    data = tmp_path / "labeled.jsonl"
    rows = [{"state": {"id": i, "text": HUGE if i == 1 else "t"}, "labels": labels(i)} for i in range(3)]
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"questions": QUESTIONS}))
    assert cli.main(["run", "--data", str(data), "--config", str(cfg), "--cache", str(tmp_path / "c.jsonl")]) == 0
    err = capsys.readouterr().err
    assert "warning: row 1 is ~" in err and "usage: 20 input tokens over 2 calls" in err
    assert "1 rows rejected by Jev" in err
