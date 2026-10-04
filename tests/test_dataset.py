from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_eval.dataset import (
    LabeledRow,
    check_labels,
    gold_problem,
    load_config,
    load_rows,
    normalize_question,
    parse_inline,
    parse_row,
)
from tests.fakes import QUESTIONS


def test_load_rows_skips_blank_lines_and_keeps_string_state(tmp_path: Path) -> None:
    data = tmp_path / "d.jsonl"
    data.write_text('{"state": "hi", "labels": {"urgent": true}}\n\n{"state": {"a": 1}, "labels": {}}\n')
    rows = load_rows(data)
    assert rows == [LabeledRow("hi", {"urgent": True}), LabeledRow({"a": 1}, {})]


@pytest.mark.parametrize("line", ['{"labels": {}}', '{"state": 1, "labels": []}', "[1]"])
def test_parse_row_rejects_bad_shapes(line: str) -> None:
    with pytest.raises(ValueError, match="x:3"):
        parse_row(line, "x:3")


def test_parse_inline_all_three_types() -> None:
    assert parse_inline("urgent:noul") == ("urgent", {"type": "noul", "instructions": "urgent"})
    name, team = parse_inline("team:choice(billing, tech)")
    assert name == "team"
    assert team["criteria"] == {"billing": "billing", "tech": "tech"}
    assert parse_inline("s:score(low,mid,high)")[1]["criteria"] == ["low", "mid", "high"]


@pytest.mark.parametrize("spec", ["noname", ":noul", "x:choice(a)", "x:score(a)", "x:bogus(a,b)", "x:choice(a"])
def test_parse_inline_rejects(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_inline(spec)


def test_normalize_accepts_jev_sort_and_wire_shapes() -> None:
    sort_shape = {"kind": "choice", "instructions": "Which team?", "options": {"a": "A", "b": "B"}}
    expected = {"type": "choice", "instructions": "Which team?", "criteria": {"a": "A", "b": "B"}}
    assert normalize_question("t", sort_shape) == expected
    assert normalize_question("t", {"type": "choice", "criteria": ["a", "b"]})["criteria"] == {"a": "a", "b": "b"}
    assert normalize_question("s", {"kind": "score", "levels": ["lo", "hi"]})["criteria"] == ["lo", "hi"]
    assert normalize_question("u", {"kind": "noul"}) == {"type": "noul", "instructions": "u"}
    with pytest.raises(ValueError, match="unknown type"):
        normalize_question("x", {"kind": "rank"})


def test_load_config(tmp_path: Path) -> None:
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"model": "jev-1", "questions": {"u": {"kind": "noul", "instructions": "Urgent?"}}}))
    questions, model = load_config(cfg)
    assert model == "jev-1"
    assert questions["u"]["instructions"] == "Urgent?"
    cfg.write_text(json.dumps({"questions": {}}))
    with pytest.raises(ValueError, match="non-empty"):
        load_config(cfg)


def test_gold_problems() -> None:
    assert gold_problem(QUESTIONS["urgent"], "yes") is not None
    assert gold_problem(QUESTIONS["team"], "legal") is not None
    assert gold_problem(QUESTIONS["sentiment"], "neutral") is None


def test_check_labels_names_the_row_and_ignores_unknown_questions() -> None:
    check_labels([LabeledRow("s", {"other": 1, "urgent": False})], QUESTIONS)
    with pytest.raises(ValueError, match="row 2, question 'team'"):
        check_labels([LabeledRow("s", {}), LabeledRow("s", {"team": "legal"})], QUESTIONS)
