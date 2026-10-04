from __future__ import annotations

import pytest

from jev_eval import run, slice_report
from jev_eval.slices import length_bucket, slice_key
from tests.fakes import QUESTIONS, FakeProvider, make_rows


def test_length_buckets() -> None:
    assert length_bucket(10) == "<500 chars"
    assert length_bucket(500) == "500-2k chars"
    assert length_bucket(7999) == "2k-8k chars"
    assert length_bucket(9000) == "8k+ chars"
    assert length_bucket(None) == "unknown"


def test_slice_key_reads_meta_and_falls_back_to_unknown() -> None:
    assert slice_key({"meta": {"source": "email"}}, "meta.source") == "email"
    assert slice_key({"meta": {}}, "meta.source") == "unknown"
    assert slice_key({}, "meta.source") == "unknown"  # a cache from before meta existed
    assert slice_key({"state_chars": 3}, "length") == "<500 chars"
    for bad in ("source", "meta.", "chars"):
        with pytest.raises(ValueError, match="bad --slice"):
            slice_key({}, bad)


def test_slice_report_splits_every_question() -> None:
    records = run(make_rows(40), QUESTIONS, FakeProvider())
    rows = slice_report(records, "meta.source")
    assert {(r.qid, r.slice) for r in rows} == {(q, s) for q in QUESTIONS for s in ("chat", "email")}
    assert sum(r.n for r in rows if r.qid == "team") == 40
    by_length = slice_report(records, "length")
    assert {r.slice for r in by_length} == {"<500 chars", "500-2k chars"}
