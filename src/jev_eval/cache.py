"""`run`: the only network step. One batched Jev call per row, raw answers to a JSONL cache.

Each cache line is one (row, question) with everything the offline commands need - the
question definition and the gold label ride along, so report/thresholds never re-read the
dataset or the config. Answers are keyed by (model, hash(state), hash(question)): a rerun
resumes, and a repeated state reuses its answer instead of paying for it twice.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import LabeledRow, Questions
from .provider import Provider, RequestRejected

Record = dict[str, Any]
Key = tuple[str, str, str]


def stable_hash(value: Any) -> str:
    """Content hash that is stable across processes (unlike the salted built-in hash())."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def state_chars(state: Any) -> int:
    """Size of the state as Jev sees it (a string verbatim, anything else as JSON)."""
    return len(state) if isinstance(state, str) else len(json.dumps(state, ensure_ascii=False))


def record_key(record: Record) -> Key:
    return (record["model"], record["state_hash"], record["question_hash"])


def load_cache(path: str | Path) -> list[Record]:
    file = Path(path)
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class Asker:
    """Who answers (`provider` under `model`), what is asked, and the earlier records to reuse."""

    provider: Provider
    model: str
    questions: Questions
    known: dict[Key, Record]


@dataclass(frozen=True)
class Reply:
    """One Jev call: the answers, which call it was, its metered input, or why it was refused."""

    answers: dict[str, Any]
    call: str | None = None
    input_tokens: int | None = None
    error: str | None = None


def _record(asker: Asker, index: int, row: LabeledRow, qid: str, source: Record) -> Record:
    question = asker.questions[qid]
    return {
        "model": asker.model,
        "row": index,
        "state_hash": stable_hash(row.state),
        "id": qid,
        "type": question["type"],
        "question_hash": stable_hash(question),
        "question": question,
        "answer": source.get("answer"),
        "gold": row.labels[qid],
        "meta": row.meta,
        "state_chars": state_chars(row.state),
        "call": source.get("call"),
        "input_tokens": source.get("input_tokens"),
        **({"error": source["error"]} if source.get("error") else {}),
    }


def _ask(provider: Provider, row: LabeledRow, asked: Questions, call: str) -> Reply:
    if not asked:
        return Reply({})
    try:
        response = provider.evaluate(row.state, asked)
    except RequestRejected as err:
        return Reply({}, call, error=str(err))
    answers = response.get("answers")
    tokens = (response.get("usage") or {}).get("input_tokens")
    return Reply(answers if isinstance(answers, dict) else {}, call, tokens if isinstance(tokens, int) else None)


def _fresh(reply: Reply, qid: str) -> Record:
    answer = reply.answers.get(qid)
    return {"answer": answer, "call": reply.call, "input_tokens": reply.input_tokens, "error": reply.error}


def run_row(asker: Asker, index: int, row: LabeledRow) -> list[Record]:
    """Score one row: only its labeled questions, and only those not already answered."""
    state_hash = stable_hash(row.state)
    keys = {qid: (asker.model, state_hash, stable_hash(q)) for qid, q in asker.questions.items() if qid in row.labels}
    missing = {qid: asker.questions[qid] for qid, key in keys.items() if key not in asker.known}
    call = stable_hash([asker.model, state_hash, sorted(keys[qid][2] for qid in missing)])
    reply = _ask(asker.provider, row, missing, call)
    records = []
    for qid, key in keys.items():
        records.append(_record(asker, index, row, qid, _fresh(reply, qid) if qid in missing else asker.known[key]))
        if records[-1]["answer"] is not None:
            asker.known[key] = records[-1]
    return records


def _known_records(cache_path: Path) -> dict[Key, Record]:
    """Answered records from a finished cache plus any `.partial` left by an interrupted run."""
    records = load_cache(cache_path) + load_cache(_partial(cache_path))
    return {record_key(r): r for r in records if r.get("answer") is not None}


def _partial(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".partial")


def run(
    rows: Iterable[LabeledRow],
    questions: Questions,
    provider: Provider,
    *,
    model: str | None = None,
    cache_path: str | Path | None = None,
) -> list[Record]:
    """Call Jev once per row (labeled questions only) and return the cache records.

    With `cache_path`, answers already cached are reused (a rerun resumes), progress streams
    to `<cache>.partial`, and the finished run atomically replaces the cache.
    """
    name = model or str(getattr(provider, "model", "unknown"))
    if cache_path is None:
        asker = Asker(provider, name, questions, {})
        return [rec for i, row in enumerate(rows) for rec in run_row(asker, i, row)]
    path = Path(cache_path)
    records = _run_streaming(rows, Asker(provider, name, questions, _known_records(path)), path)
    write_cache(cache_path, records)
    _partial(Path(cache_path)).unlink()
    return records


def _run_streaming(rows: Iterable[LabeledRow], asker: Asker, path: Path) -> list[Record]:
    records: list[Record] = []
    with _partial(path).open("a", encoding="utf-8") as sink:
        for i, row in enumerate(rows):
            fresh = run_row(asker, i, row)
            sink.writelines(_line(r) for r in fresh)
            sink.flush()
            records.extend(fresh)
    return records


def _line(record: Record) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"


def write_cache(path: str | Path, records: list[Record]) -> None:
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_text("".join(_line(r) for r in records), encoding="utf-8")
    tmp.replace(path)
