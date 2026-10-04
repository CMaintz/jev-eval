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
from pathlib import Path
from typing import Any

from .dataset import LabeledRow, Questions
from .provider import Provider

Record = dict[str, Any]
Key = tuple[str, str, str]


def stable_hash(value: Any) -> str:
    """Content hash that is stable across processes (unlike the salted built-in hash())."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def record_key(record: Record) -> Key:
    return (record["model"], record["state_hash"], record["question_hash"])


def load_cache(path: str | Path) -> list[Record]:
    file = Path(path)
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _record(model: str, index: int, row: LabeledRow, qid: str, question: dict[str, Any], answer: Any) -> Record:
    return {
        "model": model,
        "row": index,
        "state_hash": stable_hash(row.state),
        "id": qid,
        "type": question["type"],
        "question_hash": stable_hash(question),
        "question": question,
        "answer": answer,
        "gold": row.labels[qid],
    }


def _ask(provider: Provider, row: LabeledRow, asked: Questions) -> dict[str, Any]:
    if not asked:
        return {}
    answers = provider.evaluate(row.state, asked).get("answers", {})
    return answers if isinstance(answers, dict) else {}


def run_row(
    provider: Provider, model: str, index: int, row: LabeledRow, questions: Questions, known: dict[Key, Any]
) -> list[Record]:
    """Score one row: only its labeled questions, and only those not already answered."""
    labeled = {qid: q for qid, q in questions.items() if qid in row.labels}
    state_hash = stable_hash(row.state)
    missing = {qid: q for qid, q in labeled.items() if (model, state_hash, stable_hash(q)) not in known}
    fresh = _ask(provider, row, missing)
    records = []
    for qid, question in labeled.items():
        key = (model, state_hash, stable_hash(question))
        records.append(_record(model, index, row, qid, question, fresh.get(qid) if qid in missing else known[key]))
        if records[-1]["answer"] is not None:
            known[key] = records[-1]["answer"]
    return records


def _known_answers(cache_path: Path) -> dict[Key, Any]:
    """Answers from a finished cache plus any `.partial` left by an interrupted run."""
    records = load_cache(cache_path) + load_cache(_partial(cache_path))
    return {record_key(r): r["answer"] for r in records if r.get("answer") is not None}


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
        known: dict[Key, Any] = {}
        return [rec for i, row in enumerate(rows) for rec in run_row(provider, name, i, row, questions, known)]
    records = _run_streaming(rows, questions, provider, name, Path(cache_path))
    write_cache(cache_path, records)
    _partial(Path(cache_path)).unlink()
    return records


def _run_streaming(
    rows: Iterable[LabeledRow], questions: Questions, provider: Provider, model: str, path: Path
) -> list[Record]:
    known = _known_answers(path)
    records: list[Record] = []
    with _partial(path).open("a", encoding="utf-8") as sink:
        for i, row in enumerate(rows):
            fresh = run_row(provider, model, i, row, questions, known)
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
