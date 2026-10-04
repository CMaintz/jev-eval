"""Per-slice reports: does Jev do worse on one source, or on long states?

`--slice length` buckets rows by state size; `--slice meta.<field>` groups by a field of the
row's optional `meta` object (carried through the cache, never sent to Jev). Records from a
cache without that data land in the slice "unknown".
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .analysis import score_records
from .cache import Record
from .metrics import accuracy, bin_by_confidence, ece

LENGTH_BUCKETS = ((500, "<500 chars"), (2000, "500-2k chars"), (8000, "2k-8k chars"))
LONGEST = "8k+ chars"
UNKNOWN = "unknown"


def length_bucket(chars: object) -> str:
    if not isinstance(chars, int):
        return UNKNOWN
    return next((label for limit, label in LENGTH_BUCKETS if chars < limit), LONGEST)


def slice_key(record: Record, spec: str) -> str:
    """The slice a record falls in, for a spec of `length` or `meta.<field>`."""
    if spec == "length":
        return length_bucket(record.get("state_chars"))
    if spec.startswith("meta.") and len(spec) > len("meta."):
        meta = record.get("meta")
        value = meta.get(spec[len("meta.") :]) if isinstance(meta, dict) else None
        return UNKNOWN if value is None else str(value)
    raise ValueError(f'bad --slice "{spec}": use "length" or "meta.<field>"')


@dataclass(frozen=True)
class SliceRow:
    qid: str
    kind: str
    slice: str
    n: int
    accuracy: float
    ece: float


def slice_report(records: Sequence[Record], spec: str, bins: int = 10) -> list[SliceRow]:
    """Accuracy and ECE per (question, slice), questions then slices in sorted order."""
    groups: dict[str, list[Record]] = defaultdict(list)
    for rec in records:
        groups[slice_key(rec, spec)].append(rec)
    rows = []
    for name, group in groups.items():
        scoring = score_records(group)
        for qid, items in scoring.items.items():
            reliability = bin_by_confidence(items, bins)
            rows.append(SliceRow(qid, scoring.kinds[qid], name, len(items), accuracy(items), ece(reliability)))
    return sorted(rows, key=lambda r: (r.qid, r.slice))
