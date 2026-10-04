"""Plain-text rendering (zero deps) for report and thresholds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .analysis import QuestionReport
from .contract import Recommendation
from .thresholds import MIN_REFUSE, MIN_WARN

ECE_WARN = 0.1


def pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def _header(rep: QuestionReport) -> list[str]:
    lines = [f"== {rep.qid} ({rep.kind}) n={rep.n}" + (f", unanswered={rep.unanswered}" if rep.unanswered else "")]
    lines.append(f"accuracy {pct(rep.accuracy)}   ECE {rep.ece:.3f}   MCE {rep.mce:.3f}")
    lines += [f"{name.replace('_', ' ')} {value:.3f}" for name, value in rep.extras.items()]
    if rep.kind == "score":
        lines.append("(score correct = nearest criterion label equals gold; within-1 and mean distance are ordinal)")
    return lines


def _warnings(rep: QuestionReport) -> list[str]:
    out = []
    if rep.ece > ECE_WARN:
        out.append(f"! ECE {rep.ece:.3f} > {ECE_WARN}: confidence is poorly calibrated; thresholds on it are shaky")
    if rep.n < MIN_WARN:
        out.append(f"! only {rep.n} labeled rows (< {MIN_WARN}): calibration numbers are noisy")
    return out


def _reliability(rep: QuestionReport) -> list[str]:
    label = "mean p(yes)  yes-rate" if rep.kind == "noul" else "mean conf   correct"
    lines = [f"  reliability   bin        count  {label}"]
    for b in rep.reliability:
        if b.count:
            lines.append(f"              {b.lo:.2f}-{b.hi:.2f}  {b.count:5d}  {pct(b.mean_prob)}     {pct(b.rate)}")
    return lines


def _risk_coverage(rep: QuestionReport) -> list[str]:
    lines = ["  risk-coverage  gate>=  coverage  accuracy  escalate"]
    for p in rep.risk_coverage:
        cells = f"{pct(p.coverage)}    {pct(p.accuracy)}    {pct(p.escalation)}"
        lines.append(f"                {p.threshold:.2f}    {cells}")
    return lines


def render_question(rep: QuestionReport) -> str:
    return "\n".join(_header(rep) + _warnings(rep) + _reliability(rep) + _risk_coverage(rep))


def render_report(summary: dict[str, Any]) -> str:
    blocks = [f"jev-eval report - model {summary['model']}"]
    blocks += [render_question(rep) for rep in summary["questions"].values()]
    if summary["composite"] is not None:
        blocks.append("Row-level gate (min of Choice/Score confidences; correct = all gated correct)")
        blocks.append(render_question(summary["composite"]))
    return "\n\n".join(blocks) + "\n"


def _range(rec: Recommendation) -> str:
    if rec.interval is None:
        return ""
    (t_lo, t_hi), (a_lo, a_hi) = rec.interval.threshold, rec.interval.accuracy
    return f", plausibly {t_lo:.2f}-{t_hi:.2f} (accuracy {pct(a_lo).strip()}-{pct(a_hi).strip()})"


def render_recommendation(rec: Recommendation, guard: str) -> str:
    name = f"{rec.qid} ({rec.kind}, n={rec.n})"
    if rec.status == "refused":
        return f"{name}: refused - fewer than {MIN_REFUSE} labeled rows; gather more labels"
    if rec.status == "unreachable" or rec.point is None:
        return f"{name}: no threshold meets the goal on this data"
    p = rec.point
    line = f"{name}: gate at {p.threshold:.2f}{_range(rec)}, auto-handle {pct(p.coverage).strip()} at"
    line += f" {pct(p.accuracy).strip()} accuracy, escalate {pct(p.escalation).strip()}"
    line += " (on the held-out split)" if guard == "holdout" else ""
    return line + (
        f"\n  ! only {rec.n} labeled rows (< {MIN_WARN}): treat as provisional" if rec.status == "warn" else ""
    )


def render_thresholds(recs: Sequence[Recommendation], guard: str) -> str:
    return "\n".join(render_recommendation(rec, guard) for rec in recs) + "\n"
