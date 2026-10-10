"""Plain-text rendering (zero deps) for report and thresholds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .analysis import QuestionReport
from .budget import NEAR, STATE_BUDGET, Usage
from .contract import Recommendation
from .metrics import Point
from .thresholds import MIN_REFUSE, MIN_WARN

ECE_WARN = 0.1
MIDPOINT_NOTE = 0.02  # report a shifted Noul 0.5 only when it moved more than this


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


def _recalibration(rep: QuestionReport) -> list[str]:
    recal = rep.recalibration
    if recal is None:
        return []
    gain = recal.ece_raw - recal.ece_cv
    verdict = f"ECE {recal.ece_raw:.3f} -> {recal.ece_cv:.3f}" if gain > 0 else "no gain over raw"
    lines = [f"recalibration (isotonic, 5-fold, measured only): {verdict}"]
    if recal.midpoint is not None and abs(recal.midpoint - 0.5) > MIDPOINT_NOTE:
        lines.append(
            f"calibrated P(yes) reaches 0.5 at raw {recal.midpoint:.2f}: Jev's 0.5 is really {recal.midpoint:.2f}"
        )
    return lines


def render_question(rep: QuestionReport) -> str:
    sections = _header(rep) + _recalibration(rep) + _warnings(rep) + _reliability(rep) + _risk_coverage(rep)
    return "\n".join(sections)


def render_usage(use: Usage) -> str:
    """Spend plus the rows that came close to, or past, Jev's context budget."""
    lines = [f"usage: {use.input_tokens} input tokens over {use.calls} calls (est. ${use.cost:.4f})"]
    if use.unmetered:
        lines[0] += f", {use.unmetered} calls without usage (older cache)"
    budget = f"Jev's {STATE_BUDGET // 1000}k state + longest question budget"
    if use.near_limit:
        lines.append(f"! {use.near_limit} rows estimated at >= {NEAR:.0%} of {budget}: check they were not cut short")
    if use.rejected:
        lines.append(f"! {use.rejected} rows rejected by Jev (see `error` in the cache); they count as unanswered")
    return "\n".join(lines)


def render_report(summary: dict[str, Any]) -> str:
    blocks = [f"jev-eval report - model {summary['model']}", render_usage(summary["usage"])]
    blocks += [render_question(rep) for rep in summary["questions"].values()]
    if summary["composite"] is not None:
        blocks.append("Row-level gate (min of Choice/Score confidences; correct = all gated correct)")
        blocks.append(render_question(summary["composite"]))
    return "\n\n".join(blocks) + "\n"


BASIS = {"bootstrap": "out-of-bag", "holdout": "on the held-out split", "none": "on the same rows (optimistic)"}


def _buys(rec: Recommendation, guard: str) -> str:
    p = rec.point or Point(0.0, 0.0, 0.0, 0)
    line = f"  auto-handle {pct(p.coverage).strip()} (~{p.covered} rows) at {pct(p.accuracy).strip()} accuracy"
    line += f" {BASIS[guard]}, escalate {pct(p.escalation).strip()}"
    if rec.interval is not None:
        (t_lo, t_hi), (a_lo, a_hi) = rec.interval.threshold, rec.interval.accuracy
        line += f"\n  95% range: gate {t_lo:.2f}-{t_hi:.2f}, accuracy {pct(a_lo).strip()}-{pct(a_hi).strip()}"
    if guard != "none" and rec.in_sample is not None:
        line += f"\n  scored on all rows instead: {pct(rec.in_sample.accuracy).strip()} (optimistic)"
    return line


def render_recommendation(rec: Recommendation, guard: str) -> str:
    name = f"{rec.qid} ({rec.kind}, n={rec.n})"
    if rec.status == "refused":
        return f"{name}: refused - fewer than {MIN_REFUSE} labeled rows; gather more labels"
    if rec.status == "unreachable":
        return f"{name}: no threshold meets the goal on this data"
    if rec.status == "unstable" or rec.point is None:
        return f"{name}: no stable gate - the guard could not score a pick on unseen rows; gather more labels"
    line = f"{name}: gate at {rec.point.threshold:.2f}\n{_buys(rec, guard)}"
    return line + (
        f"\n  ! only {rec.n} labeled rows (< {MIN_WARN}): treat as provisional" if rec.status == "warn" else ""
    )


def render_thresholds(recs: Sequence[Recommendation], guard: str) -> str:
    return "\n".join(render_recommendation(rec, guard) for rec in recs) + "\n"
