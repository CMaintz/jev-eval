"""Plain-text rendering for compare, slices, recalibration, yes cut-points and check."""

from __future__ import annotations

from collections.abc import Sequence

from .calibrate import Recalibration
from .check import GateCheck
from .compare import Comparison, QuestionDiff
from .render import BASIS, pct, render_recommendation
from .slices import SliceRow
from .thresholds import MIN_WARN
from .yes import YesCut


def _diff_lines(d: QuestionDiff) -> list[str]:
    (acc_a, acc_b), (ece_a, ece_b) = d.accuracy, d.ece
    head = f"== {d.qid} ({d.kind}) paired n={d.paired}" + (f", unpaired={d.unpaired}" if d.unpaired else "")
    lines = [head + (" - question wording differs between runs" if d.reworded else "")]
    lines.append(f"accuracy {pct(acc_a)} -> {pct(acc_b)}   ECE {ece_a:.3f} -> {ece_b:.3f}")
    lines.append(f"agreement {pct(d.agreement).strip()}: B fixes {d.fixes}, breaks {d.breaks}")
    lines.append(f"  both right {d.both_right}, both wrong {d.both_wrong}")
    if d.paired < MIN_WARN:
        lines.append(f"! only {d.paired} paired rows (< {MIN_WARN}): small differences here are noise")
    return lines


def render_compare(result: Comparison) -> str:
    model_a, model_b = result.models
    blocks = [f"jev-eval compare - A: {model_a}  B: {model_b}"]
    blocks += ["\n".join(_diff_lines(d)) for d in result.questions]
    return "\n\n".join(blocks) + "\n"


def render_slices(rows: Sequence[SliceRow], spec: str) -> str:
    lines = [f"per-slice report by {spec}", "  question        slice               n   accuracy   ECE"]
    for r in rows:
        flag = "  (small)" if r.n < MIN_WARN else ""
        lines.append(f"  {r.qid:<15} {r.slice:<18} {r.n:4d}   {pct(r.accuracy)}   {r.ece:.3f}{flag}")
    return "\n".join(lines) + "\n"


def render_calibration(recs: Sequence[Recalibration]) -> str:
    lines = ["recalibration (isotonic, 5-fold cross-validated, measured only)"]
    for r in recs:
        line = f"  {r.qid} ({r.kind}, n={r.n}): ECE {r.ece_raw:.3f} -> {r.ece_cv:.3f}, {len(r.steps)} steps"
        lines.append(line + (f", calibrated 0.5 at raw {r.midpoint:.2f}" if r.midpoint is not None else ""))
    return "\n".join(lines) + "\n"


def _yes_line(cut: YesCut, guard: str) -> str:
    point = cut.rec.point
    if not cut.rec.usable or point is None:
        return f"{cut.qid} yes@{cut.target}: " + render_recommendation(cut.rec, guard).split(": ", 1)[1]
    head = f"{cut.qid} (noul, n={cut.rec.n}): flag yes at P(yes) >= {point.threshold:.2f} for {pct(cut.target).strip()}"
    body = f"  precision {pct(point.accuracy).strip()} {BASIS[guard]}, flags {pct(point.coverage).strip()} of rows"
    return f"{head} precision\n{body}, catches {pct(cut.recall).strip()} of the real yeses"


def render_yes(cuts: Sequence[YesCut], guard: str) -> str:
    return "".join(_yes_line(cut, guard) + "\n" for cut in cuts)


def _check_line(c: GateCheck) -> str:
    mark = {"holds": " ", "drifted": "!", "too few rows": "?", "no data": "?"}[c.status]
    return (
        f"{mark} {c.name:<22} gate {c.threshold:.2f}  recorded {pct(c.recorded)}  now {pct(c.accuracy)}"
        f"  coverage {pct(c.coverage)} ({c.covered} rows)  {c.status}"
    )


def render_check(checks: Sequence[GateCheck], notes: Sequence[str]) -> str:
    lines = ["jev-eval check - recorded gates applied to the new rows"]
    lines += [f"! {note}" for note in notes]
    lines += [_check_line(c) for c in checks]
    drifted = sum(c.status == "drifted" for c in checks)
    lines.append(f"{drifted} of {len(checks)} gates drifted" + ("; re-run thresholds" if drifted else ""))
    return "\n".join(lines) + "\n"
