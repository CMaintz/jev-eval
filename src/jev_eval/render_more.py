"""Plain-text rendering for compare, per-slice reports and the recalibration summary."""

from __future__ import annotations

from collections.abc import Sequence

from .calibrate import Recalibration
from .compare import Comparison, QuestionDiff
from .render import pct
from .slices import SliceRow
from .thresholds import MIN_WARN


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
