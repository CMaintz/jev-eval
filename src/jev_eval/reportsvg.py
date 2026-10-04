"""Optional reliability chart as a hand-templated SVG string (no plotting dependency).

One panel per question: a bar per non-empty bin at its empirical rate, the diagonal a
perfectly calibrated model would follow, and the ECE in the title. Bars under the diagonal
mean overconfidence; bars over it, underconfidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from xml.sax.saxutils import escape

from .analysis import QuestionReport

PANEL = 240
PLOT = 160
LEFT, TOP = 50, 36
COLUMNS = 3
STYLE = (
    "<style>text{font:11px sans-serif;fill:#333}.t{font-weight:bold}.ax{stroke:#888}"
    ".d{stroke:#c33;stroke-dasharray:4 3}.b{fill:#4a7bd0;fill-opacity:.8}</style>"
)


def _x(value: float) -> float:
    return LEFT + value * PLOT


def _y(value: float) -> float:
    return TOP + (1 - value) * PLOT


def _bars(rep: QuestionReport) -> list[str]:
    out = []
    for b in rep.reliability:
        if b.count:
            x, w, y = _x(b.lo), (b.hi - b.lo) * PLOT - 1, _y(b.rate)
            title = f"{b.lo:.2f}-{b.hi:.2f}: {b.count} rows, observed {b.rate:.2f}, mean {b.mean_prob:.2f}"
            out.append(f'<rect class="b" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{_y(0) - y:.1f}">')
            out.append(f"<title>{escape(title)}</title></rect>")
    return out


def _frame(rep: QuestionReport) -> list[str]:
    x_label = "P(yes)" if rep.kind == "noul" else "confidence"
    y_label = "yes-rate" if rep.kind == "noul" else "correct"
    title = f"{rep.qid} ({rep.kind}) n={rep.n} ECE {rep.ece:.3f}"
    return [
        f'<text class="t" x="{LEFT}" y="{TOP - 14}">{escape(title)}</text>',
        f'<line class="ax" x1="{_x(0)}" y1="{_y(0)}" x2="{_x(1)}" y2="{_y(0)}"/>',
        f'<line class="ax" x1="{_x(0)}" y1="{_y(0)}" x2="{_x(0)}" y2="{_y(1)}"/>',
        f'<line class="d" x1="{_x(0)}" y1="{_y(0)}" x2="{_x(1)}" y2="{_y(1)}"/>',
        f'<text x="{_x(0.5) - 25}" y="{_y(0) + 26}">{x_label}</text>',
        f'<text x="{LEFT - 44}" y="{_y(0.5)}">{y_label}</text>',
        f'<text x="{_x(0) - 4}" y="{_y(0) + 13}">0</text><text x="{_x(1) - 4}" y="{_y(0) + 13}">1</text>',
    ]


def panel(rep: QuestionReport, index: int) -> str:
    dx, dy = (index % COLUMNS) * PANEL, (index // COLUMNS) * PANEL
    return f'<g transform="translate({dx},{dy})">' + "".join(_bars(rep) + _frame(rep)) + "</g>"


def reliability_svg(reports: Sequence[QuestionReport]) -> str:
    """A standalone SVG document with one reliability panel per question."""
    cols = min(COLUMNS, max(1, len(reports)))
    rows = max(1, -(-len(reports) // COLUMNS))
    width, height = cols * PANEL, rows * PANEL
    head = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    body = "".join(panel(rep, i) for i, rep in enumerate(reports))
    return f'{head}{STYLE}<rect width="100%" height="100%" fill="#fff"/>{body}</svg>\n'
