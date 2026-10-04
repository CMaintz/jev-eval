from __future__ import annotations

import xml.etree.ElementTree as ET

from jev_eval import report, run
from jev_eval.reportsvg import reliability_svg
from tests.fakes import QUESTIONS, FakeProvider, make_rows

SVG = "{http://www.w3.org/2000/svg}"


def test_one_bar_per_non_empty_bin_and_a_panel_per_question() -> None:
    summary = report(run(make_rows(90), QUESTIONS, FakeProvider()), bins=5)
    panels = list(summary["questions"].values())
    root = ET.fromstring(reliability_svg(panels))
    groups = root.findall(f"{SVG}g")
    assert len(groups) == 3
    for group, rep in zip(groups, panels, strict=True):
        bars = group.findall(f"{SVG}rect")
        assert len(bars) == sum(1 for b in rep.reliability if b.count)
        title = group.find(f"{SVG}text")
        assert title is not None and rep.qid in (title.text or "")


def test_titles_are_escaped_and_layout_wraps() -> None:
    summary = report(run(make_rows(30), QUESTIONS, FakeProvider()))
    reps = list(summary["questions"].values()) * 2
    text = reliability_svg(reps)
    root = ET.fromstring(text)
    assert root.get("width") == "720" and root.get("height") == "480"
    assert reliability_svg([]).startswith("<svg")
