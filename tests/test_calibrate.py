from __future__ import annotations

from jev_eval.calibrate import (
    Step,
    apply,
    calibration_document,
    cross_validated_ece,
    isotonic,
    noul_midpoint,
    recalibration,
)
from jev_eval.score import Scored


def item(p: float, outcome: bool, kind: str = "choice") -> Scored:
    return Scored(0, kind, p, outcome, p, outcome, None, None)


def test_isotonic_pools_violators_into_a_non_decreasing_fit() -> None:
    steps = isotonic([(0.1, False), (0.2, True), (0.3, False), (0.4, True)])
    assert [s.value for s in steps] == [0.0, 0.5, 1.0]
    assert (steps[1].x_lo, steps[1].x_hi) == (0.2, 0.3)
    values = [s.value for s in isotonic([(i / 50, (i * 7) % 3 == 0) for i in range(50)])]
    assert values == sorted(values)


def test_tied_raw_values_share_one_block() -> None:
    steps = isotonic([(0.5, False), (0.5, True), (0.9, True)])
    assert steps[0] == Step(0.5, 0.5, 0.5)
    assert apply(steps, 0.5) == 0.5


def test_tie_groups_are_complete_before_pooling() -> None:
    # Per raw value: 0.2 -> 1 of 2 right, 0.5 -> 3 of 4 right. Already increasing: two steps.
    steps = isotonic([(0.2, False), (0.2, True), (0.5, False), (0.5, True), (0.5, True), (0.5, True)])
    assert [(s.x_lo, s.value) for s in steps] == [(0.2, 0.5), (0.5, 0.75)]


def test_increasing_accuracy_gives_an_increasing_fit() -> None:
    steps = isotonic([(c / 10, i < c) for c in range(1, 10) for i in range(10)])
    assert [round(s.value, 1) for s in steps] == [c / 10 for c in range(1, 10)]


def test_apply_maps_between_and_beyond_blocks() -> None:
    steps = [Step(0.1, 0.2, 0.1), Step(0.6, 0.8, 0.7)]
    assert apply(steps, 0.15) == 0.1
    assert apply(steps, 0.4) == 0.7  # in the gap: the next block up
    assert apply(steps, 0.95) == 0.7  # above the last block: its value


def test_overconfident_model_gains_from_recalibration() -> None:
    # Claims 0.9 but is right half the time; claims 0.6 and is right 1 in 4.
    items = [item(0.9, i % 2 == 0) for i in range(200)] + [item(0.6, i % 4 == 0) for i in range(200)]
    rec = recalibration("q", "choice", items, bins=10, folds=5)
    assert rec.ece_raw > 0.3
    assert rec.ece_cv < 0.05
    assert rec.midpoint is None


def test_cross_validation_never_sees_its_own_fold() -> None:
    # A single extreme row cannot calibrate itself: its fold's fit comes from the other rows.
    items = [item(0.5, True) for _ in range(40)] + [item(0.99, False)]
    assert cross_validated_ece(items, bins=10, folds=5) > 0.0


def test_noul_midpoint_finds_where_calibrated_half_begins() -> None:
    steps = [Step(0.0, 0.3, 0.1), Step(0.42, 0.6, 0.55), Step(0.7, 1.0, 0.9)]
    assert noul_midpoint(steps) == 0.42
    assert noul_midpoint([Step(0.0, 1.0, 0.2)]) is None
    rec = recalibration("u", "noul", [item(0.45, i % 3 != 0, "noul") for i in range(60)], bins=5)
    assert rec.midpoint == 0.45


def test_calibration_document_shape() -> None:
    rec = recalibration("q", "choice", [item(0.8, i % 5 != 0) for i in range(60)])
    doc = calibration_document([rec], "jev-x", "2026-10-04T00:00:00Z")
    assert doc["version"] == 1 and doc["method"] == "isotonic" and doc["model"] == "jev-x"
    entry = doc["questions"]["q"]
    assert entry["input"] == "confidence" and entry["n"] == 60
    assert entry["steps"] == [[0.8, 0.8, 0.8]]
