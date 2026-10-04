from __future__ import annotations

from jev_eval import metrics
from jev_eval.score import Scored


def item(conf: float, ok: bool, row: int = 0, predicted: object = None, gold: object = None) -> Scored:
    return Scored(row, "choice", conf, ok, conf, ok, predicted, gold)


def test_perfectly_calibrated_has_zero_ece() -> None:
    # 10 rows at 0.75 confidence, 3 of 4 right in every block of 4 -> 75% empirical.
    items = [item(0.75, i % 4 != 0) for i in range(20)]
    bins = metrics.bin_by_confidence(items, 10)
    assert len(bins) == 10
    assert metrics.ece(bins) == 0.0
    assert metrics.mce(bins) == 0.0


def test_confident_and_wrong_has_ece_one() -> None:
    bins = metrics.bin_by_confidence([item(1.0, False) for _ in range(5)], 10)
    assert bins[-1].count == 5  # 1.0 lands in the top bin, not off the end
    assert metrics.ece(bins) == 1.0
    assert metrics.mce(bins) == 1.0


def test_empty_inputs_are_safe() -> None:
    assert metrics.ece(metrics.bin_by_confidence([], 5)) == 0.0
    assert metrics.mce([]) == 0.0
    assert metrics.accuracy([]) == 0.0
    assert metrics.within([], 1) == 0.0
    assert metrics.mean_distance([]) == 0.0


def test_selective_accuracy_matches_the_spec_sketch() -> None:
    items = [item(0.9, True), item(0.8, True), item(0.6, False), item(0.3, False)]
    assert metrics.selective_accuracy(items, 0.8) == (1.0, 0.5)
    assert metrics.selective_accuracy(items, 0.5) == (2 / 3, 0.75)
    assert metrics.selective_accuracy(items, 0.95) == (0.0, 0.0)


def test_risk_coverage_table_and_curve() -> None:
    items = [item(0.9, True), item(0.9, False), item(0.6, True), item(0.3, False)]
    table = metrics.risk_coverage(items, (0.0, 0.5))
    assert [(p.threshold, p.coverage, p.covered) for p in table] == [(0.0, 1.0, 4), (0.5, 0.75, 3)]
    assert abs(table[1].escalation - 0.25) < 1e-9
    curve = metrics.risk_coverage_curve(items)
    assert [(p.threshold, p.coverage, p.accuracy) for p in curve] == [
        (0.9, 0.5, 0.5),
        (0.6, 0.75, 2 / 3),
        (0.3, 1.0, 0.5),
    ]


def test_yes_precision_recall() -> None:
    items = [item(0.9, True, predicted=True, gold=True), item(0.9, False, predicted=True, gold=False)]
    items += [item(0.9, False, predicted=False, gold=True), item(0.1, True, predicted=True, gold=True)]
    assert metrics.yes_precision_recall(items, 0.5) == (0.5, 0.5)
    assert metrics.yes_precision_recall([], 0.5) == (0.0, 0.0)


def test_score_ordinal_metrics_and_confusion() -> None:
    items = [Scored(0, "score", 0.5, d == 0, 0.5, d == 0, "a", "b", d) for d in (0, 1, 2)]
    assert metrics.within(items, 1) == 2 / 3
    assert metrics.mean_distance(items) == 1.0
    assert metrics.confusion(items)[("b", "a")] == 3
