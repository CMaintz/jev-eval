from __future__ import annotations

import random

import pytest

from jev_eval import thresholds as th
from jev_eval.score import Scored


def item(conf: float, ok: bool) -> Scored:
    return Scored(0, "choice", conf, ok, conf, ok, None, None)


# Gate at 0.9 -> 2/2 right; at 0.7 -> 3/4; at 0.6 -> 4/5; at 0.5 -> 4/6; at 0.2 -> 4/8.
ITEMS = [item(0.95, True), item(0.9, True), item(0.8, True), item(0.7, False)]
ITEMS += [item(0.6, True), item(0.5, False), item(0.3, False), item(0.2, False)]


def test_target_accuracy_takes_the_most_coverage_that_clears_it() -> None:
    point = th.recommend_threshold(ITEMS, target_accuracy=0.75)
    assert point is not None
    assert (point.threshold, point.coverage, point.accuracy) == (0.6, 0.625, 0.8)


def test_max_escalation_takes_the_strictest_gate_within_budget() -> None:
    point = th.recommend_threshold(ITEMS, max_escalation=0.5)
    assert point is not None
    assert point.threshold == 0.7
    assert point.coverage == 0.5
    loose = th.recommend_threshold(ITEMS, max_escalation=0.0)
    assert loose is not None and loose.coverage == 1.0


def test_unreachable_target_is_none() -> None:
    assert th.recommend_threshold([item(0.9, False)], target_accuracy=0.5) is None


def test_exactly_one_goal() -> None:
    with pytest.raises(ValueError):
        th.recommend_threshold(ITEMS)
    with pytest.raises(ValueError):
        th.recommend_threshold(ITEMS, target_accuracy=0.9, max_escalation=0.1)


def test_percentile() -> None:
    assert th.percentile([3.0, 1.0, 2.0], 0.0) == 1.0
    assert th.percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert th.percentile([3.0, 1.0, 2.0], 1.0) == 3.0


def test_bootstrap_range_brackets_a_plausible_gate() -> None:
    items = ITEMS * 20
    interval = th.bootstrap(items, th.picker(0.75, None), 200, random.Random(1))
    assert interval is not None
    lo, hi = interval.threshold
    assert lo <= 0.7 <= hi
    assert interval.resamples == 200
    assert interval.accuracy[0] <= interval.oob_accuracy <= interval.accuracy[1]
    assert 0.0 < interval.oob_coverage <= 1.0


def test_out_of_bag_scoring_exposes_an_overfit_gate() -> None:
    # Confidence carries no signal: 1 in 2 right at every level. In-sample, the pick finds a
    # lucky high-confidence pocket at 60%; out of bag that luck does not repeat.
    rng = random.Random(5)
    items = [item(rng.random(), rng.random() < 0.5) for _ in range(300)]
    pick = th.picker(0.6, None)
    in_sample = pick(items)
    interval = th.bootstrap(items, pick, 300, random.Random(2))
    assert in_sample is not None and in_sample.accuracy >= 0.6
    assert interval is not None and interval.oob_accuracy < 0.5


def test_bootstrap_with_no_reachable_resample_is_none() -> None:
    assert th.bootstrap([item(0.9, False)] * 5, th.picker(0.9, None), 10, random.Random(0)) is None


def test_holdout_scores_the_train_pick_on_unseen_rows() -> None:
    items = ITEMS * 25
    point = th.holdout(items, th.picker(0.75, None), 0.3, random.Random(3))
    assert point is not None
    assert point.covered <= 60
    assert 0.0 < point.coverage <= 1.0
    assert th.holdout([item(0.9, False)] * 4, th.picker(0.9, None), 0.5, random.Random(0)) is None


def test_bootstrap_is_the_default_guard() -> None:
    assert th.DEFAULT_GUARD == "bootstrap"
    assert th.DEFAULT_BOOTSTRAP == 1000
