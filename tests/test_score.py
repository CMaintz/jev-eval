from __future__ import annotations

from jev_eval.score import choice_prediction, nearest_level, score_answer, with_tolerance
from tests.fakes import QUESTIONS

LEVELS = ["negative", "neutral", "positive"]


def test_choice_argmax_of_probabilities_wins() -> None:
    answer = {"choice": "tech", "confidence": 0.7, "probabilities": {"billing": 0.7, "tech": 0.2, "sales": 0.1}}
    scored = score_answer(0, QUESTIONS["team"], answer, "billing")
    assert scored is not None
    assert scored.correct
    assert scored.predicted == "billing"
    assert scored.confidence == 0.7


def test_choice_falls_back_to_choice_field_and_max_probability() -> None:
    assert choice_prediction({"choice": "tech"}) == ("tech", 0.0)
    assert choice_prediction({"probabilities": {"a": 0.4, "b": 0.6}}) == ("b", 0.6)


def test_noul_confidence_is_distance_from_uncertain() -> None:
    yes = score_answer(1, QUESTIONS["urgent"], {"noul": 0.95}, True)
    no = score_answer(2, QUESTIONS["urgent"], {"noul": 0.2}, True)
    assert yes is not None and no is not None
    assert yes.correct and yes.outcome and yes.prob == 0.95
    assert abs(yes.confidence - 0.9) < 1e-9
    assert not no.correct
    assert abs(no.confidence - 0.6) < 1e-9


def test_score_nearest_level_by_index_and_by_legend() -> None:
    assert nearest_level(1.4, LEVELS) == 1
    assert nearest_level(7.0, LEVELS) == 2
    assert nearest_level(-3.0, LEVELS) == 0
    assert nearest_level(0.1, LEVELS, {"negative": 0.0, "neutral": 0.5, "positive": 1.0}) == 0
    assert nearest_level(2.0, LEVELS, {"negative": "x"}) == 2


def test_score_distance_and_tolerance() -> None:
    scored = score_answer(0, QUESTIONS["sentiment"], {"score": 2.0, "confidence": 0.8}, "neutral")
    assert scored is not None
    assert not scored.correct
    assert scored.distance == 1
    assert scored.predicted == "positive"
    widened = with_tolerance(scored, 1)
    assert widened.correct and widened.outcome
    assert with_tolerance(scored, 0) is scored


def test_unusable_answers_are_none() -> None:
    assert score_answer(0, QUESTIONS["urgent"], None, True) is None
    assert score_answer(0, QUESTIONS["urgent"], {"choice": "x"}, True) is None
    assert score_answer(0, QUESTIONS["sentiment"], {"score": "high"}, "neutral") is None
