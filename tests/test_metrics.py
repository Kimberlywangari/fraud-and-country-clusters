import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraud.metrics import (WeightedAP, best_cost_threshold, best_f1_threshold, evaluate,
                           expected_cost, missed_amount)


def test_best_f1_threshold_on_a_perfectly_separable_set():
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.6, 0.9])
    # Every score >= 0.6 is fraud and nothing below is: F1 = 1 at threshold 0.6.
    assert best_f1_threshold(y, score) == 0.6


def test_best_f1_threshold_when_the_classes_overlap():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.35, 0.8])
    # By hand: thr 0.8 -> F1 0.667, thr 0.4 -> 0.5, thr 0.35 -> 0.8, thr 0.1 -> 0.667. The best is 0.35.
    assert best_f1_threshold(y, score) == 0.35


def test_evaluate_counts_and_rates():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.6, 0.2, 0.9, 0.4, 0.7])
    row = evaluate("toy", y, score, threshold=0.5)
    # score >= 0.5 flags rows 1, 3, 5: one false alarm (row 1), two caught frauds, one missed fraud (row 4).
    assert (row["TP"], row["FP"], row["FN"], row["TN"]) == (2, 1, 1, 2)
    assert row["precision"] == pytest.approx(2 / 3)
    assert row["recall"] == pytest.approx(2 / 3)
    assert row["F1"] == pytest.approx(2 / 3)
    assert row["accuracy"] == pytest.approx(4 / 6)


def test_evaluate_a_model_that_never_flags_anything():
    """The naive baseline: 0 alerts must give precision 0 and F1 0, not a division error."""
    y = np.array([0, 0, 0, 0, 1])
    row = evaluate("naive", y, np.zeros(5), threshold=0.5)
    assert row["precision"] == 0 and row["recall"] == 0 and row["F1"] == 0
    assert row["accuracy"] == pytest.approx(0.8)  # looks fine, catches nothing


def random_scores(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    # Rounded scores create many ties, which is where a hand-written AP is easiest to get wrong.
    score = np.round(rng.random(n) * 0.5 + y * 0.4, 1)
    return y, score


def test_weighted_ap_with_unit_weights_equals_sklearn():
    y, score = random_scores()
    assert WeightedAP(y, score)(np.ones(len(y))) == pytest.approx(average_precision_score(y, score))


def test_weighted_ap_with_integer_weights_equals_sklearn_sample_weight():
    y, score = random_scores()
    w = np.random.default_rng(1).poisson(1.0, len(y)).astype(float)
    expected = average_precision_score(y, score, sample_weight=w)
    assert WeightedAP(y, score)(w) == pytest.approx(expected)


def test_weighted_ap_of_a_perfect_ranking_is_one():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert WeightedAP(y, score)(np.ones(4)) == pytest.approx(1.0)


# A five-row toy for the cost metric, small enough to check by hand.
# Rows 0 and 2 are frauds (amounts 1000 and 10); rows 1, 3, 4 are legitimate.
COST_Y = np.array([1, 0, 1, 0, 0])
COST_AMOUNT = np.array([1000.0, 50.0, 10.0, 70.0, 20.0])
COST_SCORE = np.array([0.9, 0.8, 0.3, 0.2, 0.1])


def test_missed_amount_adds_up_only_the_frauds_that_were_not_flagged():
    flagged = np.array([True, True, False, False, False])  # catches the 1000 fraud, misses the 10 one
    assert missed_amount(COST_Y, COST_AMOUNT, flagged) == 10.0


def test_expected_cost_is_missed_money_plus_review_cost_per_alert():
    flagged = np.array([True, True, False, False, False])  # 2 alerts, 10 missed
    assert expected_cost(COST_Y, COST_AMOUNT, flagged, review_cost=100) == 10 + 2 * 100


def test_expected_cost_of_flagging_nothing_is_all_the_fraud_money():
    flagged = np.zeros(5, dtype=bool)
    assert expected_cost(COST_Y, COST_AMOUNT, flagged, review_cost=100) == 1010.0


def test_cost_threshold_stops_early_when_alerts_are_expensive():
    # Review cost 100: top-1 costs 10 + 100 = 110, top-2 costs 210, top-3 costs 300, ... so flag only the 0.9 row.
    assert best_cost_threshold(COST_Y, COST_AMOUNT, COST_SCORE, review_cost=100) == 0.9


def test_cost_threshold_goes_lower_when_alerts_are_cheap():
    # Review cost 1: top-1 costs 11, top-2 costs 12, top-3 costs 0 + 3 = 3 (both frauds caught) -> threshold 0.3.
    assert best_cost_threshold(COST_Y, COST_AMOUNT, COST_SCORE, review_cost=1) == 0.3


def test_cost_threshold_flags_nothing_when_every_alert_costs_more_than_the_fraud():
    thr = best_cost_threshold(COST_Y, COST_AMOUNT, COST_SCORE, review_cost=2000)
    assert thr == float("inf")
    assert not (COST_SCORE >= thr).any()


def test_cost_threshold_never_cuts_inside_a_run_of_tied_scores():
    score = np.array([0.9, 0.5, 0.5, 0.5, 0.1])  # the three 0.5s must be flagged together or not at all
    y = np.array([1, 0, 1, 0, 0])
    amount = np.array([100.0, 1.0, 100.0, 1.0, 1.0])
    thr = best_cost_threshold(y, amount, score, review_cost=1)
    assert thr == 0.5  # catching the second fraud (100) is worth 3 alerts at cost 1
    assert (score >= thr).sum() == 4
