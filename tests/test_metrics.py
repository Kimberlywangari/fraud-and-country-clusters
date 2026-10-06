import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraud.metrics import WeightedAP, best_f1_threshold, evaluate


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
