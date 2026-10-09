import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from fraud.features import make_features
from fraud.leakage import (LeakageError, check_rows_within, features_ignore_label, permuted_label_scores, rows_shared,
                           univariate_pr_auc)
from fraud.model import make_logreg, signed_log1p


# --- features_ignore_label ------------------------------------------------------------------------------------------

def transactions(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"step": rng.integers(1, 100, n), "type": rng.choice(["TRANSFER", "CASH_OUT"], n),
                         "amount": rng.random(n) * 1000, "oldbalanceOrg": rng.random(n) * 5000,
                         "newbalanceOrig": rng.random(n) * 5000, "oldbalanceDest": rng.random(n) * 5000,
                         "newbalanceDest": rng.random(n) * 5000, "isFraud": rng.integers(0, 2, n)})


def test_the_real_feature_builder_ignores_the_label():
    assert features_ignore_label(make_features, transactions(), "isFraud")


def test_a_feature_builder_that_uses_the_label_is_caught():
    leaky = lambda f: pd.DataFrame({"amount": f["amount"], "looks_innocent": f["isFraud"] * 2.0})
    assert not features_ignore_label(leaky, transactions(), "isFraud")


# --- rows_shared ----------------------------------------------------------------------------------------------------

def test_planted_duplicates_are_found_and_only_those():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    test = pd.DataFrame({"a": [2.0, 9.0, 3.0], "b": [20.0, 90.0, 30.0]})
    assert rows_shared(train, test).tolist() == [True, False, True]


def test_a_row_must_match_in_every_column_to_count_as_shared():
    train = pd.DataFrame({"a": [1.0], "b": [10.0]})
    test = pd.DataFrame({"a": [1.0], "b": [11.0]})  # same a, different b: not a twin
    assert not rows_shared(train, test).any()


def test_no_rows_are_shared_between_distinct_data():
    rng = np.random.default_rng(0)
    assert not rows_shared(pd.DataFrame(rng.random((100, 3))), pd.DataFrame(rng.random((100, 3)))).any()


# --- check_rows_within ----------------------------------------------------------------------------------------------

def test_rows_inside_the_allowed_set_pass():
    check_rows_within([3, 5, 7], allowed=np.arange(10))
    check_rows_within([], allowed=np.arange(10))


def test_rows_outside_the_allowed_set_raise_and_are_reported():
    with pytest.raises(LeakageError, match="2 rows are outside"):
        check_rows_within([3, 12, 15], allowed=np.arange(10), name="search rows")


# --- permuted_label_scores ------------------------------------------------------------------------------------------

def signal_data(n=3000, rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    return pd.DataFrame({"signal": y * 2.0 + rng.normal(size=n), "noise": rng.normal(size=n)}), y


def logistic_fit_predict(X_tr, y_tr, X_te):
    return LogisticRegression(max_iter=500).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]


def test_with_shuffled_labels_an_honest_pipeline_falls_to_the_fraud_rate():
    X, y = signal_data()
    X_tr, y_tr, X_te, y_te = X.iloc[:2000], y[:2000], X.iloc[2000:], y[2000:]

    # With the real labels the pipeline learns the signal ...
    real = average_precision_score(y_te, logistic_fit_predict(X_tr, y_tr, X_te))
    assert real > 0.7
    # ... and with the training labels shuffled there is nothing to learn, so it falls to about the fraud rate.
    shuffled = permuted_label_scores(logistic_fit_predict, X_tr, y_tr, X_te, y_te, seeds=[0, 1, 2])
    assert max(shuffled) < y_te.mean() + 0.05


def test_a_pipeline_that_peeks_at_test_labels_stays_high_even_with_shuffled_training_labels():
    X, y = signal_data()
    X_tr, y_tr, X_te, y_te = X.iloc[:2000], y[:2000], X.iloc[2000:], y[2000:]
    peeking = lambda a, b, c: y_te + np.random.default_rng(0).normal(0, 0.1, len(y_te))  # uses test labels directly
    assert min(permuted_label_scores(peeking, X_tr, y_tr, X_te, y_te, seeds=[0, 1])) > 0.9


# --- univariate_pr_auc ----------------------------------------------------------------------------------------------

def test_a_feature_that_gives_the_label_away_is_ranked_first_whatever_its_sign():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    X = pd.DataFrame({"noise": rng.normal(size=2000), "inverse_label": -y + rng.normal(0, 0.01, 2000)})
    ranked = univariate_pr_auc(X, y)
    assert ranked.index[0] == "inverse_label"
    assert ranked["inverse_label"] > 0.99
    assert ranked["noise"] < y.mean() + 0.05


# --- preprocessing leakage: the scaler only sees the rows the pipeline is fitted on --------------------------------

def test_the_scaler_is_fitted_on_the_training_rows_only():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame({"amount": rng.random(500) * 100, "balance": rng.random(500) * 1000})
    X_test = X_train * 50  # a very different distribution: if the scaler had seen it, the mean would move
    y = (rng.random(500) < 0.2).astype(int)

    pipeline = make_logreg().fit(X_train, y)
    scaler = pipeline.named_steps["standardscaler"]

    np.testing.assert_allclose(scaler.mean_, signed_log1p(X_train).mean(axis=0))
    pooled = signed_log1p(pd.concat([X_train, X_test])).mean(axis=0)
    assert not np.allclose(scaler.mean_, pooled)
