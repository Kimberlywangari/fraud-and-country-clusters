import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from fraud.search import SCORING, grid_search, random_search, stratified_folds, stratified_subsample


def toy(n=600, rate=0.1, seed=0):
    """A small imbalanced set where `signal` carries the label and `noise` does not."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    X = pd.DataFrame({"signal": y * 1.5 + rng.normal(size=n), "noise": rng.normal(size=n)})
    return X, y


def test_scoring_is_average_precision_not_accuracy():
    assert SCORING == "average_precision"


def test_stratified_folds_keep_the_fraud_count_in_every_fold():
    y = np.array([1] * 30 + [0] * 270)
    for _, held_out in stratified_folds(3).split(np.zeros(len(y)), y):
        assert y[held_out].sum() == 10  # plain KFold could put all 30 frauds in one fold


def test_stratified_subsample_keeps_the_fraud_rate():
    y = np.array([1] * 100 + [0] * 900)
    rows = stratified_subsample(y, 0.3)
    assert len(rows) == 300
    assert y[rows].mean() == pytest.approx(0.1)
    assert len(set(rows)) == len(rows)  # no row drawn twice


def test_grid_search_tries_every_combination_and_ranks_them():
    X, y = toy()
    grid = {"C": [0.01, 1, 100], "class_weight": [None, "balanced"]}
    best, table = grid_search(LogisticRegression(max_iter=500), grid, X, y)
    assert len(table) == 6  # 3 x 2 combinations, all tried
    assert best["C"] in grid["C"] and best["class_weight"] in grid["class_weight"]
    assert table.loc[0, "rank"] == 1
    assert (table["cv_PR-AUC"].diff().dropna() <= 1e-12).all()  # best first


def test_the_reported_score_is_the_stratified_cv_average_precision():
    """The search's best score must equal a hand-run cross_val_score on the same folds and metric."""
    X, y = toy()
    best, table = grid_search(LogisticRegression(max_iter=500), {"C": [0.01, 1, 100]}, X, y)
    manual = cross_val_score(LogisticRegression(max_iter=500, **best), X, y, cv=stratified_folds(), scoring=SCORING)
    assert table.loc[0, "cv_PR-AUC"] == pytest.approx(manual.mean())


SPACE = {"max_depth": [2, 3, 4, 5], "learning_rate": [0.05, 0.1, 0.3], "subsample": [0.6, 0.8, 1.0],
         "min_child_weight": [1, 5, 10]}


def xgb():
    return XGBClassifier(n_estimators=15, tree_method="hist", n_jobs=1, random_state=0)


def test_random_search_draws_the_requested_number_of_configs_from_the_space():
    X, y = toy()
    best, table = random_search(xgb(), SPACE, X, y, n_iter=5)
    assert len(table) == 5
    for name, values in SPACE.items():
        assert set(table[name]) <= set(values)
        assert best[name] in values


def test_random_search_is_repeatable_for_a_fixed_seed_and_differs_across_seeds():
    X, y = toy()
    _, a = random_search(xgb(), SPACE, X, y, n_iter=5, seed=1)
    _, b = random_search(xgb(), SPACE, X, y, n_iter=5, seed=1)
    _, c = random_search(xgb(), SPACE, X, y, n_iter=5, seed=2)
    cols = list(SPACE) + ["cv_PR-AUC"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    drawn = lambda t: set(map(tuple, t[list(SPACE)].to_numpy()))
    assert drawn(a) != drawn(c)  # a different seed samples different configurations: it is a random search
