"""Hyper-parameter search (J2): random and grid search with stratified cross-validation.

Both wrappers sit on scikit-learn's RandomizedSearchCV / GridSearchCV and score with average precision (PR-AUC),
the headline metric for this data (0.3 % fraud). They receive training rows only: the caller passes X_train and
y_train, so validation and test rows cannot reach the search. `refit=False` because the final model is fitted
separately, on the whole training set, after the search.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split

from fraud.data import SEED

N_FOLDS = 3
SCORING = "average_precision"


def stratified_folds(n_splits=N_FOLDS, seed=SEED):
    """Cross-validation folds that each keep the fraud rate. Plain KFold could leave a fold with almost no frauds."""
    return StratifiedKFold(n_splits, shuffle=True, random_state=seed)


def stratified_subsample(y, fraction, seed=SEED):
    """Row positions of a random subsample that keeps the fraud rate (the search runs on a sample to save time)."""
    positions, _ = train_test_split(np.arange(len(y)), train_size=fraction, stratify=y, random_state=seed)
    return positions


def _summary(search):
    """Best parameters plus one row per configuration, best first: parameters, mean and std of CV PR-AUC, fit time."""
    res = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame(list(res["params"]))
    table.insert(0, "rank", res["rank_test_score"].to_numpy())
    table["cv_PR-AUC"] = res["mean_test_score"].to_numpy()
    table["cv_std"] = res["std_test_score"].to_numpy()
    table["mean_fit_seconds"] = res["mean_fit_time"].to_numpy()
    table = table.sort_values(["rank", "cv_PR-AUC"], ascending=[True, False]).reset_index(drop=True)
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    return best, table


def random_search(estimator, space, X_train, y_train, n_iter, seed=SEED, n_splits=N_FOLDS):
    """RandomizedSearchCV: draw `n_iter` configurations from `space`, score each by stratified CV PR-AUC."""
    search = RandomizedSearchCV(estimator, space, n_iter=n_iter, scoring=SCORING, cv=stratified_folds(n_splits, seed),
                                random_state=seed, n_jobs=1, refit=False)
    search.fit(X_train, y_train)
    return _summary(search)


def grid_search(estimator, grid, X_train, y_train, seed=SEED, n_splits=N_FOLDS):
    """GridSearchCV: try every combination in `grid`, score each by stratified CV PR-AUC."""
    search = GridSearchCV(estimator, grid, scoring=SCORING, cv=stratified_folds(n_splits, seed), n_jobs=1, refit=False)
    search.fit(X_train, y_train)
    return _summary(search)
