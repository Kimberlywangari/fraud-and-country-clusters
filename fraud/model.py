"""XGBoost fitting for the fraud model, following notebooks/01_supervised_fraud.ipynb (cells 34-37).

The hyper-parameters are not typed in here: they are read back from the J1 search results
(reports/xgboost_tuning_results.csv, rank 1), so the M2.1 baseline is exactly what was committed.
"""
import time

import pandas as pd
from xgboost import XGBClassifier

from fraud.data import ROOT, SEED

TUNING_CSV = ROOT / "reports" / "xgboost_tuning_results.csv"
# n_estimators is left out on purpose: in the search it is only an upper bound. The final fit uses early stopping.
PARAM_NAMES = ["max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "scale_pos_weight"]
MAX_TREES = 2000
EARLY_STOPPING_ROUNDS = 50


def baseline_params(path=TUNING_CSV):
    """Hyper-parameters of the best configuration (rank 1) from the J1 search, as plain Python numbers."""
    tuning = pd.read_csv(path)
    return tuning.loc[tuning["rank"] == 1, PARAM_NAMES].to_dict("records")[0]


def fit_xgb(X_train, y_train, X_val, y_val, params, seed=SEED, max_trees=MAX_TREES):
    """Fit XGBoost with early stopping on the validation set (stop after 50 rounds without PR-AUC gain).

    Returns (model, seconds). `max_trees` is only a ceiling: early stopping decides the real number of trees.
    """
    model = XGBClassifier(**params, n_estimators=max_trees, tree_method="hist", eval_metric="aucpr",
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS, n_jobs=-1, random_state=seed)
    t = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, time.time() - t
