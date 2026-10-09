"""XGBoost fitting for the fraud model, following notebooks/01_supervised_fraud.ipynb (cells 34-37).

The hyper-parameters are not typed in here: they are read back from the J1 search results
(reports/xgboost_tuning_results.csv, rank 1), so the M2.1 baseline is exactly what was committed.
"""
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
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


# --- Logistic regression (moved from notebooks/01_supervised_fraud.ipynb, cells 30-31) -------------------------------
# Money amounts and balances are heavy-tailed, so inputs get a signed log transform and standard scaling before the
# linear model. The scaler lives inside the pipeline, so it is fitted on whatever rows the pipeline is fitted on
# (a training fold during cross-validation) and never on rows it will later score.

def signed_log1p(a):
    # Cast first: on an int8 column (isTransfer, origEmptied, hour) numpy would compute, and round, in float16.
    a = np.asarray(a, dtype="float64")
    return np.sign(a) * np.log1p(np.abs(a))


def make_logreg(**logreg_params):
    """The J1 logistic-regression pipeline. Pass LogisticRegression arguments to change its settings."""
    params = {"max_iter": 1000, "random_state": SEED, **logreg_params}
    return make_pipeline(FunctionTransformer(signed_log1p), StandardScaler(), LogisticRegression(**params))
