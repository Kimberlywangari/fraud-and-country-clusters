import numpy as np
import pytest
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud.model import PARAM_NAMES, baseline_params, fit_xgb, signed_log1p


def test_baseline_params_are_the_rank_1_row_of_the_j1_search():
    p = baseline_params()
    assert list(p) == PARAM_NAMES
    # Values from reports/xgboost_tuning_results.csv, rank 1.
    assert p == {"max_depth": 4, "learning_rate": 0.03, "subsample": 0.8,
                 "colsample_bytree": 0.6, "min_child_weight": 5, "scale_pos_weight": 18.3}
    assert isinstance(p["max_depth"], int)  # XGBoost wants an int here, not 4.0


def test_baseline_params_picks_rank_1_not_the_first_row(tmp_path):
    csv = tmp_path / "tuning.csv"
    rows = [{"rank": 2, "max_depth": 3, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 1.0,
             "min_child_weight": 1, "scale_pos_weight": 1.0},
            {"rank": 1, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.6, "colsample_bytree": 0.8,
             "min_child_weight": 5, "scale_pos_weight": 18.3}]
    pd.DataFrame(rows).to_csv(csv, index=False)
    assert baseline_params(csv)["max_depth"] == 6


def test_fit_xgb_learns_a_simple_rule_and_stops_early():
    """Smoke test on a tiny synthetic set: fraud iff x0 > 0.9. The model must beat the naive PR-AUC by a lot."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((3000, 3)), columns=["x0", "x1", "x2"])
    y = (X["x0"] > 0.9).astype(int).to_numpy()
    X_tr, y_tr, X_va, y_va = X.iloc[:2000], y[:2000], X.iloc[2000:], y[2000:]

    model, seconds = fit_xgb(X_tr, y_tr, X_va, y_va, baseline_params(), max_trees=300)

    score = model.predict_proba(X_va)[:, 1]
    assert average_precision_score(y_va, score) > 0.95  # the naive rate is about 0.10
    assert model.best_iteration + 1 < 300  # early stopping ended training before the ceiling
    assert seconds > 0


def test_signed_log1p_keeps_the_sign_and_is_zero_at_zero():
    out = signed_log1p(np.array([-99.0, 0.0, 99.0]))
    assert out[0] == -out[2] and out[1] == 0 and out[2] == np.log1p(99.0)


def test_signed_log1p_is_computed_in_float64_even_for_int8_columns():
    """An int8 column used to come back as float16, which rounds log1p(23) = 3.17805... to 3.178."""
    frame = pd.DataFrame({"hour": np.array([0, 1, 23], dtype="int8"), "amount": [10.0, 20.0, 30.0]})
    out = signed_log1p(frame)
    assert out.dtype == np.float64
    assert out[2, 0] == pytest.approx(np.log1p(23.0), abs=1e-12)
