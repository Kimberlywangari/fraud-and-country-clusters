"""Slow regression test: the refactored package must reproduce the committed J1 baseline.

Run with:  python -m pytest -m slow     (needs data/PS_20174392719_1491204439457_log.csv)
"""
import pandas as pd
import pytest

from fraud.data import DATA_FILE, ROOT, load_paysim, scope_transfer_cashout, split_70_15_15
from fraud.features import make_features
from fraud.metrics import best_f1_threshold, evaluate
from fraud.model import baseline_params, fit_xgb

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not DATA_FILE.exists(), reason="PaySim CSV not downloaded (see README)")]


def test_split_sizes_match_the_j1_notebook():
    d = scope_transfer_cashout(load_paysim())
    y = d["isFraud"].to_numpy()
    train, val, test = split_70_15_15(y)
    # Numbers printed in notebook 01, section 3: test = 415,562 rows with 1,232 frauds.
    assert len(test) == 415_562
    assert y[test].sum() == 1_232


def test_baseline_xgboost_reproduces_model_comparison_csv():
    d = scope_transfer_cashout(load_paysim())
    X, _, ENG = make_features(d)
    y = d["isFraud"].to_numpy()
    train, val, test = split_70_15_15(y)

    model, _ = fit_xgb(X.iloc[train][ENG], y[train], X.iloc[val][ENG], y[val], baseline_params())
    thr = best_f1_threshold(y[val], model.predict_proba(X.iloc[val][ENG])[:, 1])
    row = evaluate("XGBoost", y[test], model.predict_proba(X.iloc[test][ENG])[:, 1], thr, "engineered")

    committed = pd.read_csv(ROOT / "reports" / "model_comparison.csv").set_index("model").loc["XGBoost"]
    assert row["PR-AUC"] == pytest.approx(committed["PR-AUC"], abs=5e-4)
    assert abs(row["FN"] - committed["FN"]) <= 2
    assert abs(row["FP"] - committed["FP"]) <= 2
