"""Feature building for the fraud model.

make_features is moved from notebooks/01_supervised_fraud.ipynb (cell 21) without changing behaviour.
Every feature is computed from the transaction's own row only, so nothing from the label can leak in.
"""
import pandas as pd

# Columns that must never be model inputs (checked by tests/test_features.py).
FORBIDDEN = ["isFraud", "isFlaggedFraud", "step", "nameOrig", "nameDest"]


def make_features(frame):
    """Return (X, raw_cols, all_cols): the raw columns plus four engineered ones.

    The engineered columns encode bookkeeping: a normal transaction moves `amount` out of the origin
    balance and into the destination balance, so both balance errors should be 0. A tree can
    approximate those subtractions; a linear model cannot compute them from raw balances.
    """
    X = pd.DataFrame({
        "isTransfer": (frame["type"] == "TRANSFER").astype("int8"),
        "amount": frame["amount"],
        "oldbalanceOrg": frame["oldbalanceOrg"],
        "newbalanceOrig": frame["newbalanceOrig"],
        "oldbalanceDest": frame["oldbalanceDest"],
        "newbalanceDest": frame["newbalanceDest"],
    })
    raw_cols = list(X.columns)
    X["errorBalanceOrig"] = (frame["oldbalanceOrg"] - frame["amount"] - frame["newbalanceOrig"]).round(2)
    X["errorBalanceDest"] = (frame["oldbalanceDest"] + frame["amount"] - frame["newbalanceDest"]).round(2)
    X["origEmptied"] = (frame["newbalanceOrig"] == 0).astype("int8")
    # Hour of day. `step` itself is not a feature: the simulator makes the fraud rate depend on the day (notebook 01, 2.4).
    X["hour"] = ((frame["step"] - 1) % 24).astype("int8")
    return X, raw_cols, list(X.columns)
