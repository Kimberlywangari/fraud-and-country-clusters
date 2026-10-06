"""Loading, scoping and splitting the PaySim data.

Moved from notebooks/01_supervised_fraud.ipynb (cells 6, 13 and 23) without changing behaviour,
so the numbers in reports/model_comparison.csv can be reproduced from here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "PS_20174392719_1491204439457_log.csv"

# The identifier columns (nameOrig, nameDest) are never loaded: they are almost unique per row.
COLS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]
# Compact types: the file is 470 MB and about 1.5 GB in memory with pandas defaults.
DTYPES = {"step": "int16", "type": "category", "isFraud": "int8", "isFlaggedFraud": "int8"}


def load_paysim(path=DATA_FILE):
    return pd.read_csv(path, usecols=COLS, dtype=DTYPES)


def scope_transfer_cashout(df):
    """Keep TRANSFER and CASH_OUT only: in this data no other transaction type is ever fraudulent."""
    d = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
    d["type"] = d["type"].cat.remove_unused_categories()
    return d


def split_70_15_15(y, seed=SEED):
    """Row positions for train / validation / test, stratified on the label so each part keeps the fraud rate.

    Two chained splits: 70 % train, then the remaining 30 % cut in half. The split is random, not
    chronological, because PaySim's fraud rate per day is a simulation artefact (notebook 01, section 2.4).
    """
    idx = np.arange(len(y))
    train_idx, rest_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.50, stratify=y[rest_idx], random_state=seed)
    return train_idx, val_idx, test_idx
