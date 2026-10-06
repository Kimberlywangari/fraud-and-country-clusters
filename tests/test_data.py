import numpy as np
import pandas as pd

from fraud.data import scope_transfer_cashout, split_70_15_15


def make_frame():
    """Six rows covering every PaySim type, with one fraud among the TRANSFER / CASH_OUT rows."""
    types = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "TRANSFER"]
    return pd.DataFrame({
        "type": pd.Categorical(types),
        "amount": [10.0, 500.0, 200.0, 30.0, 5.0, 900.0],
        "isFraud": [0, 1, 0, 0, 0, 0],
    })


def test_scope_keeps_only_transfer_and_cash_out():
    d = scope_transfer_cashout(make_frame())
    assert sorted(d["type"].unique()) == ["CASH_OUT", "TRANSFER"]
    assert len(d) == 3


def test_scope_drops_unused_categories_and_resets_index():
    d = scope_transfer_cashout(make_frame())
    # Unused categories would leak into later groupbys as empty groups.
    assert set(d["type"].cat.categories) == {"CASH_OUT", "TRANSFER"}
    assert list(d.index) == [0, 1, 2]


def test_scope_does_not_lose_any_fraud():
    df = make_frame()
    assert scope_transfer_cashout(df)["isFraud"].sum() == df["isFraud"].sum()


def imbalanced_labels(n=2000, fraud_rate=0.05):
    y = np.zeros(n, dtype=int)
    y[: int(n * fraud_rate)] = 1
    return np.random.default_rng(0).permutation(y)


def test_split_sizes_are_70_15_15():
    y = imbalanced_labels()
    train, val, test = split_70_15_15(y)
    assert len(train) == 1400
    assert len(val) == 300
    assert len(test) == 300


def test_split_parts_are_disjoint_and_cover_every_row():
    y = imbalanced_labels()
    train, val, test = split_70_15_15(y)
    together = np.concatenate([train, val, test])
    assert len(set(together)) == len(together)  # no row appears twice
    assert sorted(together) == list(range(len(y)))  # no row is lost


def test_split_keeps_the_fraud_rate_in_each_part():
    y = imbalanced_labels()
    for part in split_70_15_15(y):
        assert abs(y[part].mean() - y.mean()) < 0.005


def test_split_is_deterministic_for_a_fixed_seed():
    y = imbalanced_labels()
    first = split_70_15_15(y, seed=1)
    second = split_70_15_15(y, seed=1)
    other = split_70_15_15(y, seed=2)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert not np.array_equal(first[0], other[0])
