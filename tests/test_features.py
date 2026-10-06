import pandas as pd
import pytest

from fraud.features import FORBIDDEN, make_features


def make_frame(**overrides):
    """One consistent TRANSFER: 1000 leaves the origin (5000 -> 4000) and arrives at the destination (200 -> 1200)."""
    row = {"step": [1], "type": ["TRANSFER"], "amount": [1000.0],
           "oldbalanceOrg": [5000.0], "newbalanceOrig": [4000.0],
           "oldbalanceDest": [200.0], "newbalanceDest": [1200.0]}
    row.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(row)


def test_consistent_transaction_has_zero_balance_errors():
    X, _, _ = make_features(make_frame())
    assert X.loc[0, "errorBalanceOrig"] == 0
    assert X.loc[0, "errorBalanceDest"] == 0


def test_origin_error_is_old_minus_amount_minus_new():
    # Origin only dropped by 700 although 1000 was sent: 5000 - 1000 - 4300 = -300.
    X, _, _ = make_features(make_frame(newbalanceOrig=4300.0))
    assert X.loc[0, "errorBalanceOrig"] == -300


def test_destination_error_is_old_plus_amount_minus_new():
    # Destination balance did not change although 1000 arrived: 200 + 1000 - 200 = 1000.
    X, _, _ = make_features(make_frame(newbalanceDest=200.0))
    assert X.loc[0, "errorBalanceDest"] == 1000


def test_origin_emptied_flag():
    emptied, _, _ = make_features(make_frame(newbalanceOrig=0.0))
    not_emptied, _, _ = make_features(make_frame())
    assert emptied.loc[0, "origEmptied"] == 1
    assert not_emptied.loc[0, "origEmptied"] == 0


def test_transfer_flag_distinguishes_transfer_from_cash_out():
    transfer, _, _ = make_features(make_frame(type="TRANSFER"))
    cash_out, _, _ = make_features(make_frame(type="CASH_OUT"))
    assert transfer.loc[0, "isTransfer"] == 1
    assert cash_out.loc[0, "isTransfer"] == 0


@pytest.mark.parametrize("step, hour", [(1, 0), (24, 23), (25, 0), (743, 22)])  # (743 - 1) % 24 = 22
def test_hour_wraps_every_24_steps(step, hour):
    X, _, _ = make_features(make_frame(step=step))
    assert X.loc[0, "hour"] == hour


def test_engineered_set_is_raw_plus_four_columns():
    _, raw, engineered = make_features(make_frame())
    assert engineered[: len(raw)] == raw
    assert engineered[len(raw):] == ["errorBalanceOrig", "errorBalanceDest", "origEmptied", "hour"]


def test_feature_columns_never_contain_labels_or_identifiers():
    """Leakage guard: label, the simulator's own rule output, `step` and identifiers must not be inputs."""
    frame = make_frame(isFraud=1, isFlaggedFraud=1, nameOrig="C123", nameDest="C456")
    X, _, engineered = make_features(frame)
    assert not set(FORBIDDEN) & set(engineered)
    assert not set(FORBIDDEN) & set(X.columns)


def test_features_do_not_change_when_the_label_changes():
    """Same transaction, opposite label: the features must be identical."""
    a, _, _ = make_features(make_frame(isFraud=0))
    b, _, _ = make_features(make_frame(isFraud=1))
    pd.testing.assert_frame_equal(a, b)
