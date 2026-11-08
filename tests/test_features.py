import numpy as np
import pandas as pd

from fraud_detection.features import BasicFeatures, UserProfileEncoder


def test_user_profile_leave_one_out():
    X = pd.DataFrame({"user_id": [1, 1, 1, 2], "amount": [10.0, 20.0, 30.0, 5.0], "hour": [3, 12, 12, 12]})
    enc = UserProfileEncoder()
    train = enc.fit_transform(X)
    # row 0 is described by the user's *other* rows (20, 30)
    assert train.loc[0, "user_txn_count"] == 2
    assert np.isclose(train.loc[0, "user_mean_amount"], 25.0)
    assert np.isclose(train.loc[0, "user_night_share"], 0.0)  # its own night txn is excluded
    # at inference the same rows see the full history
    infer = enc.transform(X)
    assert infer.loc[0, "user_txn_count"] == 3
    assert np.isclose(infer.loc[0, "user_mean_amount"], 20.0)


def test_unknown_user_falls_back_to_global():
    X = pd.DataFrame({"user_id": [1, 1, 2, 2], "amount": [10.0, 30.0, 50.0, 70.0], "hour": [12, 12, 12, 12]})
    enc = UserProfileEncoder().fit(X)
    new = enc.transform(pd.DataFrame({"user_id": [99], "amount": [40.0], "hour": [2]}))
    assert new.loc[0, "user_txn_count"] == 0
    assert np.isclose(new.loc[0, "user_mean_amount"], 40.0)
    assert np.isclose(new.loc[0, "amount_ratio_to_user"], 1.0, atol=1e-6)


def test_basic_features_with_and_without_oracle():
    X = pd.DataFrame({"amount": [5.0, 500.0], "hour": [2, 14],
                      "device_risk_score": [0.1, 0.9], "ip_risk_score": [0.2, 0.8]})
    full = BasicFeatures(include_oracle=True).fit(X).transform(X)
    assert {"risk_total", "is_night", "is_micro_amount"} <= set(full.columns)
    assert full["is_night"].tolist() == [1, 0] and full["is_micro_amount"].tolist() == [1, 0]
    hard = BasicFeatures(include_oracle=False).fit(X).transform(X)
    assert "risk_total" not in hard.columns
