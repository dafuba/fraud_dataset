"""Feature engineering that is safe to put inside a scikit-learn pipeline.

Two transformers live here:

* ``BasicFeatures`` – stateless domain features (log amount, night flag, micro-amount
  flag, cyclical hour, and vendor-risk combinations when those scores are present).
* ``UserProfileEncoder`` – a *leakage-safe* behavioural profile per user. It learns
  per-user statistics from the training rows only and, during ``fit_transform``,
  describes every training row by the user's *other* transactions (leave-one-out),
  so a row never sees itself. Unknown users at inference fall back to global stats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import ORACLE

NIGHT_END_HOUR = 6          # 00:00–05:59 counts as night
MICRO_AMOUNT_CEILING = 30.0  # "card testing" range: tiny probes with a stolen card


class BasicFeatures(BaseEstimator, TransformerMixin):
    """Add stateless domain features to a raw transaction frame."""

    def __init__(self, include_oracle: bool = True):
        self.include_oracle = include_oracle

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.oracle_present_ = self.include_oracle and all(c in X.columns for c in ORACLE)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["log_amount"] = np.log1p(X["amount"])
        X["is_night"] = (X["hour"] < NIGHT_END_HOUR).astype(int)
        X["is_micro_amount"] = (X["amount"] <= MICRO_AMOUNT_CEILING).astype(int)
        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
        if self.oracle_present_:
            X["risk_total"] = X[ORACLE].sum(axis=1)
            X["risk_max"] = X[ORACLE].max(axis=1)
            X["amount_x_risk"] = X["amount"] * X["risk_total"]
        return X

    def get_feature_names_out(self, input_features=None):
        extra = ["log_amount", "is_night", "is_micro_amount", "hour_sin", "hour_cos"]
        if self.oracle_present_:
            extra += ["risk_total", "risk_max", "amount_x_risk"]
        return np.concatenate([self.feature_names_in_, np.asarray(extra, dtype=object)])


class UserProfileEncoder(BaseEstimator, TransformerMixin):
    """Per-user behavioural statistics without target leakage.

    Output columns
    --------------
    user_txn_count        number of *other* transactions seen for this user
    user_mean_amount      the user's typical amount (global mean if unknown)
    amount_ratio_to_user  this amount / user's typical amount
    amount_z_user         z-score of this amount against the user's history
    user_night_share      share of the user's transactions that happen at night
    """

    OUTPUT = [
        "user_txn_count",
        "user_mean_amount",
        "amount_ratio_to_user",
        "amount_z_user",
        "user_night_share",
    ]

    def __init__(self, user_col="user_id", amount_col="amount", hour_col="hour"):
        self.user_col = user_col
        self.amount_col = amount_col
        self.hour_col = hour_col

    # ------------------------------------------------------------------ fit
    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "u": np.asarray(X[self.user_col]),
                "a": np.asarray(X[self.amount_col], dtype=float),
                "n": (np.asarray(X[self.hour_col]) < NIGHT_END_HOUR).astype(float),
            }
        )

    def fit(self, X: pd.DataFrame, y=None):
        df = self._frame(X)
        df["a2"] = df["a"] ** 2
        g = df.groupby("u").agg(cnt=("a", "size"), s=("a", "sum"), ss=("a2", "sum"), nn=("n", "sum"))
        self.stats_ = g
        self.global_mean_ = float(df["a"].mean())
        self.global_std_ = float(df["a"].std(ddof=1)) if len(df) > 1 else 1.0
        self.global_night_share_ = float(df["n"].mean())
        return self

    # ------------------------------------------------------------ helpers
    def _features(self, cnt, s, ss, nn, amount) -> pd.DataFrame:
        cnt = cnt.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(cnt > 0, s / np.maximum(cnt, 1), self.global_mean_)
            var = np.where(cnt > 1, (ss - s**2 / np.maximum(cnt, 1)) / np.maximum(cnt - 1, 1), self.global_std_**2)
            std = np.sqrt(np.clip(var, 0, None))
            std = np.where(std > 1e-9, std, self.global_std_)
            night = np.where(cnt > 0, nn / np.maximum(cnt, 1), self.global_night_share_)
        return pd.DataFrame(
            {
                "user_txn_count": cnt,
                "user_mean_amount": mean,
                "amount_ratio_to_user": amount / (mean + 1e-9),
                "amount_z_user": (amount - mean) / (std + 1e-9),
                "user_night_share": night,
            }
        )

    def _lookup(self, users) -> tuple[np.ndarray, ...]:
        st = self.stats_.reindex(users).fillna(0.0)
        return (st["cnt"].to_numpy(), st["s"].to_numpy(), st["ss"].to_numpy(), st["nn"].to_numpy())

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._frame(X)
        cnt, s, ss, nn = self._lookup(df["u"].to_numpy())
        out = self._features(cnt, s, ss, nn, df["a"].to_numpy())
        out.index = X.index
        return out

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Leave-one-out: each training row is profiled by the user's *other* rows."""
        self.fit(X, y)
        df = self._frame(X)
        cnt, s, ss, nn = self._lookup(df["u"].to_numpy())
        a = df["a"].to_numpy()
        out = self._features(cnt - 1, s - a, ss - a**2, nn - df["n"].to_numpy(), a)
        out.index = X.index
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.OUTPUT, dtype=object)
