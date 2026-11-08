"""Turn probabilities into decisions using an explicit business cost model.

A fraud model is only useful through the actions it triggers. Every action has a
price, so the right threshold is the one that minimises *expected cost*, not the
one that maximises F1. All amounts are in the dataset's currency unit (shown as €).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    review_cost: float = 5.0        # analyst time for one manual review
    chargeback_fee: float = 15.0    # fixed fee per missed fraud, on top of the lost amount
    recovery_rate: float = 0.0      # share of a missed fraud amount recovered later
    block_friction_cost: float = 25.0  # a legit customer blocked outright: churn / support cost

    def missed_fraud_cost(self, amount) -> np.ndarray:
        return np.asarray(amount, dtype=float) * (1 - self.recovery_rate) + self.chargeback_fee

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------- two-way policy
def policy_cost(y_true, flagged, amount, cm: CostModel) -> dict:
    """Cost of a 'flag for review / approve' policy. Flagged rows are reviewed (cost each);
    reviewed fraud is caught; un-flagged fraud is lost."""
    y = np.asarray(y_true).astype(bool)
    f = np.asarray(flagged).astype(bool)
    amount = np.asarray(amount, dtype=float)
    missed = y & ~f
    caught = y & f
    review = f.sum() * cm.review_cost
    loss = cm.missed_fraud_cost(amount[missed]).sum()
    return {
        "total_cost": float(review + loss),
        "review_cost": float(review),
        "fraud_loss": float(loss),
        "n_flagged": int(f.sum()),
        "n_caught": int(caught.sum()),
        "n_missed": int(missed.sum()),
        "caught_amount": float(amount[caught].sum()),
        "missed_amount": float(amount[missed].sum()),
    }


def cost_curve(y_true, proba, amount, cm: CostModel, thresholds=None) -> pd.DataFrame:
    thresholds = np.linspace(0.01, 0.99, 99) if thresholds is None else np.asarray(thresholds)
    proba = np.asarray(proba)
    rows = []
    for t in thresholds:
        r = policy_cost(y_true, proba >= t, amount, cm)
        r["threshold"] = float(t)
        rows.append(r)
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> tuple[float, pd.Series]:
    row = curve.loc[curve["total_cost"].idxmin()]
    return float(row["threshold"]), row


def baseline_costs(y_true, amount, cm: CostModel) -> dict:
    n = len(np.asarray(y_true))
    return {
        "approve everything": policy_cost(y_true, np.zeros(n, bool), amount, cm)["total_cost"],
        "review everything": policy_cost(y_true, np.ones(n, bool), amount, cm)["total_cost"],
    }


# --------------------------------------------------------------- three-way policy
DECISIONS = ("approve", "review", "block")


def three_way_decisions(proba, review_threshold: float, block_threshold: float) -> np.ndarray:
    p = np.asarray(proba)
    out = np.full(len(p), "approve", dtype=object)
    out[p >= review_threshold] = "review"
    out[p >= block_threshold] = "block"
    return out


def three_way_cost(y_true, decisions, amount, cm: CostModel) -> float:
    y = np.asarray(y_true).astype(bool)
    d = np.asarray(decisions)
    amount = np.asarray(amount, dtype=float)
    review = (d == "review").sum() * cm.review_cost
    blocked_legit = ((d == "block") & ~y).sum() * cm.block_friction_cost
    missed = cm.missed_fraud_cost(amount[(d == "approve") & y]).sum()
    return float(review + blocked_legit + missed)


def optimise_three_way(y_true, proba, amount, cm: CostModel, grid=None) -> tuple[float, float, pd.DataFrame]:
    grid = np.linspace(0.02, 0.98, 49) if grid is None else np.asarray(grid)
    rows = []
    for t_rev in grid:
        for t_blk in grid:
            if t_blk <= t_rev:
                continue
            d = three_way_decisions(proba, t_rev, t_blk)
            rows.append({"review_threshold": t_rev, "block_threshold": t_blk,
                         "total_cost": three_way_cost(y_true, d, amount, cm)})
    df = pd.DataFrame(rows)
    best = df.loc[df["total_cost"].idxmin()]
    return float(best["review_threshold"]), float(best["block_threshold"]), df


def three_way_summary(y_true, decisions, amount) -> pd.DataFrame:
    df = pd.DataFrame({"decision": decisions, "is_fraud": np.asarray(y_true), "amount": np.asarray(amount)})
    out = (df.groupby("decision")
             .agg(transactions=("is_fraud", "size"), frauds=("is_fraud", "sum"),
                  fraud_rate=("is_fraud", "mean"), amount=("amount", "sum"))
             .reindex(list(DECISIONS)).fillna(0))
    out["share_of_volume"] = out["transactions"] / out["transactions"].sum()
    return out
