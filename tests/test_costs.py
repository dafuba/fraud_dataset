import numpy as np

from fraud_detection.costs import (CostModel, baseline_costs, best_threshold, cost_curve,
                                   policy_cost, three_way_cost, three_way_decisions)

y = np.array([0, 0, 1, 1, 0])
amount = np.array([10.0, 20.0, 100.0, 50.0, 30.0])
cm = CostModel(review_cost=5, chargeback_fee=15, block_friction_cost=25)


def test_policy_cost_extremes():
    approve_all = policy_cost(y, np.zeros(5, bool), amount, cm)
    assert approve_all["total_cost"] == (100 + 15) + (50 + 15)
    review_all = policy_cost(y, np.ones(5, bool), amount, cm)
    assert review_all["total_cost"] == 5 * 5 and review_all["n_missed"] == 0
    assert baseline_costs(y, amount, cm)["approve everything"] == approve_all["total_cost"]


def test_cost_curve_picks_perfect_threshold():
    proba = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    t, row = best_threshold(cost_curve(y, proba, amount, cm))
    assert 0.3 < t <= 0.8 and row["n_missed"] == 0 and row["n_flagged"] == 2


def test_three_way():
    d = three_way_decisions([0.05, 0.5, 0.95], 0.3, 0.9)
    assert d.tolist() == ["approve", "review", "block"]
    cost = three_way_cost([0, 0, 1], d, [10, 10, 10], cm)
    assert cost == 5  # one review, block was fraud (free), approve was legit (free)
