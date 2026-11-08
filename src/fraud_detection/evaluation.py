"""Model comparison and hold-out evaluation helpers."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from .config import SEED
from .pipeline import build_pipeline


def compare_models(models: dict, X, y, mode: str = "full", user_profile: bool = True,
                   n_splits: int = 5, seed: int = SEED) -> pd.DataFrame:
    """Stratified k-fold comparison. PR-AUC (average precision) is the primary metric
    because with 5 % positives ROC-AUC saturates long before the model is useful."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        pipe = build_pipeline(model, mode=mode, user_profile=user_profile)
        t0 = time.perf_counter()
        res = cross_validate(pipe, X, y, cv=cv, scoring={"ap": "average_precision", "roc_auc": "roc_auc"}, n_jobs=1)
        rows.append({
            "model": name,
            "PR-AUC": res["test_ap"].mean(), "PR-AUC std": res["test_ap"].std(),
            "ROC-AUC": res["test_roc_auc"].mean(), "ROC-AUC std": res["test_roc_auc"].std(),
            "fit+score time (s)": time.perf_counter() - t0,
        })
    return pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)


def oof_probabilities(model, X, y, mode: str = "full", user_profile: bool = True,
                      n_splits: int = 5, seed: int = SEED) -> np.ndarray:
    """Out-of-fold probabilities: honest scores for every training row (used to pick thresholds)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipe = build_pipeline(model, mode=mode, user_profile=user_profile)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "PR-AUC": float(average_precision_score(y_true, proba)),
        "ROC-AUC": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "n_flagged": int(pred.sum()),
    }


def precision_at_k(y_true, proba, ks=(25, 50, 100, 200)) -> pd.DataFrame:
    """If analysts can only review the k riskiest transactions, what do they find?"""
    order = np.argsort(-np.asarray(proba))
    y = np.asarray(y_true)[order]
    total = y.sum()
    rows = [{"k": k, "precision@k": y[:k].mean(), "recall@k": y[:k].sum() / total} for k in ks if k <= len(y)]
    return pd.DataFrame(rows)
