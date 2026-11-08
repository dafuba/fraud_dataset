"""Population Stability Index (PSI) – the standard check that production traffic still
looks like the data the model was trained on."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi_numeric(reference, current, bins: int = 10, eps: float = 1e-6) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + eps
    c = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(reference, current, eps: float = 1e-6) -> float:
    r = pd.Series(reference).value_counts(normalize=True)
    c = pd.Series(current).value_counts(normalize=True)
    cats = r.index.union(c.index)
    r = r.reindex(cats).fillna(0) + eps
    c = c.reindex(cats).fillna(0) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def psi_status(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate shift"
    return "severe shift"


def psi_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        if f not in reference or f not in current:
            continue
        if pd.api.types.is_numeric_dtype(reference[f]) and reference[f].nunique() > 10:
            v = psi_numeric(reference[f], current[f])
        else:
            v = psi_categorical(reference[f], current[f])
        rows.append({"feature": f, "psi": v, "status": psi_status(v)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
