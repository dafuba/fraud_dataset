"""Reason codes: translate a model score into the three things an analyst should look at."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline import feature_names

PRETTY = {
    "amount": "amount", "hour": "hour of day", "log_amount": "log(amount)",
    "is_night": "night-time transaction", "is_micro_amount": "micro-amount (≤ €30)",
    "hour_sin": "hour (cyclical)", "hour_cos": "hour (cyclical)",
    "device_risk_score": "device risk score", "ip_risk_score": "IP risk score",
    "risk_total": "combined device+IP risk", "risk_max": "max(device, IP) risk",
    "amount_x_risk": "amount × risk", "user_txn_count": "user's transaction history size",
    "user_mean_amount": "user's typical amount", "amount_ratio_to_user": "amount vs user's typical",
    "amount_z_user": "amount z-score vs user", "user_night_share": "user's share of night activity",
}


def _pretty(name: str) -> str:
    if name in PRETTY:
        return PRETTY[name]
    for prefix in ("transaction_type_", "merchant_category_", "country_"):
        if name.startswith(prefix):
            return f"{prefix[:-1].replace('_', ' ')} = {name[len(prefix):]}"
    return name


class ReasonCoder:
    """Wraps a fitted pipeline with a SHAP explainer and produces human-readable reason codes."""

    def __init__(self, pipeline):
        import shap  # imported lazily so the package works without shap

        self.pipeline = pipeline
        self.features = feature_names(pipeline)
        self.explainer = shap.TreeExplainer(pipeline.named_steps["model"])

    def transformed(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline[:-1].transform(X)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP contributions toward the *fraud* class, shape (n_rows, n_features)."""
        sv = self.explainer.shap_values(self.transformed(X))
        if isinstance(sv, list):           # older shap: one array per class
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                   # (n, features, classes)
            sv = sv[:, :, -1]
        return sv

    def reason_codes(self, X: pd.DataFrame, top_k: int = 3) -> list[list[dict]]:
        sv = self.shap_values(X)
        Z = self.transformed(X)
        out = []
        for i in range(len(X)):
            order = np.argsort(-np.abs(sv[i]))[:top_k]
            codes = []
            for j in order:
                contrib = float(sv[i, j])
                codes.append({
                    "feature": self.features[j],
                    "label": _pretty(self.features[j]),
                    "value": float(Z[i, j]),
                    "contribution": contrib,
                    "direction": "raises risk" if contrib > 0 else "lowers risk",
                })
            out.append(codes)
        return out

    @staticmethod
    def as_text(codes: list[dict]) -> str:
        parts = []
        for c in codes:
            v = c["value"]
            value = f"{v:,.2f}" if abs(v) >= 100 else f"{v:.3g}"
            parts.append(f"{c['label']} = {value} ({c['direction']}, {c['contribution']:+.2f})")
        return "; ".join(parts)
