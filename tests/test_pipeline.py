import numpy as np
import pytest

from fraud_detection.pipeline import build_pipeline, default_model, feature_names


@pytest.mark.parametrize("mode", ["full", "hardmode"])
def test_pipeline_fits_and_scores(sample, mode):
    X, y = sample
    pipe = build_pipeline(default_model(), mode=mode).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and np.all((proba >= 0) & (proba <= 1))
    names = feature_names(pipe)
    assert len(names) == pipe.named_steps["model"].n_features_in_
    assert ("device_risk_score" in names) == (mode == "full")


def test_hardmode_ignores_oracle_columns(sample):
    X, y = sample
    pipe = build_pipeline(default_model(), mode="hardmode").fit(X, y)
    X_no = X.drop(columns=["device_risk_score", "ip_risk_score"])
    a = pipe.predict_proba(X)[:, 1]
    b = pipe.predict_proba(X_no)[:, 1]
    assert np.allclose(a, b)
