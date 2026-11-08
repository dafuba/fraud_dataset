"""Pipelines and candidate models.

Two *modes* are supported:

* ``"full"``      – every column, including the vendor risk scores.
* ``"hardmode"``  – the vendor risk scores are removed. This is the realistic
                    problem: the scores are an oracle for the label in this dataset.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, ORACLE, SEED
from .features import BasicFeatures, UserProfileEncoder

ENGINEERED = ["log_amount", "is_night", "is_micro_amount", "hour_sin", "hour_cos"]
ORACLE_ENGINEERED = ["risk_total", "risk_max", "amount_x_risk"]


def numeric_features(mode: str = "full") -> list[str]:
    feats = ["amount", "hour", *ENGINEERED]
    if mode == "full":
        feats += [*ORACLE, *ORACLE_ENGINEERED]
    return feats


def build_preprocessor(mode: str = "full", user_profile: bool = True) -> ColumnTransformer:
    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", numeric_features(mode)),
    ]
    if user_profile:
        transformers.append(("user", UserProfileEncoder(), ["user_id", "amount", "hour"]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(model, mode: str = "full", user_profile: bool = True) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", BasicFeatures(include_oracle=(mode == "full"))),
            ("prep", build_preprocessor(mode, user_profile)),
            ("model", model),
        ]
    )


def feature_names(pipeline: Pipeline):
    """Names of the columns the final estimator actually sees."""
    return list(pipeline[:-1].get_feature_names_out())


def candidate_models(seed: int = SEED) -> dict:
    """A spread of model families; boosting libraries are optional."""
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, random_state=seed
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, eval_metric="aucpr", random_state=seed, n_jobs=-1,
        )
    except ImportError:  # pragma: no cover
        pass
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
        )
    except ImportError:  # pragma: no cover
        pass
    return models


def default_model(seed: int = SEED):
    """The model shipped to production: fast, small, SHAP-friendly, strong on tabular data."""
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, random_state=seed)
