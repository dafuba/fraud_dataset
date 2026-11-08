"""Reproducible training: fit, choose thresholds on out-of-fold scores, evaluate on a
hold-out set, then refit on everything and export deployable artefacts + a model card."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from . import __version__
from .config import DATA_PATH, MODELS_DIR, MODES, SEED, TARGET, TEST_SIZE
from .costs import (CostModel, best_threshold, cost_curve, optimise_three_way,
                    three_way_decisions, three_way_summary)
from .evaluation import classification_metrics, oof_probabilities
from .pipeline import build_pipeline, default_model, feature_names


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split(df: pd.DataFrame, seed: int = SEED):
    X = df.drop(columns=[TARGET, "transaction_id"])
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def train_mode(df: pd.DataFrame, mode: str, cm: CostModel, seed: int = SEED, model_factory=default_model) -> tuple:
    """Return (fitted-on-all-data pipeline, card entry) for one mode."""
    X_train, X_test, y_train, y_test = split(df, seed)

    # 1. honest out-of-fold scores on the training set -> thresholds
    oof = oof_probabilities(model_factory(seed), X_train, y_train, mode=mode, seed=seed)
    curve = cost_curve(y_train, oof, X_train["amount"], cm)
    t_review, _ = best_threshold(curve)
    t3_review, t3_block, _ = optimise_three_way(y_train, oof, X_train["amount"], cm)

    # 2. hold-out evaluation with those thresholds frozen
    pipe = build_pipeline(model_factory(seed), mode=mode).fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, proba, threshold=t_review)
    decisions = three_way_decisions(proba, t3_review, t3_block)
    policy = three_way_summary(y_test, decisions, X_test["amount"])

    # 3. refit on all rows for deployment
    final = build_pipeline(model_factory(seed), mode=mode).fit(df.drop(columns=[TARGET, "transaction_id"]), df[TARGET])

    card = {
        "mode": mode,
        "features": feature_names(final),
        "n_features": len(feature_names(final)),
        "review_threshold": round(t_review, 4),
        "three_way": {"review_threshold": round(t3_review, 4), "block_threshold": round(t3_block, 4)},
        "holdout_metrics": metrics,
        "holdout_policy": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in policy.T.to_dict().items()},
    }
    return final, card


def train_and_export(df: pd.DataFrame | None = None, out_dir: Path = MODELS_DIR, cm: CostModel | None = None,
                     seed: int = SEED, modes=MODES, model_factory=default_model, data_path: Path = DATA_PATH) -> dict:
    df = load_data(data_path) if df is None else df
    cm = cm or CostModel()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    card = {
        "name": "fraud-detection",
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": type(model_factory(seed)).__name__,
        "seed": seed,
        "data": {"path": str(data_path), "sha256_16": _file_hash(data_path) if Path(data_path).exists() else None,
                 "rows": int(len(df)), "fraud_rate": float(df[TARGET].mean())},
        "cost_model": cm.to_dict(),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
        "modes": {},
    }
    for mode in modes:
        pipe, entry = train_mode(df, mode, cm, seed, model_factory)
        path = out_dir / f"fraud_pipeline_{mode}.joblib"
        joblib.dump(pipe, path)
        entry["artifact"] = path.name
        card["modes"][mode] = entry
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    return card


def load_artifacts(models_dir: Path = MODELS_DIR) -> tuple[dict, dict]:
    card = json.loads((Path(models_dir) / "model_card.json").read_text())
    pipes = {m: joblib.load(Path(models_dir) / e["artifact"]) for m, e in card["modes"].items()}
    return pipes, card
