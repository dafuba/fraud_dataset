"""FastAPI scoring service.

    uvicorn app.api:app --reload

* ``POST /score``        – score one transaction, get a probability, a decision and reason codes
* ``POST /score/batch``  – score many transactions at once
* ``GET  /model-card``   – what is deployed, with which thresholds and metrics
* ``GET  /health``

Graceful degradation: if the vendor risk scores are missing from a request (vendor
outage, new market without coverage) the service silently falls back to the model
trained without them, and tells you which one it used.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud_detection.config import MODELS_DIR, ORACLE
from fraud_detection.costs import three_way_decisions
from fraud_detection.explain import ReasonCoder
from fraud_detection.training import load_artifacts

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (MODELS_DIR / "model_card.json").exists():
        raise RuntimeError("No trained artefacts found. Run `python scripts/train.py` first.")
    pipes, card = load_artifacts(MODELS_DIR)
    STATE["pipes"] = pipes
    STATE["card"] = card
    STATE["coders"] = {}
    for mode, pipe in pipes.items():
        try:
            STATE["coders"][mode] = ReasonCoder(pipe)
        except Exception:  # shap unavailable or unsupported model -> no reason codes
            STATE["coders"][mode] = None
    yield
    STATE.clear()


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


class Transaction(BaseModel):
    transaction_id: Optional[int] = None
    user_id: int = Field(..., ge=0)
    amount: float = Field(..., gt=0)
    transaction_type: str
    merchant_category: str
    country: str
    hour: int = Field(..., ge=0, le=23)
    device_risk_score: Optional[float] = Field(None, ge=0, le=1)
    ip_risk_score: Optional[float] = Field(None, ge=0, le=1)


class ReasonCode(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float
    direction: str


class ScoreResponse(BaseModel):
    transaction_id: Optional[int]
    fraud_probability: float
    decision: Literal["approve", "review", "block"]
    model_used: Literal["full", "hardmode"]
    reason_codes: list[ReasonCode]


def _choose_mode(df: pd.DataFrame) -> str:
    has_oracle = all(c in df.columns for c in ORACLE) and df[ORACLE].notna().all(axis=None)
    return "full" if has_oracle and "full" in STATE["pipes"] else "hardmode"


def _score_frame(df: pd.DataFrame, explain: bool = True) -> list[ScoreResponse]:
    mode = _choose_mode(df)
    pipe = STATE["pipes"][mode]
    thresholds = STATE["card"]["modes"][mode]["three_way"]
    proba = pipe.predict_proba(df)[:, 1]
    decisions = three_way_decisions(proba, thresholds["review_threshold"], thresholds["block_threshold"])
    coder = STATE["coders"].get(mode)
    codes = coder.reason_codes(df) if (explain and coder is not None) else [[] for _ in range(len(df))]
    ids = df["transaction_id"] if "transaction_id" in df else [None] * len(df)
    return [
        ScoreResponse(transaction_id=None if pd.isna(tid) else int(tid), fraud_probability=float(p),
                      decision=str(d), model_used=mode, reason_codes=[ReasonCode(**c) for c in rc])
        for tid, p, d, rc in zip(ids, proba, decisions, codes)
    ]


@app.get("/health")
def health():
    return {"status": "ok", "models": list(STATE.get("pipes", {}).keys())}


@app.get("/model-card")
def model_card():
    return STATE["card"]


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction):
    df = pd.DataFrame([tx.model_dump()])
    return _score_frame(df)[0]


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(txs: list[Transaction], explain: bool = False):
    if not txs:
        raise HTTPException(400, "empty batch")
    if len(txs) > 5000:
        raise HTTPException(413, "batch too large (max 5000)")
    df = pd.DataFrame([t.model_dump() for t in txs])
    return _score_frame(df, explain=explain)
