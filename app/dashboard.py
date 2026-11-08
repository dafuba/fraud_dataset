"""Streamlit analyst console.

    streamlit run app/dashboard.py

Tabs: live triage of a batch (upload a CSV or use the bundled sample), a what-if
threshold explorer driven by the business cost model, and a single-transaction
inspector with SHAP reason codes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from fraud_detection.config import DATA_PATH, MODELS_DIR, ORACLE, TARGET
from fraud_detection.costs import (CostModel, baseline_costs, best_threshold, cost_curve,
                                   three_way_decisions, three_way_summary)
from fraud_detection.explain import ReasonCoder
from fraud_detection.training import load_artifacts

st.set_page_config(page_title="Fraud Analyst Console", layout="wide")


@st.cache_resource
def artifacts():
    pipes, card = load_artifacts(MODELS_DIR)
    coders = {}
    for m, p in pipes.items():
        try:
            coders[m] = ReasonCoder(p)
        except Exception:
            coders[m] = None
    return pipes, card, coders


pipes, card, coders = artifacts()

st.title("Fraud Analyst Console")
st.caption(f"Model: {card['model']} · trained {card['trained_at']} · modes: {', '.join(card['modes'])}")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Transactions CSV", type="csv")
    df = pd.read_csv(upload) if upload else pd.read_csv(DATA_PATH).sample(2000, random_state=7)
    has_oracle = all(c in df.columns for c in ORACLE)
    mode = st.radio("Model", ["full", "hardmode"], index=0 if has_oracle else 1,
                    help="'hardmode' ignores the vendor risk scores.")
    if mode == "full" and not has_oracle:
        st.warning("Vendor risk scores missing in this file – switching to hardmode.")
        mode = "hardmode"
    st.header("Cost assumptions (€)")
    cm = CostModel(
        review_cost=st.number_input("Manual review", 0.0, 100.0, 5.0),
        chargeback_fee=st.number_input("Chargeback fee (missed fraud)", 0.0, 200.0, 15.0),
        recovery_rate=st.slider("Recovery rate on missed fraud", 0.0, 1.0, 0.0),
        block_friction_cost=st.number_input("Blocking a legit customer", 0.0, 500.0, 25.0),
    )

pipe = pipes[mode]
features = df.drop(columns=[c for c in [TARGET, "transaction_id"] if c in df.columns])
proba = pipe.predict_proba(features)[:, 1]
df = df.assign(fraud_probability=proba)
t3 = card["modes"][mode]["three_way"]

tab_triage, tab_whatif, tab_inspect = st.tabs(["Triage queue", "What-if thresholds", "Inspect a transaction"])

with tab_triage:
    c1, c2 = st.columns(2)
    t_review = c1.slider("Review threshold", 0.0, 1.0, float(t3["review_threshold"]), 0.01)
    t_block = c2.slider("Block threshold", 0.0, 1.0, float(t3["block_threshold"]), 0.01)
    decisions = three_way_decisions(proba, t_review, max(t_block, t_review + 1e-9))
    df["decision"] = decisions
    counts = pd.Series(decisions).value_counts().reindex(["approve", "review", "block"]).fillna(0).astype(int)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transactions", f"{len(df):,}")
    k2.metric("Auto-approved", f"{counts['approve']:,}")
    k3.metric("Manual review queue", f"{counts['review']:,}")
    k4.metric("Auto-blocked", f"{counts['block']:,}")
    if TARGET in df:
        st.dataframe(three_way_summary(df[TARGET], decisions, df["amount"]).style.format(
            {"fraud_rate": "{:.1%}", "share_of_volume": "{:.1%}", "amount": "€{:,.0f}"}))
    st.subheader("Review queue (riskiest first)")
    show = df[df.decision != "approve"].sort_values("fraud_probability", ascending=False)
    st.dataframe(show.head(200), width="stretch")
    st.download_button("Download scored file", df.to_csv(index=False).encode(), "scored_transactions.csv")

with tab_whatif:
    if TARGET not in df:
        st.info("Upload a file with an `is_fraud` column to evaluate costs.")
    else:
        curve = cost_curve(df[TARGET], proba, df["amount"], cm)
        t_best, row = best_threshold(curve)
        base = baseline_costs(df[TARGET], df["amount"], cm)
        a, b, c = st.columns(3)
        a.metric("Approve everything", f"€{base['approve everything']:,.0f}")
        b.metric("Review everything", f"€{base['review everything']:,.0f}")
        c.metric(f"Model @ {t_best:.2f}", f"€{row['total_cost']:,.0f}",
                 delta=f"-€{base['approve everything'] - row['total_cost']:,.0f} vs approve-all", delta_color="inverse")
        st.line_chart(curve.set_index("threshold")[["total_cost", "review_cost", "fraud_loss"]])
        st.caption("Total cost = analyst review cost of everything flagged + losses on fraud that slipped through.")

with tab_inspect:
    st.write("Describe a transaction and see how the model reasons about it.")
    c = st.columns(4)
    tx = {
        "user_id": c[0].number_input("user_id", 0, 10_000, 42),
        "amount": c[1].number_input("amount", 0.01, 100_000.0, 1250.0),
        "hour": c[2].number_input("hour", 0, 23, 3),
        "transaction_type": c[3].selectbox("type", sorted(pd.read_csv(DATA_PATH, usecols=["transaction_type"]).iloc[:, 0].unique())),
    }
    c = st.columns(4)
    tx["merchant_category"] = c[0].selectbox("merchant", sorted(pd.read_csv(DATA_PATH, usecols=["merchant_category"]).iloc[:, 0].unique()))
    tx["country"] = c[1].selectbox("country", sorted(pd.read_csv(DATA_PATH, usecols=["country"]).iloc[:, 0].unique()))
    if mode == "full":
        tx["device_risk_score"] = c[2].slider("device risk", 0.0, 1.0, 0.2)
        tx["ip_risk_score"] = c[3].slider("IP risk", 0.0, 1.0, 0.2)
    row = pd.DataFrame([tx])
    p = float(pipe.predict_proba(row)[0, 1])
    d = three_way_decisions([p], t3["review_threshold"], t3["block_threshold"])[0]
    st.metric("Fraud probability", f"{p:.1%}", help=f"decision: {d}")
    st.markdown(f"**Decision:** `{d}`")
    coder = coders.get(mode)
    if coder is not None:
        codes = coder.reason_codes(row, top_k=5)[0]
        st.table(pd.DataFrame(codes)[["label", "value", "contribution", "direction"]])
    else:
        st.info("Reason codes unavailable (shap not installed or model unsupported).")
