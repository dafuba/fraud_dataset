"""Score a CSV of transactions offline and write decisions + reason codes.

    python scripts/score_batch.py input.csv output.csv [--mode full|hardmode|auto] [--no-explain]
"""
import argparse
from pathlib import Path

import pandas as pd

from fraud_detection.config import MODELS_DIR, ORACLE, TARGET
from fraud_detection.costs import three_way_decisions
from fraud_detection.explain import ReasonCoder
from fraud_detection.training import load_artifacts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mode", choices=["full", "hardmode", "auto"], default="auto")
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--models", type=Path, default=MODELS_DIR)
    args = ap.parse_args()

    pipes, card = load_artifacts(args.models)
    df = pd.read_csv(args.input)
    mode = args.mode
    if mode == "auto":
        mode = "full" if all(c in df.columns for c in ORACLE) else "hardmode"
    pipe = pipes[mode]
    X = df.drop(columns=[c for c in [TARGET, "transaction_id"] if c in df.columns])
    proba = pipe.predict_proba(X)[:, 1]
    t3 = card["modes"][mode]["three_way"]
    out = df.assign(fraud_probability=proba,
                    decision=three_way_decisions(proba, t3["review_threshold"], t3["block_threshold"]),
                    model_used=mode)
    if not args.no_explain:
        coder = ReasonCoder(pipe)
        out["reason_codes"] = [ReasonCoder.as_text(c) for c in coder.reason_codes(X)]
    out.to_csv(args.output, index=False)
    print(f"scored {len(out):,} rows with the '{mode}' model -> {args.output}")
    print(out["decision"].value_counts().to_string())


if __name__ == "__main__":
    main()
