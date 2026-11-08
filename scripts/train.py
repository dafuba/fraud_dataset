"""Train both models, pick cost-optimal thresholds, export artefacts and a model card.

    python scripts/train.py [--data PATH] [--out DIR] [--review-cost 5] [--chargeback-fee 15]
"""
import argparse
import json
from pathlib import Path

from fraud_detection.config import DATA_PATH, MODELS_DIR
from fraud_detection.costs import CostModel
from fraud_detection.training import train_and_export


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    ap.add_argument("--out", type=Path, default=MODELS_DIR)
    ap.add_argument("--review-cost", type=float, default=5.0)
    ap.add_argument("--chargeback-fee", type=float, default=15.0)
    ap.add_argument("--block-friction-cost", type=float, default=25.0)
    args = ap.parse_args()
    cm = CostModel(review_cost=args.review_cost, chargeback_fee=args.chargeback_fee,
                   block_friction_cost=args.block_friction_cost)
    card = train_and_export(out_dir=args.out, cm=cm, data_path=args.data)
    for mode, e in card["modes"].items():
        m = e["holdout_metrics"]
        print(f"[{mode:8s}] PR-AUC={m['PR-AUC']:.4f} ROC-AUC={m['ROC-AUC']:.4f} "
              f"precision={m['precision']:.3f} recall={m['recall']:.3f} @ threshold {m['threshold']:.2f} "
              f"| three-way review≥{e['three_way']['review_threshold']:.2f} block≥{e['three_way']['block_threshold']:.2f}")
    print(f"artefacts written to {args.out}/ ; model card:\n{json.dumps({k: card[k] for k in ('model', 'trained_at', 'cost_model')}, indent=2)}")


if __name__ == "__main__":
    main()
