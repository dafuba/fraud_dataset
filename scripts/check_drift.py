"""Compare a new batch of transactions against the training data with PSI.

    python scripts/check_drift.py new_batch.csv [--reference synthetic_fraud_dataset.csv]

Exit code 1 if any feature shows a severe shift – handy as a CI / cron guard.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from fraud_detection.config import BASE_NUMERIC, CATEGORICAL, DATA_PATH, ORACLE
from fraud_detection.drift import psi_report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("current", type=Path)
    ap.add_argument("--reference", type=Path, default=DATA_PATH)
    args = ap.parse_args()
    ref, cur = pd.read_csv(args.reference), pd.read_csv(args.current)
    report = psi_report(ref, cur, [*BASE_NUMERIC, *CATEGORICAL, *ORACLE])
    print(report.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if (report["status"] == "severe shift").any():
        print("\nSEVERE DRIFT detected – consider retraining.")
        sys.exit(1)


if __name__ == "__main__":
    main()
