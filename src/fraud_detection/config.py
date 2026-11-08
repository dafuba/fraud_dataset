"""Central configuration: paths, column groups and the random seed."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "synthetic_fraud_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42
TEST_SIZE = 0.2

TARGET = "is_fraud"
ID_COLUMNS = ["transaction_id", "user_id"]
CATEGORICAL = ["transaction_type", "merchant_category", "country"]
BASE_NUMERIC = ["amount", "hour"]
# Third-party "vendor" risk scores. In this dataset they are effectively an oracle
# for the label, which is why the project also trains a model without them.
ORACLE = ["device_risk_score", "ip_risk_score"]

RAW_INPUT_COLUMNS = ["user_id", *BASE_NUMERIC, *CATEGORICAL, *ORACLE]

MODES = ("full", "hardmode")
