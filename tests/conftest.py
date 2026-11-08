import pandas as pd
import pytest

from fraud_detection.config import DATA_PATH, TARGET


@pytest.fixture(scope="session")
def sample():
    df = pd.read_csv(DATA_PATH).sample(1500, random_state=0)
    X = df.drop(columns=[TARGET, "transaction_id"])
    return X, df[TARGET]
