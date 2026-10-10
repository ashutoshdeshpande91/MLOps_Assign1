"""Shared test fixtures.

All tests run on small synthetic data, so they need no network access and no
downloaded dataset, which makes them fast and safe to run in CI.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from data_processing import COLUMNS, clean_data
from train import RANDOM_STATE, build_pipeline


def make_row(**overrides) -> dict:
    """One valid raw patient record; override any field to change it."""
    row = {
        "age": 55, "sex": 1, "cp": 3, "trestbps": 130, "chol": 240, "fbs": 0,
        "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 2,
        "ca": 0.0, "thal": 3.0, "target": 0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Synthetic raw data shaped like the file read by load_raw()
    (target 0-4, some missing values in chol, ca and thal)."""
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(
        {
            "age": rng.integers(30, 77, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(1, 5, n),
            "trestbps": rng.integers(94, 200, n),
            "chol": rng.integers(126, 400, n).astype(float),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.integers(71, 202, n),
            "exang": rng.integers(0, 2, n),
            "oldpeak": rng.random(n) * 6,
            "slope": rng.integers(1, 4, n),
            "ca": rng.integers(0, 4, n).astype(float),
            "thal": rng.choice([3.0, 6.0, 7.0], n),
            "target": rng.integers(0, 5, n),
        }
    )[COLUMNS]
    df.loc[[3, 10, 20], "ca"] = np.nan
    df.loc[[5, 15], "thal"] = np.nan
    df.loc[[7], "chol"] = np.nan
    df.loc[:5, "target"] = [0, 1, 0, 2, 0, 1]  # guarantee both classes
    return df


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    return clean_data(raw_df)


@pytest.fixture
def trained_pipeline(clean_df):
    """A small fitted preprocessing + Logistic Regression pipeline."""
    X = clean_df.drop(columns="target").astype("float64")
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    return pipe.fit(X, clean_df["target"])
