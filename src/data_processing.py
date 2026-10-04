"""Data loading and cleaning for the Heart Disease dataset.

These functions are reused by the EDA script, training code, and unit tests.
"""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/processed.cleveland.data")
CLEAN_PATH = Path("data/heart_clean.csv")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw file. Missing values are encoded as '?' in the source."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe.

    Steps:
      1. Drop duplicate rows.
      2. Coerce every column to numeric.
      3. Binarise the target (0 = no disease, 1..4 = disease present).
      4. Impute missing numeric values with the median and
         categorical values with the mode.
    """
    df = df.copy().drop_duplicates().reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors="coerce")

    df[TARGET] = (df[TARGET] > 0).astype(int)

    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode().iloc[0]).astype(int)

    return df


def get_clean_data(save: bool = True) -> pd.DataFrame:
    """Load raw data, clean it, and optionally save to data/heart_clean.csv."""
    df = clean_data(load_raw())
    if save:
        CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEAN_PATH, index=False)
    return df


if __name__ == "__main__":
    raw = load_raw()
    print("Missing values per column (raw):")
    print(raw.isna().sum()[raw.isna().sum() > 0], "\n")
    clean = get_clean_data()
    print(f"Cleaned data saved to {CLEAN_PATH} with shape {clean.shape}")
    print("Remaining missing values:", int(clean.isna().sum().sum()))
