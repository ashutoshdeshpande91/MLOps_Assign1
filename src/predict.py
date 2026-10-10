"""Inference: load the saved model and predict heart disease risk.

The saved file is a full scikit-learn Pipeline (scaling + one-hot encoding +
classifier), so raw patient values can be passed in directly.

Usage (from the repo root):
    python src/predict.py                       # predict for a built-in sample patient
    python src/predict.py '{"age": 60, ...}'    # predict for your own JSON record
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_PATH = Path("models/heart_model.joblib")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def load_model(path: Path = MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run 'python src/train.py' first."
        )
    return joblib.load(path)


def predict(model, records) -> list:
    """Predict for one record (dict) or several (list of dicts).

    Returns, per record: the predicted class (1 = disease), the confidence in
    that class, and the raw probability of disease.
    """
    if isinstance(records, dict):
        records = [records]
    df = pd.DataFrame(records)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")
    df = df[FEATURES].astype("float64")

    p_disease = model.predict_proba(df)[:, 1]
    results = []
    for p in p_disease:
        label = int(p >= 0.5)
        results.append(
            {
                "prediction": label,
                "confidence": round(float(p if label else 1 - p), 4),
                "probability_disease": round(float(p), 4),
            }
        )
    return results


if __name__ == "__main__":
    patient = json.loads(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_PATIENT
    print("Input:", patient)
    print("Result:", predict(load_model(), patient)[0])
