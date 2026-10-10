"""Task 2: feature engineering, model training, tuning and evaluation.

Trains Logistic Regression and Random Forest on the cleaned heart disease data, tunes each with GridSearchCV (stratified 5-fold CV, scored on ROC-AUC),
and compares them on cross-validated metrics and a held-out test set.

Usage (from the repo root):
    python src/train.py
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_processing import (
    CATEGORICAL_FEATURES,
    CLEAN_PATH,
    NUMERIC_FEATURES,
    TARGET,
)

RANDOM_STATE = 42
RESULTS_DIR = Path("results")
SCORING = ["accuracy", "precision", "recall", "roc_auc"]


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features, one-hot encode categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    """Preprocessing + model in one object, so training and serving match."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def get_candidates() -> dict:
    """Models and their hyperparameter grids (keys use the 'model__' prefix)."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1, 10],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5, None],
                "model__min_samples_split": [2, 5],
            },
        ),
    }


def load_dataset():
    df = pd.read_csv(CLEAN_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def evaluate_on_test(estimator, X_test, y_test) -> dict:
    pred = estimator.predict(X_test)
    proba = estimator.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


def main() -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, (model, grid) in get_candidates().items():
        print(f"\n=== {name} ===")
        search = GridSearchCV(
            build_pipeline(model), grid, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        search.fit(X_train, y_train)
        print("Best params:", search.best_params_)

        # Cross-validated metrics of the tuned model on the training split
        cv_res = cross_validate(
            search.best_estimator_, X_train, y_train, cv=cv, scoring=SCORING
        )
        row = {"model": name, "best_params": str(search.best_params_)}
        for metric in SCORING:
            row[f"cv_{metric}"] = cv_res[f"test_{metric}"].mean()
            row[f"cv_{metric}_std"] = cv_res[f"test_{metric}"].std()
        row.update(evaluate_on_test(search.best_estimator_, X_test, y_test))
        rows.append(row)

    results = pd.DataFrame(rows).set_index("model")
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "model_comparison.csv")

    show = [c for c in results.columns if c.startswith(("cv_", "test_")) and "std" not in c]
    print("\nModel comparison:\n", results[show].round(3).T)
    best = results["cv_roc_auc"].idxmax()
    print(f"\nBest model by CV ROC-AUC: {best}")


if __name__ == "__main__":
    main()