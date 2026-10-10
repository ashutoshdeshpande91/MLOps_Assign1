"""Tasks 2 + 3: model training, tuning, evaluation and MLflow experiment tracking.

For each model (Logistic Regression, Random Forest):
  * a parent MLflow run logs the best parameters, CV + test metrics,
    plots (ROC curve, confusion matrix, feature importance) and the model;
  * one nested child run per grid-search candidate logs its parameters
    and cross-validated ROC-AUC, so the tuning process is visible in the UI.

Usage (from the repo root):
    python src/train.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://127.0.0.1:5000
"""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed (works in CI/Docker)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
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
TEST_SIZE = 0.2
CV_FOLDS = 5
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "heart_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
SCORING = ["accuracy", "precision", "recall", "roc_auc"]

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "heart-disease-classification"


# --------------------------------------------------------------------------
# Pipeline and models
# --------------------------------------------------------------------------
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
            {"model__C": [0.01, 0.1, 1, 10]},
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
    # float64 avoids MLflow schema warnings about integer columns
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].astype("float64")
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


# --------------------------------------------------------------------------
# Plots (logged to MLflow as artifacts)
# --------------------------------------------------------------------------
def make_roc_figure(estimator, X_test, y_test, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(f"ROC curve - {title}")
    fig.tight_layout()
    return fig


def make_confusion_figure(estimator, X_test, y_test, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        estimator, X_test, y_test, display_labels=["No disease", "Disease"],
        cmap="Blues", ax=ax,
    )
    ax.set_title(f"Confusion matrix - {title}")
    fig.tight_layout()
    return fig


def make_importance_figure(estimator, title, top_n=15):
    """Coefficients for Logistic Regression, impurity importance for the forest."""
    names = estimator.named_steps["preprocess"].get_feature_names_out()
    model = estimator.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values, label = model.feature_importances_, "Feature importance"
    else:
        values, label = model.coef_[0], "Coefficient"
    order = np.argsort(np.abs(values))[-top_n:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(names)[order], values[order])
    ax.set_title(f"{label} (top {top_n}) - {title}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# MLflow logging
# --------------------------------------------------------------------------
def clean_params(params: dict) -> dict:
    """'model__C' -> 'C' for readability in the MLflow UI."""
    return {k.replace("model__", ""): v for k, v in params.items()}


def log_grid_candidates(model_name: str, search: GridSearchCV) -> None:
    """One nested child run per hyperparameter combination tried."""
    res = search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"{model_name}_candidate_{i}", nested=True):
            mlflow.set_tag("run_type", "grid_search_candidate")
            mlflow.log_params(clean_params(params))
            mlflow.log_metric("cv_roc_auc_mean", res["mean_test_score"][i])
            mlflow.log_metric("cv_roc_auc_std", res["std_test_score"][i])
            mlflow.log_metric("rank", int(res["rank_test_score"][i]))


def train_one_model(name, model, grid, cv, data):
    X_train, X_test, y_train, y_test = data

    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id
        mlflow.set_tag("run_type", "best_model")
        mlflow.log_params(
            {
                "model_type": type(model).__name__,
                "cv_folds": CV_FOLDS,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "numeric_features": ",".join(NUMERIC_FEATURES),
                "categorical_features": ",".join(CATEGORICAL_FEATURES),
            }
        )

        # --- tuning (child runs are logged under this run) ---
        search = GridSearchCV(
            build_pipeline(model), grid, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        print(f"Best params: {search.best_params_}")
        log_grid_candidates(name, search)
        mlflow.log_params({f"best_{k}": v for k, v in clean_params(search.best_params_).items()})

        # --- metrics ---
        cv_res = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING)
        row = {"model": name, "best_params": str(clean_params(search.best_params_))}
        for metric in SCORING:
            row[f"cv_{metric}"] = cv_res[f"test_{metric}"].mean()
            row[f"cv_{metric}_std"] = cv_res[f"test_{metric}"].std()
        row.update(evaluate_on_test(best, X_test, y_test))
        mlflow.log_metrics({k: float(v) for k, v in row.items()
                            if k.startswith(("cv_", "test_"))})

        # --- plots ---
        for fname, fig in {
            "roc_curve.png": make_roc_figure(best, X_test, y_test, name),
            "confusion_matrix.png": make_confusion_figure(best, X_test, y_test, name),
            "feature_importance.png": make_importance_figure(best, name),
        }.items():
            mlflow.log_figure(fig, f"plots/{fname}")
            plt.close(fig)

        # --- artifacts: dataset snapshot + the fitted pipeline ---
        mlflow.log_artifact(str(CLEAN_PATH), artifact_path="data")
        signature = infer_signature(X_train, best.predict(X_train))
        mlflow.sklearn.log_model(
            best,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            # newer MLflow defaults to 'skops', which rejects tree models;
            # cloudpickle is fine here because we trained and saved this model ourselves
            serialization_format="cloudpickle",
        )
    return row, best, run_id


def save_final_model(name, estimator, row, run_id) -> None:
    """Save the winning pipeline (preprocessing + model) as one reusable file.

    The .joblib file contains the whole scikit-learn Pipeline, so loading it is
    enough to go from raw patient features to a prediction. The JSON file next to
    it records what the model expects and how it was trained.
    """
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(estimator, MODEL_PATH)

    metadata = {
        "model_name": name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mlflow_run_id": run_id,
        "best_params": row["best_params"],
        "metrics": {k: round(float(v), 4) for k, v in row.items()
                    if k.startswith(("cv_", "test_")) and "std" not in k},
        "input_features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved final model to {MODEL_PATH} and metadata to {METADATA_PATH}")

    # mark the selected run in the MLflow UI
    MlflowClient().set_tag(run_id, "selected_for_serving", "true")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_dataset()
    data = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {data[0].shape}, Test: {data[1].shape}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, (model, grid) in get_candidates().items():
        print(f"\n=== {name} ===")
        row, estimator, run_id = train_one_model(name, model, grid, cv, data)
        rows.append(row)
        fitted[name] = (estimator, run_id)

    results = pd.DataFrame(rows).set_index("model")
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "model_comparison.csv")

    show = [c for c in results.columns if c.startswith(("cv_", "test_")) and "std" not in c]
    print("\nModel comparison:\n", results[show].round(3).T)
    best = results["cv_roc_auc"].idxmax()
    print(f"\nBest model by CV ROC-AUC: {best}")

    best_row = next(r for r in rows if r["model"] == best)
    best_estimator, best_run_id = fitted[best]
    save_final_model(best, best_estimator, best_row, best_run_id)
    print("\nView runs with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
