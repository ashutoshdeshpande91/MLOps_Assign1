"""Unit tests for the model code in src/train.py."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import train
from data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].astype("float64")


def test_preprocessor_scales_numeric_and_expands_categorical(clean_df):
    X = features(clean_df)
    out = train.build_preprocessor().fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else np.asarray(out)

    assert out.shape[0] == len(X)
    assert out.shape[1] > len(FEATURES)  # one-hot encoding adds columns
    # the first columns are the scaled numeric features: mean ~0, std ~1
    assert np.allclose(out[:, : len(NUMERIC_FEATURES)].mean(axis=0), 0, atol=1e-8)
    assert np.allclose(out[:, : len(NUMERIC_FEATURES)].std(axis=0), 1, atol=1e-8)


def test_preprocessor_handles_unseen_category(clean_df, trained_pipeline):
    row = features(clean_df).iloc[[0]].copy()
    row["cp"] = 99.0  # a category never seen during training
    proba = trained_pipeline.predict_proba(row)
    assert proba.shape == (1, 2)


def test_pipeline_probabilities_are_valid(clean_df, trained_pipeline):
    proba = trained_pipeline.predict_proba(features(clean_df))
    assert proba.shape == (len(clean_df), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1)


def test_candidates_define_two_models_with_prefixed_grids():
    candidates = train.get_candidates()
    assert set(candidates) == {"logistic_regression", "random_forest"}
    for _, grid in candidates.values():
        assert grid  # not empty
        assert all(key.startswith("model__") for key in grid)


def test_evaluate_on_test_returns_metrics_between_0_and_1(clean_df, trained_pipeline):
    X, y = features(clean_df), clean_df[TARGET]
    metrics = train.evaluate_on_test(trained_pipeline, X, y)
    assert set(metrics) == {
        "test_accuracy", "test_precision", "test_recall", "test_roc_auc",
    }
    assert all(0 <= v <= 1 for v in metrics.values())


def test_load_dataset_returns_float_features(monkeypatch, tmp_path, clean_df):
    path = tmp_path / "heart_clean.csv"
    clean_df.to_csv(path, index=False)
    monkeypatch.setattr(train, "CLEAN_PATH", path)

    X, y = train.load_dataset()

    assert list(X.columns) == FEATURES
    assert (X.dtypes == "float64").all()
    assert len(X) == len(y) == len(clean_df)


def test_training_is_reproducible(clean_df):
    X, y = features(clean_df), clean_df[TARGET]

    def fit_once():
        model = RandomForestClassifier(n_estimators=20, random_state=train.RANDOM_STATE)
        return train.build_pipeline(model).fit(X, y).predict_proba(X)

    assert np.array_equal(fit_once(), fit_once())


def test_save_final_model_writes_loadable_model_and_metadata(
    monkeypatch, tmp_path, clean_df, trained_pipeline
):
    class FakeClient:  # avoids needing a real MLflow run
        def set_tag(self, *args, **kwargs):
            pass

    model_path = tmp_path / "models" / "heart_model.joblib"
    meta_path = tmp_path / "models" / "model_metadata.json"
    monkeypatch.setattr(train, "MODEL_DIR", model_path.parent)
    monkeypatch.setattr(train, "MODEL_PATH", model_path)
    monkeypatch.setattr(train, "METADATA_PATH", meta_path)
    monkeypatch.setattr(train, "MlflowClient", FakeClient)

    row = {"best_params": "{'C': 1}", "cv_roc_auc": 0.9, "test_accuracy": 0.8,
           "cv_roc_auc_std": 0.01}
    train.save_final_model("logistic_regression", trained_pipeline, row, "run123")

    meta = json.loads(meta_path.read_text())
    assert meta["model_name"] == "logistic_regression"
    assert meta["mlflow_run_id"] == "run123"
    assert meta["input_features"] == FEATURES
    assert "cv_roc_auc_std" not in meta["metrics"]  # std values are left out

    reloaded = joblib.load(model_path)
    X = features(clean_df)
    assert np.allclose(reloaded.predict_proba(X), trained_pipeline.predict_proba(X))


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_each_candidate_trains_inside_pipeline(clean_df, name):
    model, _ = train.get_candidates()[name]
    pipe = train.build_pipeline(model).fit(features(clean_df), clean_df[TARGET])
    assert pipe.predict(features(clean_df)).shape == (len(clean_df),)
