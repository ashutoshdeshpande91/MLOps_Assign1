"""Unit tests for the inference code in src/predict.py."""
import joblib
import pytest

import predict


def test_sample_patient_has_every_required_feature():
    assert set(predict.FEATURES) <= set(predict.SAMPLE_PATIENT)


def test_load_model_raises_clear_error_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        predict.load_model(tmp_path / "nope.joblib")


def test_load_model_roundtrip(tmp_path, trained_pipeline):
    path = tmp_path / "model.joblib"
    joblib.dump(trained_pipeline, path)
    assert predict.load_model(path).predict_proba is not None


def test_predict_single_record_returns_expected_fields(trained_pipeline):
    result = predict.predict(trained_pipeline, predict.SAMPLE_PATIENT)
    assert len(result) == 1
    item = result[0]
    assert set(item) == {"prediction", "confidence", "probability_disease"}
    assert item["prediction"] in (0, 1)
    assert 0.5 <= item["confidence"] <= 1
    assert 0 <= item["probability_disease"] <= 1


def test_confidence_matches_predicted_class(trained_pipeline):
    item = predict.predict(trained_pipeline, predict.SAMPLE_PATIENT)[0]
    p = item["probability_disease"]
    expected = p if item["prediction"] == 1 else 1 - p
    assert item["confidence"] == pytest.approx(expected, abs=1e-3)


def test_predict_multiple_records(trained_pipeline):
    other = {**predict.SAMPLE_PATIENT, "age": 40, "thalach": 180}
    results = predict.predict(trained_pipeline, [predict.SAMPLE_PATIENT, other])
    assert len(results) == 2


def test_predict_rejects_missing_features(trained_pipeline):
    with pytest.raises(ValueError, match="Missing required feature"):
        predict.predict(trained_pipeline, {"age": 50})


def test_predict_ignores_extra_keys(trained_pipeline):
    patient = {**predict.SAMPLE_PATIENT, "unexpected": 123}
    assert len(predict.predict(trained_pipeline, patient)) == 1
