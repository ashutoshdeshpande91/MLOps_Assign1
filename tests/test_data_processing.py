"""Unit tests for src/data_processing.py."""
import numpy as np
import pandas as pd
import pytest

import data_processing as dp
from conftest import make_row


def test_feature_lists_cover_all_columns_without_overlap():
    features = dp.NUMERIC_FEATURES + dp.CATEGORICAL_FEATURES
    assert len(features) == len(set(features))
    assert set(features) | {dp.TARGET} == set(dp.COLUMNS)


def test_load_raw_converts_question_marks_to_nan(tmp_path):
    path = tmp_path / "raw.data"
    path.write_text(
        "63,1,1,145,233,1,2,150,0,2.3,3,0.0,6.0,0\n"
        "67,1,4,160,286,0,2,108,1,1.5,2,?,3.0,2\n"
    )
    df = dp.load_raw(path)
    assert list(df.columns) == dp.COLUMNS
    assert len(df) == 2
    assert np.isnan(df.loc[1, "ca"])


def test_clean_data_binarizes_target(raw_df):
    clean = dp.clean_data(raw_df)
    assert set(clean[dp.TARGET].unique()) <= {0, 1}
    # a raw value of 0 stays 0, anything above 0 becomes 1
    sample = pd.DataFrame([make_row(age=40, target=0), make_row(age=41, target=3)])
    assert dp.clean_data(sample)[dp.TARGET].tolist() == [0, 1]


def test_clean_data_leaves_no_missing_values(raw_df):
    assert raw_df.isna().sum().sum() > 0  # the fixture really has gaps
    assert dp.clean_data(raw_df).isna().sum().sum() == 0


def test_clean_data_imputes_median_and_mode():
    rows = [
        make_row(age=30, chol=200.0, ca=1.0),
        make_row(age=31, chol=220.0, ca=1.0),
        make_row(age=32, chol=240.0, ca=2.0),
        make_row(age=33, chol=np.nan, ca=np.nan),
        make_row(age=34, chol=260.0, ca=0.0),
    ]
    clean = dp.clean_data(pd.DataFrame(rows))
    gap_row = clean[clean["age"] == 33].iloc[0]
    assert gap_row["chol"] == 230.0  # median of 200, 220, 240, 260
    assert gap_row["ca"] == 1  # most frequent value


def test_clean_data_drops_duplicate_rows():
    df = pd.DataFrame([make_row(age=50), make_row(age=50), make_row(age=51)])
    assert len(dp.clean_data(df)) == 2


def test_clean_data_does_not_modify_input(raw_df):
    before = raw_df.copy()
    dp.clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_categorical_columns_become_integers(clean_df):
    for col in dp.CATEGORICAL_FEATURES:
        assert pd.api.types.is_integer_dtype(clean_df[col]), col


def test_get_clean_data_saves_csv(monkeypatch, tmp_path, raw_df):
    out = tmp_path / "out" / "heart_clean.csv"
    monkeypatch.setattr(dp, "load_raw", lambda: raw_df)
    monkeypatch.setattr(dp, "CLEAN_PATH", out)

    df = dp.get_clean_data(save=True)

    assert out.exists()
    assert pd.read_csv(out).shape == df.shape


@pytest.mark.parametrize("col", dp.NUMERIC_FEATURES)
def test_numeric_features_are_numeric(clean_df, col):
    assert pd.api.types.is_numeric_dtype(clean_df[col])
