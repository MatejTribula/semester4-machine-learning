import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from data_standardization import (
    preprocess,
    data_frame_to_supervised,
    get_inference_windows,
)


def make_sample_df():
    # Small fake dataset used only for testing
    return pd.DataFrame({
        "terra_user_id": ["p1"] * 6,
        "date": pd.date_range("2026-01-01", periods=6),
        "symptom_degree": [0, 1, 2, 3, 4, 5],
        "label": [0, 1, 2, 3, 4, 5],

        # These are strings because the real dataset stores arrays as text
        "hrv_rmssd": ["[10, 20, 30]"] * 6,
        "bpm": ["[70, 80, 90]"] * 6,

        # Columns removed by preprocess()
        "timestamp_intervals_seconds_hrv_rmssd": [None] * 6,
        "hrv_rmssd_array_length": [3] * 6,
        "timestamp_intervals_seconds_bpm": [None] * 6,
        "bpm_array_length": [3] * 6,
        "provider": ["terra"] * 6,
        "userId": ["u1"] * 6,
        "other": [None] * 6,

        # Normal feature columns
        "sleep": [7, 7, 6, 8, 7, 6],
        "activity": [100, 120, 130, 140, 150, 160],
        "diarrhea": [0, 0, 1, 0, 1, 0],
        "stomachPain": [0, 1, 0, 1, 0, 1],
    })


def test_preprocess_creates_hrv_and_bpm_features():
    df = make_sample_df()

    result = preprocess(df)

    assert "hrv_rmssd_mean" in result.columns
    assert "hrv_rmssd_std" in result.columns
    assert "hrv_rmssd_min" in result.columns
    assert "hrv_rmssd_max" in result.columns
    assert "hrv_rmssd_range" in result.columns
    assert "hrv_rmssd_trend" in result.columns
    assert "hrv_rmssd_median" in result.columns

    assert "bpm_mean" in result.columns
    assert "bpm_std" in result.columns
    assert "bpm_min" in result.columns
    assert "bpm_max" in result.columns
    assert "bpm_range" in result.columns
    assert "bpm_trend" in result.columns
    assert "bpm_median" in result.columns


def test_preprocess_removes_original_array_columns():
    df = make_sample_df()

    result = preprocess(df)

    assert "hrv_rmssd" not in result.columns
    assert "bpm" not in result.columns


def test_preprocess_removes_irrelevant_columns():
    df = make_sample_df()

    result = preprocess(df)

    assert "timestamp_intervals_seconds_hrv_rmssd" not in result.columns
    assert "hrv_rmssd_array_length" not in result.columns
    assert "timestamp_intervals_seconds_bpm" not in result.columns
    assert "bpm_array_length" not in result.columns
    assert "provider" not in result.columns
    assert "userId" not in result.columns
    assert "other" not in result.columns


def test_data_frame_to_supervised_creates_training_arrays():
    df = preprocess(make_sample_df())

    X, Y, pids, feature_names = data_frame_to_supervised(df, window_size=3, predict_ahead=1)

    assert len(X) > 0
    assert len(Y) > 0
    assert len(pids) > 0

    assert X.shape[0] == Y.shape[0] 
    assert X.shape[0] == pids.shape[0]  

    assert feature_names is not None # Checks that feature names were generated
    assert len(feature_names) == X.shape[1] # Checks that feature names were generated for all features in the window


def test_data_frame_to_supervised_skips_patients_with_too_little_data():
    df = preprocess(make_sample_df().head(2))

    X, Y, pids, feature_names = data_frame_to_supervised(df, window_size=5, predict_ahead=1)

    assert len(X) == 0
    assert len(Y) == 0
    assert len(pids) == 0


def test_get_inference_windows_returns_none_when_not_enough_data():
    df = preprocess(make_sample_df().head(2))

    results = get_inference_windows(df, window_size=5, predict_ahead=1)

    assert len(results) == 2

    for patient_id, date, window in results:
        assert patient_id == "p1"
        assert window is None


def test_get_inference_windows_creates_flattened_window():
    df = preprocess(make_sample_df())

    results = get_inference_windows(df, window_size=5, predict_ahead=1)

    assert len(results) == 1

    patient_id, prediction_date, window = results[0]

    assert patient_id == "p1"
    assert window is not None
    assert len(window.shape) == 1