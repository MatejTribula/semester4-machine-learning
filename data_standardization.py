import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def expand_array_features(df, col):
    arrays = df[col].values
    return pd.DataFrame({
        f'{col}_mean':   [np.mean(a) for a in arrays],
        f'{col}_std':    [np.std(a) for a in arrays],
        f'{col}_min':    [np.min(a) for a in arrays],
        f'{col}_max':    [np.max(a) for a in arrays],
        f'{col}_range':  [np.max(a) - np.min(a) for a in arrays],
        f'{col}_trend':  [a[-1] - a[0] for a in arrays],
        f'{col}_median': [np.median(a) for a in arrays],
    }, index=df.index)


def data_frame_to_supervised(df, window_size=14, predict_ahead=1):
    X, Y, pids = [], [], []

    for pid, patient_df in df.groupby('terra_user_id'):
        patient_df = patient_df.groupby('date').mean(numeric_only=True).reset_index()
        patient_df['terra_user_id'] = pid
        patient_df = patient_df.set_index('date').resample('D').asfreq().ffill().reset_index()

        if len(patient_df) < window_size + predict_ahead:
            continue

        feature_cols = [c for c in patient_df.columns
                        if c not in ['terra_user_id', 'date', 'symptom_degree', 'label']]

        for i in range(len(patient_df) - window_size - predict_ahead + 1):
            window = patient_df.iloc[i: i + window_size][feature_cols]
            X.append(window.values.flatten())
            Y.append(patient_df.iloc[i + window_size + predict_ahead - 1]['label'])
            pids.append(pid)

    return np.array(X), np.array(Y), np.array(pids)


def prepare_data(filepath, window_size=14, predict_ahead=1, test_size=0.2, random_state=42):
    # load
    df = pd.read_excel(filepath)

    # parse string lists
    df['hrv_rmssd'] = df['hrv_rmssd'].apply(ast.literal_eval)
    df['bpm'] = df['bpm'].apply(ast.literal_eval)

    # feature engineering
    hrv_features = expand_array_features(df, 'hrv_rmssd')
    bpm_features = expand_array_features(df, 'bpm')

    # drop irrelevant columns
    df = df.drop(columns=[
        'hrv_rmssd', 'bpm',
        'timestamp_intervals_seconds_hrv_rmssd',
        'hrv_rmssd_array_length',
        'timestamp_intervals_seconds_bpm',
        'bpm_array_length',
        'provider', 'userId', 'other'
    ])

    df = pd.concat([df, hrv_features, bpm_features], axis=1)

    # binary label
    df['label'] = (df['symptom_degree'] > 0).astype(int)

    # sort
    df = df.sort_values(['terra_user_id', 'date'])

    # rolling window
    X, Y, pids = data_frame_to_supervised(df, window_size, predict_ahead)

    # train/test split (no patient overlap)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, Y, groups=pids))

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    train_pids = set(pids[train_idx])
    test_pids  = set(pids[test_idx])
    assert len(train_pids & test_pids) == 0, "Patient overlap detected!"

    return X_train, X_test, Y_train, Y_test
