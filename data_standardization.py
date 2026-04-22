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


def data_frame_to_supervised(df, window_size=5, predict_ahead=1):

    X = []
    Y = []
    pids = []
    feature_names = None 

    for pid, patient_df in df.groupby('terra_user_id'):

        patient_df = patient_df.sort_values('date').copy()
        patient_df = patient_df.groupby('date', as_index=False).mean(numeric_only=True)
        patient_df = patient_df.set_index('date').asfreq('D')
        patient_df = patient_df.ffill()
        patient_df = patient_df.reset_index()

        if 'label' not in patient_df.columns:
            continue

        if len(patient_df) < window_size + predict_ahead:
            continue

        feature_cols = [c for c in patient_df.columns if c not in ['date', 'symptom_degree', 'label']]
        values = patient_df[feature_cols].values

        # include feature names
        if feature_names is None:
            feature_names = [
                f"{col}_t-{window_size - t}"
                for t in range(window_size)
                for col in feature_cols
            ]

        for i in range(0, len(patient_df) - window_size, 1):

            window = values[i:i + window_size]
            target_idx = i + window_size + predict_ahead - 1

            if target_idx >= len(patient_df):
                continue

            target = patient_df.iloc[target_idx]['label']

            if pd.isna(target):
                continue

            X.append(window.flatten())
            Y.append(target)
            pids.append(pid)

    return np.array(X), np.array(Y), np.array(pids), feature_names

def prepare_data(filepath, window_size=10, predict_ahead=1, test_size=0.2, random_state=42):
  
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

    #add hrv and bpm features to cleaned data frame
    df = pd.concat([df, hrv_features, bpm_features], axis=1)

    # set regression target (0–7 symptom severity)
    df['label'] = df['symptom_degree']

    # possibility to filter patients 
    df = df[df.groupby('terra_user_id')['label'].transform('sum') >= 1]
    num_patients = df['terra_user_id'].nunique()

    # order the data frame based on user and date
    df = df.sort_values(['terra_user_id', 'date'])

    # rolling window
    X, Y, pids, feature_names = data_frame_to_supervised(df)

    # define the splitting process (for testing and training)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # train/test split (no patient overlap)
    train_idx, test_idx = next(gss.split(X, Y, groups=pids))

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    train_pids = set(pids[train_idx]) # Unique set of patients in the training set
    test_pids = set(pids[test_idx]) # Unique set of patients in the test set

    train_groups = pids[train_idx] # Patient IDs for each training sample (keeps grouping info for GroupKFold / CV)

    test_groups = pids[test_idx] # Patient IDs for each test sample

    #verify zero patient overlap
    assert len(train_pids & test_pids) == 0, "Patient leakage detected!"

    return X_train, X_test, Y_train, Y_test, train_groups, test_groups, feature_names, num_patients
