import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
import ast
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score



#loaded the excel file - data frame
df = pd.read_excel("merged_combined_samples_data.xlsx")


# print(df.shape)
# print(df.dtypes)
# print(df['hrv_rmssd'].iloc[0])   # see what one array looks like
# print(df['bpm'].iloc[0])

# string representation list -> python list
df['hrv_rmssd'] = df['hrv_rmssd'].apply(ast.literal_eval)
df['bpm'] = df['bpm'].apply(ast.literal_eval)


# expand arrays into statistical features
def expand_array_features(df, col):
    arrays = df[col].values

    return pd.DataFrame({
        f'{col}_mean': [np.mean(a) for a in arrays],
        f'{col}_std': [np.std(a) for a in arrays],
        f'{col}_min': [np.min(a) for a in arrays],
        f'{col}_max': [np.max(a) for a in arrays],
        f'{col}_range': [np.max(a) - np.min(a) for a in arrays],
        f'{col}_trend': [a[-1] - a[0] for a in arrays],
        f'{col}_median': [np.median(a) for a in arrays],
    }, index=df.index)

#create statistical features for both hrv and bpm
hrv_features = expand_array_features(df, 'hrv_rmssd')
bpm_features = expand_array_features(df, 'bpm')


# drop all irrelevant columns
df = df.drop(columns=[
    'hrv_rmssd', 'bpm',
    'timestamp_intervals_seconds_hrv_rmssd',
    'hrv_rmssd_array_length',
    'timestamp_intervals_seconds_bpm',
    'bpm_array_length',
    'provider',
    'userId',
    'other'
])

#add hrv and bpm features to cleaned data frame
df = pd.concat([df, hrv_features, bpm_features], axis=1)

#convert symptom degree to binary - check first if symptom degree is larger than 0 and then converts true/false to 1/0
df['label'] = (df['symptom_degree'] > 0).astype(int)

# print(df['label'].value_counts())


# order the data frame based on user and date
df = df.sort_values(['terra_user_id', 'date'])


# ROLLING WINDOW TIMEE

def data_frame_to_supervised(df, window_size = 20, predict_ahead = 1):
    X = []
    Y = []
    pids = []

    for pid, patient_df in df.groupby('terra_user_id'):
        # collapse duplicate dates, exclude string column
        patient_df = patient_df.groupby('date').mean(numeric_only=True).reset_index()
        patient_df['terra_user_id'] = pid

        patient_df = patient_df.set_index('date')
        patient_df = patient_df.resample('D').asfreq()
        patient_df = patient_df.ffill()
        patient_df = patient_df.reset_index()

        if len(patient_df) < window_size + predict_ahead:
            continue

        feature_cols = [c for c in patient_df.columns if c not in ['terra_user_id', 'date', 'symptom_degree', 'label']]

        for i in range(len(patient_df) - window_size - predict_ahead + 1):
            window = patient_df.iloc[i: i + window_size][feature_cols]
            X.append(window.values.flatten())
            Y.append(patient_df.iloc[i + window_size + predict_ahead - 1]['label'])
            pids.append(pid)

    return np.array(X), np.array(Y), np.array(pids)


X, Y, pids = data_frame_to_supervised(df)

# define the splitting process (for testing and training)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# next() takes the first split
train_idx, test_idx = next(gss.split(X, Y, groups=pids))

X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

train_pids = set(pids[train_idx])
test_pids  = set(pids[test_idx])

# verify zero patient overlap
assert len(train_pids & test_pids) == 0, "Patient overlap detected!"

# print("Train samples:", len(X_train), "| Patients:", len(train_pids))
# print("Test samples: ", len(X_test),  "| Patients:", len(test_pids))

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=6,
    learning_rate=0.05,
    n_estimators=300,
    random_state=42
)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(Y_test, Y_pred))
print("AUC:", roc_auc_score(Y_test, Y_prob))

# xgb.plot_importance(bst)
