import numpy as np
import pandas as pd
import xgboost as xgb
import scikit-learn
import ast

#loaded the excel file
df = pd.read_excel("merged_combined_samples_data.xlsx")


# print(df.shape)
# print(df.dtypes)
# print(df['hrv_rmssd'].iloc[0])   # see what one array looks like
# print(df['bpm'].iloc[0])

# string representation list -> python list
df['hrv_rmssd'] = df['hrv_rmssd'].apply(ast.literal_eval)
df['bpm']       = df['bpm'].apply(ast.literal_eval)


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

# sanity check

# print(df.shape)
# print(df.columns.tolist())
# print(df.head(2))

#convert symptom degree to binary - check first if symptom degree is larger than 0 and then converts true/false to 1/0
df['label'] = (df['symptom_degree'] > 0).astype(int)

# print(df['label'].value_counts())

X_train, X_test, Y_train, Y_test = train_test_split()


# ROLLING WINDOW TIMEE