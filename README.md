# semester4-machine-learning

## Setup

### 1. Add dataset to home directory

### 2. Create virtual environment
`python3 -m venv venv`

### 3. Activate it
`source venv/bin/activate`   # Mac/Linux  
`venv\Scripts\activate`      # Windows

### 4. Install dependencies
`pip install -r requirements.txt`

### 5. Deactivate when done
`deactivate`

---

## Steps Taken

### 1. Read data
Read patient data from `.xlsx` file into a pandas DataFrame.

### 2. Parse array columns
Convert string representations of lists into actual Python lists:
```python
df['hrv_rmssd'] = df['hrv_rmssd'].apply(ast.literal_eval)
df['bpm']       = df['bpm'].apply(ast.literal_eval)
```

### 3. Expand HRV and BPM arrays into statistical features
Each array of readings is reduced to scalar statistics:
```python
f'{col}_mean':   [np.mean(a) for a in arrays]
f'{col}_std':    [np.std(a) for a in arrays]
f'{col}_min':    [np.min(a) for a in arrays]
f'{col}_max':    [np.max(a) for a in arrays]
f'{col}_range':  [np.max(a) - np.min(a) for a in arrays]
f'{col}_trend':  [a[-1] - a[0] for a in arrays]
f'{col}_median': [np.median(a) for a in arrays]
```

### 4. Drop irrelevant columns
Remove columns with no predictive signal: `provider`, `userId`, `other`,
`timestamp_intervals_seconds_*`, `*_array_length`.

### 5. Convert target to binary label
Symptom degree converted to flare / no flare:
```python
df['label'] = (df['symptom_degree'] > 0).astype(int)
```

### 6. Sort data
Order rows by patient and date to ensure chronological integrity:
```python
df = df.sort_values(['terra_user_id', 'date'])
```

### 7. Build sliding window dataset
Transform the time series into a supervised learning problem:

1. Group data by patient (`terra_user_id`)
2. Collapse duplicate dates by averaging
3. Fill missing days using forward fill (`ffill`)
4. Skip patients with insufficient data
5. Exclude non-feature columns (`terra_user_id`, `date`, `symptom_degree`, `label`)
6. For each window, flatten 14 days × n_features into a single row
7. Append flattened window to `X`
8. Append next day's binary label to `Y`
9. Append patient ID to `pids` for use in splitting

### 8. Patient-safe train/test split
Ensure no patient appears in both train and test sets:
```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, Y, groups=pids))
```

### 9. Assert zero patient overlap
```python
assert len(set(pids[train_idx]) & set(pids[test_idx])) == 0
```

### 10. Define model
```python
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=6,
    learning_rate=0.05,
    n_estimators=300,
    random_state=42
)
```

### 11. Train model
```python
model.fit(X_train, Y_train)
```

### 12. Evaluate
```python
Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(Y_test, Y_pred))
print("AUC:", roc_auc_score(Y_test, Y_prob))
```