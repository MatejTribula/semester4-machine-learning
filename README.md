# semester4-machine-learning

<img width="2050" height="971" alt="ML-component-diagram" src="https://github.com/user-attachments/assets/7de5fb15-fa5f-4ad0-93ed-5b04a138f9ac" />

## Setup

### 1. Add dataset to home directory

### 2. Create virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

```bash
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the model

```bash
python main.py
```

### 6. Start the Uvicorn server (in another terminal)

```bash
source venv/bin/activate
uvicorn api:app --reload
```

### 7. Deactivate when done

```bash
deactivate
```

---

## Steps Taken

### 1. Read data

**File:** `data_standardization.py`

Read patient data from an `.xlsx` file into a pandas DataFrame.

### 2. Parse array columns

**File:** `data_standardization.py`

Convert string representations of lists into actual Python lists:

```python
df['hrv_rmssd'] = df['hrv_rmssd'].apply(ast.literal_eval)
df['bpm']       = df['bpm'].apply(ast.literal_eval)
```

### 3. Expand HRV and BPM arrays into statistical features

**File:** `data_standardization.py`

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

**File:** `data_standardization.py`

Remove columns with no predictive signal: `provider`, `userId`, `other`, `timestamp_intervals_seconds_*`, `*_array_length`.

### 5. Convert target to binary label

**File:** `data_standardization.py`

Symptom degree is converted to flare / no flare:

```python
df['label'] = (df['symptom_degree'] > 0).astype(int)
```

### 6. Sort data

**File:** `data_standardization.py`

Order rows by patient and date to ensure chronological integrity:

```python
df = df.sort_values(['terra_user_id', 'date'])
```

### 7. Build sliding window dataset

**File:** `data_standardization.py`

Transform the time series into a supervised learning problem:

1. Group data by patient (`terra_user_id`).
2. Collapse duplicate dates by averaging.
3. Fill missing days using forward fill (`ffill`).
4. Skip patients with insufficient data.
5. Exclude non-feature columns (`terra_user_id`, `date`, `symptom_degree`, `label`).
6. For each window, flatten 14 days × n_features into a single row.
7. Append the flattened window to `X`.
8. Append the next day's binary label to `Y`.
9. Append the patient ID to `pids` for use in splitting.

### 8. Patient-safe train/test split

**File:** `data_standardization.py`

Ensure no patient appears in both train and test sets:

```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, Y, groups=pids))
```

### 9. Assert zero patient overlap

**File:** `data_standardization.py`

```python
assert len(set(pids[train_idx]) & set(pids[test_idx])) == 0
```

### 10. Define model

**File:** `model_training.py`

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

**File:** `model_training.py`

```python
model.fit(X_train, Y_train)
```

### 12. Evaluate

**File:** `evaluate.py`

```python
Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test)[:, 1]
print(classification_report(Y_test, Y_pred))
print("AUC:", roc_auc_score(Y_test, Y_prob))
```

### 13. Log the AUC score

```python
mlflow.log_metric("auc", auc)
```

### 14. Save the model

```python
mlflow.xgboost.log_model(model, "model")
```

### 15. Register the model

```python
mlflow.register_model(
    f"runs:/{mlflow.active_run().info.run_id}/model",
    "xgboost-classifier"
)
```

### 16. Promote the most accurate model to production

```python
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='xgboost-classifier'")
best = max(
    versions,
    key=lambda v: client.get_metric_history(v.run_id, "auc")[-1].value
)
client.transition_model_version_stage(
    name="xgboost-classifier",
    version=best.version,
    stage="Production"
)
```

### 17. Define API endpoint

```python
@app.post("/predict")
# ...
return results
```
