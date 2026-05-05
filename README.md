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

### 6. Start the MLflow UI (in another terminal)
```bash
source venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

### 7. Start the Uvicorn server (in another terminal)

```bash
source venv/bin/activate
uvicorn api:app --reload
```

### 7*. Deactivate when done

```bash
deactivate
```

---

## How It Works

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

### 5. Set regression target

**File:** `data_standardization.py`

Symptom degree (0–7) is used directly as the regression target:

```python
df['label'] = df['symptom_degree']
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
4. Skip patients with insufficient data (fewer than `window_size + predict_ahead` days).
5. Exclude non-feature columns (`terra_user_id`, `date`, `symptom_degree`, `label`).
6. For each window, flatten `window_size` days × n_features into a single row.
7. Append the flattened window to `X`.
8. Append the target day's symptom degree (0–7) to `Y`.
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

### 10. Define and train models with hyperparameter tuning

**Files:** `models/xgb.py`, `models/lgbm.py`, `models/randomforest.py`

Three regressors are trained: XGBoost, LightGBM, and Random Forest. Each uses `RandomizedSearchCV` to randomly sample 10–20 hyperparameter combinations and evaluate them with 5-fold `GroupKFold` cross validation (no patient overlap across folds). The best combination per model is selected automatically.

```python
tuner_model = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=10,
    scoring="neg_root_mean_squared_error",
    cv=GroupKFold(n_splits=5),
)
tuner_model.fit(X_train, Y_train, groups=train_groups)
best_model = tuner_model.best_estimator_
```

### 11. Evaluate the models

**File:** `evaluation.py`

Each model is evaluated on the held-out test set with the following metrics:

- **MAE** — average absolute error
- **RMSE** — root mean squared error
- **NRMSE** — RMSE normalised by the range of the target
- **Spearman correlation** — how well the predicted ranking matches the true ranking
- **Per-patient RMSE** — mean, std, and worst RMSE broken down by patient

### 12. Log the RMSE metric

```python
mlflow.log_metric("rmse", rmse)
```

### 14. Save the model

```python
mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
```

### 16. Promote the most accurate model to production

```python
    for entry in MODEL_CONFIG.values():
        model_name = entry["name"]
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            rmse = client.get_metric_history(v.run_id, "rmse")[-1].value
            if rmse < best_rmse:
                best_rmse = rmse
                best_version = (model_name, v.version)

    if best_version:
        client.transition_model_version_stage(
            name=best_version[0],
            version=best_version[1],
            stage="Production"
        )
```

### 17. Define API endpoint

```python
@app.post("/predict")
# ...
return results
```
