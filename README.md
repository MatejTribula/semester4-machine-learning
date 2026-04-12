# semester4-machine-learning

## setup
### 1. Create the virtual environment
`python3 -m venv venv`

### 2. Activate it
`source venv/bin/activate`        # Mac/Linux
<br>
`venv\Scripts\activate`           # Windows

### 3. Install your dependencies
`pip install -r requirements.txt`

### 4. When you're done working, deactivate
`deactivate`



## Steps Taken

1. Read data from .xlsx file
2. convert representation lists --> python lists
```
df['hrv_rmssd'] = df['hrv_rmssd'].apply(ast.literal_eval)
df['bpm'] = df['bpm'].apply(ast.literal_eval)
```

3. expand bpm and hrv arrays into statistical features
```
f'{col}_mean': [np.mean(a) for a in arrays],
f'{col}_std': [np.std(a) for a in arrays],
f'{col}_min': [np.min(a) for a in arrays],
f'{col}_max': [np.max(a) for a in arrays],
.
.
.
```
4. drop irrelevant columns
5. convert symptom degree to binary - 0 or 1
6. order data based on terra_user_id and date
7. define sliding window 
   1. group data by user
   2. fill in missing days
   3. drop all patients with small number of readings
   4. drop irrelevant columns (e.g., user_id, date, symptom_degree)
   5. for each window flatten the data into single column
   6. add rows to X
   7. add label (converted symptom to binary) to Y
8. define splitting process
```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

```
GroupShuffleSplit makes sure data of one user is only either in Testing or Training, not both
9. split the dataset
10. assert no patients overlap
11. define model
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
12. train model
```python
model.fit(X_train, Y_train)

```
13.  receive AUC score of a model