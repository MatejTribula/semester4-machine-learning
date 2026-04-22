import xgboost as xgb


def train_xgboost(X_train, Y_train, max_depth=6, learning_rate=0.05, n_estimators=300, random_state=42):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, Y_train)
    return model