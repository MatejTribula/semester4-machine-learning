import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from scipy.stats import randint, uniform


def train_xgboost(X_train, Y_train, train_groups):

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2
    )

    param_dist = {
        # tree complexity
        "max_depth": randint(3, 10),
        "min_child_weight": randint(1, 10),

        # learning dynamics
        "learning_rate": uniform(0.03, 0.2),
        "n_estimators": randint(100, 600),

        # subsampling
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3),

        # regularization
        "gamma": uniform(0, 0.5),
        "reg_lambda": uniform(0.5, 2.0),
        "reg_alpha": uniform(0.0, 1.0),
    }

    cv = GroupKFold(n_splits=5)

    tuner_model = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=1
    )

    tuner_model.fit(X_train, Y_train, groups=train_groups)

    best_model = tuner_model.best_estimator_

    return best_model, tuner_model 