from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from scipy.stats import randint
import numpy as np


def train_rf(X_train, y_train, train_groups):

    base_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    param_dist = {
        # main tree complexity controls
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 25),

        # split control
        "min_samples_split": randint(2, 30),
        "min_samples_leaf": randint(1, 20),

        # feature sampling
        "max_features": ["sqrt", "log2", None],

        # bootstrap sampling behavior
        "bootstrap": [True, False],
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

    tuner_model.fit(X_train, y_train, groups=train_groups)

    best_model = tuner_model.best_estimator_

    return best_model, tuner_model  