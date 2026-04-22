from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from scipy.stats import randint, uniform
import numpy as np 

def train_lgbm(X_train, Y_train, train_groups):

    base_model = LGBMRegressor(          
        num_leaves=5,
        n_estimators=50,
        learning_rate=0.1,
        min_data_in_leaf=50,
        lambda_l1=0.5,
        lambda_l2=0.5,
        min_gain_to_split=0.1,
        random_state=42
    )

    param_dist = {
        #main
        "num_leaves": randint(3, 12),          
        "n_estimators": randint(30, 150),     
        "learning_rate": uniform(0.05, 0.2),   

        # regularization
        "min_data_in_leaf": randint(20, 80),
        "lambda_l1": uniform(0.2, 0.4),    
        "lambda_l2": uniform(0.2, 0.4),
        "min_gain_to_split": uniform(0.0,0.5),


        # keep structure stable
        "subsample": uniform(0.8, 0.2),
        "colsample_bytree": uniform(0.8, 0.2),
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
    n_jobs= 1
    )
    
    tuner_model.fit(X_train, Y_train, groups=train_groups)

    best_model = tuner_model.best_estimator_
    return best_model, tuner_model #return the actual model along with the tuner containing hyperparameters search process  