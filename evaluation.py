from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

def evaluate(model, tuner_model, X_test, Y_test, feature_names, num_patients, test_groups):

    # feature importance
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("======================================================")
    print("\nRemaining patients:", num_patients)
    print("\n================ FEATURE IMPORTANCE ================")
    print(importance_df.head(10))

    # parameters
    print("\n================ BEST PARAMETERS ================")
    print(tuner_model.best_params_)
    print("\nBest CV score (RMSE):", -tuner_model.best_score_) # Root Mean Squared Error

    # predictions
    test_pred = model.predict(X_test)
   
    # basic metrics
    test_mae = mean_absolute_error(Y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(Y_test, test_pred))
    nrmse = test_rmse / (np.max(Y_test) - np.min(Y_test))

    print("\n================ TEST RESULTS ================")
    print("TEST MAE :", test_mae) # average absolute difference between predictions and true values
    print("TEST RMSE:", test_rmse) # Root Mean Squared Error on the test set
    print("NRMSE    :", nrmse) # Normalized Root Mean Squared Error

    # spearman correlation - how well the relationship between two variables follows a monotonic order
    corr, _ = spearmanr(Y_test, test_pred)
    print("Spearman correlation:", corr)


# patient level metrics

    if test_groups is not None:

        test_groups = np.array(test_groups)  # ensure numpy array

        if len(test_groups) != len(Y_test):
            raise ValueError(
                f"Mismatch: len(test_groups)={len(test_groups)} vs len(Y_test)={len(Y_test)}. "
                "You must pass test-set pids only."
            )

        df_eval = pd.DataFrame({
            "y_true": Y_test,
            "y_pred": test_pred,
            "pid": test_groups
        })

        patient_rmses = []

        for pid, group in df_eval.groupby("pid"):
            rmse = np.sqrt(mean_squared_error(group["y_true"], group["y_pred"]))
            patient_rmses.append(rmse)
        
        worst_patient_rmse = np.max(patient_rmses)

        print("\n================ BY-PATIENET METRICS ================")
        print("Patient RMSE mean:", np.mean(patient_rmses)) # tells how much error the model makes per patient on average
        print("Patient RMSE std :", np.std(patient_rmses)) # measures how consistent the model is across different patients
        print("Worst Patient RMSE:", worst_patient_rmse) # worst patient performance

    else:
        print("\n test_groups not provided → patient-level metrics skipped")

    return test_rmse