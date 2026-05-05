from data_standardization import prepare_data
from models.xgb import train_xgboost
from models.lgbm import train_lgbm
from models.randomforest import train_rf
from evaluation import evaluate
import mlflow

from dotenv import load_dotenv
load_dotenv()

import os
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("AWS_ENDPOINT_URL")

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = mlflow.tracking.MlflowClient()
artifact_location = os.getenv("MLFLOW_ARTIFACT_ROOT")
try:
    client.create_experiment("ml-pipeline", artifact_location=artifact_location)
except Exception:
    pass  # experiment already exists

mlflow.set_experiment("ml-pipeline")

MODEL_CONFIG = {
    "xgboost": {"fn": train_xgboost, "name": "xgboost"},
    "lgbm":    {"fn": train_lgbm,    "name": "lightgbm"},
    "rf":      {"fn": train_rf,      "name": "random_forest"},
}


def main():
    X_train, X_test, Y_train, Y_test, train_groups, test_groups, feature_names, num_patients = prepare_data("merged_combined_samples_data.xlsx")

    for key, entry in MODEL_CONFIG.items():
        model_fn   = entry["fn"]
        model_name = entry["name"]

        with mlflow.start_run(run_name=model_name):
            model, tuner = model_fn(X_train, Y_train, train_groups)
            rmse = evaluate(model, tuner, X_test, Y_test, feature_names, num_patients, test_groups)

            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

    # Promote the best model to Production
    client = mlflow.tracking.MlflowClient()
    best_version = None
    best_rmse = float("inf")

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
        print(f"Promoted {best_version[0]} v{best_version[1]} to Production (RMSE: {best_rmse})")


if __name__ == "__main__":
    main()