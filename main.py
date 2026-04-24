from data_standardization import prepare_data
from models.xgb import train_xgboost
from models.lgbm import train_lgbm
from models.randomforest import train_rf
from evaluation import evaluate
import mlflow

#dict temp easy solution for separating models
MODEL_CONFIG = {
    "xgboost": {
        "fn": train_xgboost,
        "name": "xgboost"
    },
    "lgbm": {
        "fn": train_lgbm,
        "name": "lightgbm"
    },
    "rf": {
        "fn": train_rf,
        "name": "random_forest"
    }
}

def main():

    mlflow.xgboost.autolog()  # Must be called before training

    X_train, X_test, Y_train, Y_test, train_groups, test_groups, feature_names, num_patients = prepare_data("merged_combined_samples_data.xlsx") #default prepare_data parameters used
    
    with mlflow.start_run():  # Wrap in a run context

        #dict lookup
        entry = MODEL_CONFIG["lgbm"]
        model_fn = entry["fn"] 
        model_name = entry["name"]

        # return the actual model along with the tuner containing hyperparameters search process
        model, tuner = model_fn(X_train, Y_train, train_groups) 

        # a bit of a rudimentary metric
        auc = evaluate(model, tuner, X_test, Y_test, feature_names, num_patients, test_groups)
        print("evaluation complete — rmse is: ", auc)

    if (model_name == "xgboost"): # (only for xgboost)

        # Manually log the AUC if evaluate()
        mlflow.log_metric("auc", auc)

        #log the model - save it
        mlflow.xgboost.log_model(model, "model")

        #register it
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "xgboost-classifier")

        # Promote the best version to Production
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions("name='xgboost-classifier'")
        best = max(versions, key=lambda v: client.get_metric_history(v.run_id, "auc")[-1].value)
        client.transition_model_version_stage(
            name="xgboost-classifier",
            version=best.version,
            stage="Production"
        )

if __name__ == "__main__":
    main()