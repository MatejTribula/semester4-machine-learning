from data_standardization import prepare_data
from model_training import train_xgboost
from evaluation import evaluate
import mlflow


def main():
    mlflow.xgboost.autolog()  # Must be called before training

    X_train, X_test, Y_train, Y_test = prepare_data("merged_combined_samples_data.xlsx")

    with mlflow.start_run():  # Wrap in a run context
        model = train_xgboost(X_train, Y_train)
        auc = evaluate(model, X_test, Y_test)

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