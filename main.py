from data_standardization import prepare_data
from model_training import train_xgboost
from evaluation import evaluate


def main():
    X_train, X_test, Y_train, Y_test = prepare_data("merged_combined_samples_data.xlsx")

    model = train_xgboost(X_train, Y_train)

    auc = evaluate(model, X_test, Y_test)

    print(f"Pipeline complete — AUC: {auc:.4f}")


if __name__ == "__main__":
    main()