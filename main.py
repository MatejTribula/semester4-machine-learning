from data_standardization import prepare_data
from models.xgb import train_xgboost
from models.lgbm import train_lgbm
from models.randomforest import train_rf
from evaluation import evaluate


def main():

    X_train, X_test, Y_train, Y_test, train_groups, test_groups, feature_names, num_patients = prepare_data("merged_combined_samples_data.xlsx")

    model, tuner_model = train_lgbm(X_train, Y_train, train_groups)

    # a bit of a rudimentary metric
    rmse = evaluate(model, tuner_model, X_test, Y_test, feature_names, num_patients, test_groups)

    print("Pipeline complete — rmse is: ", rmse)


if __name__ == "__main__":
    main()