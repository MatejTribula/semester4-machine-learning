from sklearn.metrics import classification_report, roc_auc_score


def evaluate(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    Y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(Y_test, Y_pred))

    auc = roc_auc_score(Y_test, Y_prob)
    print("AUC:", auc)

    return auc