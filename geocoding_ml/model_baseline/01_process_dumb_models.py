from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC                       # kept for a later return (see run_against_models)
from sklearn.preprocessing import StandardScaler  # only used by the SVC block
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Fit each classifier on train features, predict on val and test features.
# Returns {model_name: {"val": [preds, probs], "test": [preds, probs]}}.
def _predict_val_test(model, val_df, test_df):
    return {
        split: [model.predict(X), model.predict_proba(X)[:, 1]]
        for split, X in [("val", val_df), ("test", test_df)]
    }


def run_against_models(train_df, train_label, val_df, test_df):
    ret = {}

    model = LogisticRegression()
    model.fit(train_df, train_label)
    ret["LogisticRegression"] = _predict_val_test(model, val_df, test_df)

    # RBF SVC dropped for Overture (2026-10-11): training cost ~quadratic in ~200k rows, beyond free-tier compute.
    # Options if brought back: LinearSVC, or SVC on a ~20k subsample.
    # scaler = StandardScaler().fit(train_df)         # fit on train only
    # svc = SVC()                                    # default kernel="rbf"
    # svc.fit(scaler.transform(train_df), train_label)
    # preds = svc.predict(scaler.transform(val_df))
    # probs = svc.decision_function(scaler.transform(val_df))
    # ret["SVC"] = [preds, probs]   # (would need the same {"val", "test"} shape; SVC has no predict_proba here)

    rf = RandomForestClassifier(random_state=0)    # n_estimators=100 by default
    rf.fit(train_df, train_label)
    rf.feature_importances_  # note: not stored or displayed here
    ret["RandomForestClassifier"] = _predict_val_test(rf, val_df, test_df)

    xgb = XGBClassifier(random_state=0)            # sklearn-style wrapper; you know these hyperparameters from the side lab
    xgb.fit(train_df, train_label)
    ret["XGBClassifier"] = _predict_val_test(xgb, val_df, test_df)

    return ret
