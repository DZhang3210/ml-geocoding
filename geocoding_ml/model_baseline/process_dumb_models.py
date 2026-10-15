from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC                       # kept for a later return (see run_against_models)
from sklearn.preprocessing import StandardScaler  # only used by the SVC block
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Fit each classifier on train features, predict on val features.
# Returns {model_name: [preds, probs]}.
def run_against_models(train_df, train_label, val_df):
    ret = {}

    model = LogisticRegression()
    model.fit(train_df, train_label)
    preds = model.predict(val_df)
    probs = model.predict_proba(val_df)[:,1]
    ret["LogisticRegression"] = [preds, probs]

    # RBF SVC dropped for Overture (2026-10-11): training cost ~quadratic in ~200k rows, beyond free-tier compute.
    # Options if brought back: LinearSVC, or SVC on a ~20k subsample.
    # scaler = StandardScaler().fit(train_df)         # fit on train only
    # svc = SVC()                                    # default kernel="rbf"
    # svc.fit(scaler.transform(train_df), train_label)
    # preds = svc.predict(scaler.transform(val_df))
    # probs = svc.decision_function(scaler.transform(val_df))
    # ret["SVC"] = [preds, probs]

    rf = RandomForestClassifier(random_state=0)    # n_estimators=100 by default
    rf.fit(train_df, train_label)
    preds = rf.predict(val_df)
    probs = rf.predict_proba(val_df)[:, 1]
    rf.feature_importances_  # note: not stored or displayed here
    ret["RandomForestClassifier"] = [preds, probs]

    xgb = XGBClassifier(random_state=0)            # sklearn-style wrapper; you know these hyperparameters from the side lab
    xgb.fit(train_df, train_label)
    preds = xgb.predict(val_df)
    probs = xgb.predict_proba(val_df)[:, 1]
    ret["XGBClassifier"] = [preds, probs]

    return ret



