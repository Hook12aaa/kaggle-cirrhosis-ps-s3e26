"""Shared training harness for all experiment worktrees.

Every worktree's 10 modules delegate here so model logic is identical and
audited once. A worktree supplies a CONFIG dict; this module loads data,
applies the locked features.py, builds the estimator for the requested
model family, runs stratified K-fold to produce out-of-fold probabilities,
refits on the full training set, and writes metrics plus the test submission.

Class order is fixed to ['C', 'CL', 'D'] so probability column 0/1/2 maps
directly to submission columns Status_C / Status_CL / Status_D.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TRAINER_DIR = "/Users/hook/Documents/coding/python/kaggle/ps-s3e26-cirrhosis/.auto-trainer"
DATA_DIR = "/Users/hook/Documents/coding/python/kaggle/ps-s3e26-cirrhosis"
CLASSES = ["C", "CL", "D"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
RAW_CATEGORICAL = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
LINEAR_FAMILIES = {"linear", "knn", "svm", "neural_net"}

import sys
sys.path.insert(0, TRAINER_DIR)
from features import engineer_features  # noqa: E402


def _prepare_frame(df):
    out = engineer_features(df)
    drop = [c for c in ["id", "Status"] if c in out.columns]
    X = out.drop(columns=drop)
    for c in RAW_CATEGORICAL:
        X[c] = X[c].astype("object")
    return X


def load_data(config):
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y = train["Status"].map(CLASS_TO_IDX).to_numpy()
    test_ids = test["id"].to_numpy()
    X = _prepare_frame(train)

    X_orig, y_orig = None, None
    if config.get("use_original", False):
        orig = pd.read_csv(os.path.join(TRAINER_DIR, "original_cirrhosis.csv"))
        orig = orig[orig["Status"].isin(CLASSES)].copy()
        y_orig = orig["Status"].map(CLASS_TO_IDX).to_numpy()
        orig = orig.rename(columns={"ID": "id"})
        X_orig = _prepare_frame(orig)
        X_orig = X_orig.reindex(columns=X.columns)

    X_test = _prepare_frame(test).reindex(columns=X.columns)
    cat_features = [c for c in RAW_CATEGORICAL if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features]
    meta = {"cat_features": cat_features, "num_features": num_features}
    return X, y, X_orig, y_orig, X_test, test_ids, meta


def _build_preprocessor(arch_class, meta):
    num, cat = meta["num_features"], meta["cat_features"]
    if arch_class in LINEAR_FAMILIES:
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler())])
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("oh", OneHotEncoder(handle_unknown="ignore"))])
    else:
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                    unknown_value=-1))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])


def _build_classifier(config):
    mt = config["model_type"]
    hp = config.get("hyperparameters", {})
    seed = config.get("random_seed", 42)
    if mt == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=hp.get("C", 1.0), max_iter=hp.get("max_iter", 2000),
                                  random_state=seed)
    if mt == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 400),
            max_depth=hp.get("max_depth", None),
            min_samples_leaf=hp.get("min_samples_leaf", 2),
            n_jobs=-1, random_state=seed)
    if mt == "hist_gbt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=hp.get("learning_rate", 0.03),
            max_iter=hp.get("max_iter", 1000),
            max_depth=hp.get("max_depth", None),
            max_leaf_nodes=hp.get("max_leaf_nodes", 31),
            l2_regularization=hp.get("l2_regularization", 1.0),
            early_stopping=hp.get("early_stopping", True),
            validation_fraction=hp.get("validation_fraction", 0.1),
            n_iter_no_change=hp.get("n_iter_no_change", 30),
            random_state=seed)
    if mt == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=hp.get("n_estimators", 600),
            learning_rate=hp.get("learning_rate", 0.03),
            max_depth=hp.get("max_depth", 4),
            subsample=hp.get("subsample", 0.8),
            colsample_bytree=hp.get("colsample_bytree", 0.8),
            reg_lambda=hp.get("reg_lambda", 1.0),
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", tree_method="hist",
            n_jobs=-1, random_state=seed)
    if mt == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=hp.get("n_estimators", 600),
            learning_rate=hp.get("learning_rate", 0.03),
            num_leaves=hp.get("num_leaves", 31),
            max_depth=hp.get("max_depth", -1),
            subsample=hp.get("subsample", 0.8),
            colsample_bytree=hp.get("colsample_bytree", 0.8),
            reg_lambda=hp.get("reg_lambda", 1.0),
            objective="multiclass", num_class=3,
            n_jobs=-1, random_state=seed, verbose=-1)
    if mt == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(
            n_neighbors=hp.get("n_neighbors", 25),
            weights=hp.get("weights", "distance"))
    if mt == "svm":
        from sklearn.svm import SVC
        return SVC(C=hp.get("C", 1.0), kernel=hp.get("kernel", "rbf"),
                   gamma=hp.get("gamma", "scale"), probability=True,
                   random_state=seed)
    if mt == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=tuple(hp.get("hidden_layer_sizes", [128, 64])),
            alpha=hp.get("alpha", 1e-3),
            learning_rate_init=hp.get("learning_rate_init", 1e-3),
            max_iter=hp.get("max_iter", 400),
            early_stopping=True, random_state=seed)
    raise ValueError(f"unknown model_type {mt}")


def build_estimator(config, meta):
    arch = config["architecture_class"]
    if config["model_type"] == "catboost":
        return None
    pre = _build_preprocessor(arch, meta)
    clf = _build_classifier(config)
    return Pipeline([("pre", pre), ("clf", clf)])


def _proba_aligned(est, X):
    proba = est.predict_proba(X)
    classes = list(est.classes_)
    out = np.zeros((X.shape[0], 3))
    for j, cls in enumerate(classes):
        out[:, int(cls)] = proba[:, j]
    return out


def _catboost_model(config):
    from catboost import CatBoostClassifier
    hp = config.get("hyperparameters", {})
    return CatBoostClassifier(
        iterations=hp.get("iterations", 800),
        learning_rate=hp.get("learning_rate", 0.03),
        depth=hp.get("depth", 6),
        l2_leaf_reg=hp.get("l2_leaf_reg", 3.0),
        loss_function="MultiClass", random_seed=config.get("random_seed", 42),
        verbose=False, allow_writing_files=False)


def _catboost_prep(X, meta):
    X = X.copy()
    for c in meta["cat_features"]:
        X[c] = X[c].astype("object").where(X[c].notna(), "missing").astype(str)
    return X


def _fit_predict_catboost(config, meta, X_tr, y_tr, X_va):
    from catboost import Pool
    model = _catboost_model(config)
    Xtr = _catboost_prep(X_tr, meta)
    Xva = _catboost_prep(X_va, meta)
    model.fit(Pool(Xtr, y_tr, cat_features=meta["cat_features"]))
    proba = model.predict_proba(Xva)
    classes = list(model.classes_)
    out = np.zeros((X_va.shape[0], 3))
    for j, cls in enumerate(classes):
        out[:, int(cls)] = proba[:, j]
    return out, model


def run_cv(config, X, y, X_orig, y_orig, meta):
    k = config.get("cv_folds", 5)
    seed = config.get("random_seed", 42)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 3))
    fold_losses = []
    is_cat = config["model_type"] == "catboost"
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        if X_orig is not None:
            X_tr = pd.concat([X_tr, X_orig], ignore_index=True)
            y_tr = np.concatenate([y_tr, y_orig])
        if is_cat:
            p, _ = _fit_predict_catboost(config, meta, X_tr, y_tr, X_va)
        else:
            est = build_estimator(config, meta)
            est.fit(X_tr, y_tr)
            p = _proba_aligned(est, X_va)
        oof[va_idx] = p
        fold_losses.append(float(log_loss(y_va, p, labels=[0, 1, 2])))
    return oof, fold_losses


def count_params(config, meta, X, y, X_orig, y_orig):
    X_full, y_full = X, y
    if X_orig is not None:
        X_full = pd.concat([X, X_orig], ignore_index=True)
        y_full = np.concatenate([y, y_orig])
    mt = config["model_type"]
    if mt == "catboost":
        m = _catboost_model(config)
        from catboost import Pool
        Xf = _catboost_prep(X_full, meta)
        m.fit(Pool(Xf, y_full, cat_features=meta["cat_features"]))
        return int(m.tree_count_ * (2 ** config.get("hyperparameters", {}).get("depth", 6))), m
    est = build_estimator(config, meta)
    est.fit(X_full, y_full)
    clf = est.named_steps["clf"]
    n_feat = est.named_steps["pre"].transform(X_full[:1]).shape[1]
    if mt == "logistic_regression":
        p = int(clf.coef_.size + clf.intercept_.size)
    elif mt == "random_forest":
        p = int(sum(t.tree_.node_count for t in clf.estimators_))
    elif mt == "hist_gbt":
        p = int(sum(pred.nodes.shape[0] for it in clf._predictors for pred in it))
    elif mt == "xgboost":
        p = int(len(clf.get_booster().trees_to_dataframe()))
    elif mt == "lightgbm":
        p = int(len(clf.booster_.trees_to_dataframe()))
    elif mt == "knn":
        p = int(X_full.shape[0] * n_feat)
    elif mt == "svm":
        p = int(clf.n_support_.sum() * n_feat)
    elif mt == "mlp":
        p = int(sum(c.size for c in clf.coefs_) + sum(b.size for b in clf.intercepts_))
    else:
        raise ValueError(f"no param counter for {mt}")
    return p, est


def fit_full_and_predict(config, meta, X, y, X_orig, y_orig, X_test):
    X_full, y_full = X, y
    if X_orig is not None:
        X_full = pd.concat([X, X_orig], ignore_index=True)
        y_full = np.concatenate([y, y_orig])
    if config["model_type"] == "catboost":
        from catboost import Pool
        model = _catboost_model(config)
        Xf = _catboost_prep(X_full, meta)
        model.fit(Pool(Xf, y_full, cat_features=meta["cat_features"]))
        Xt = _catboost_prep(X_test, meta)
        proba = model.predict_proba(Xt)
        classes = list(model.classes_)
        out = np.zeros((X_test.shape[0], 3))
        for j, cls in enumerate(classes):
            out[:, int(cls)] = proba[:, j]
        train_proba = np.zeros((X_full.shape[0], 3))
        tp = model.predict_proba(Xf)
        for j, cls in enumerate(classes):
            train_proba[:, int(cls)] = tp[:, j]
        return out, train_proba, y_full
    est = build_estimator(config, meta)
    est.fit(X_full, y_full)
    test_proba = _proba_aligned(est, X_test)
    train_proba = _proba_aligned(est, X_full)
    return test_proba, train_proba, y_full


def run_experiment(config, worktree):
    X, y, X_orig, y_orig, X_test, test_ids, meta = load_data(config)
    oof, fold_losses = run_cv(config, X, y, X_orig, y_orig, meta)
    oof_loss = float(log_loss(y, oof, labels=[0, 1, 2]))
    oof_acc = float(accuracy_score(y, np.argmax(oof, axis=1)))

    n_params, _ = count_params(config, meta, X, y, X_orig, y_orig)
    test_proba, train_proba, y_full = fit_full_and_predict(
        config, meta, X, y, X_orig, y_orig, X_test)
    train_loss = float(log_loss(y_full, train_proba, labels=[0, 1, 2]))

    np.save(os.path.join(worktree, "val_predictions.npy"), oof)
    np.save(os.path.join(worktree, "val_labels.npy"), y)
    np.save(os.path.join(worktree, "test_predictions.npy"), test_proba)

    sub = pd.DataFrame({"id": test_ids,
                        "Status_C": test_proba[:, 0],
                        "Status_CL": test_proba[:, 1],
                        "Status_D": test_proba[:, 2]})
    sub.to_csv(os.path.join(worktree, "submission.csv"), index=False)

    metrics = {
        "log_loss": oof_loss,
        "accuracy": oof_acc,
        "train_log_loss": train_loss,
        "fold_log_losses": fold_losses,
        "n_features": int(meta["num_features"].__len__() + meta["cat_features"].__len__()),
        "trainable_params": int(n_params),
        "n_epochs_run": int(config.get("cv_folds", 5)),
    }
    with open(os.path.join(worktree, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
