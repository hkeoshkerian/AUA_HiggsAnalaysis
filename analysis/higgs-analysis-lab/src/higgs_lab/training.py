"""Five-model OOF training migrated from the supplied analysis."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from .features import validate_frame

SUPPORTED_MODELS = ("xgboost", "lightgbm", "random_forest", "mlp",
                    "logistic_regression", "gaussian_nb", "qda")

@dataclass
class TrainingResult:
    predictions: pd.DataFrame
    fold_metrics: list
    oof_auc: float
    models: list

def _balanced_weights(y, w):
    result = np.asarray(w, dtype=float).copy()
    signal, background = result[y == 1].sum(), result[y == 0].sum()
    if signal <= 0 or background <= 0:
        raise ValueError("Each training class must have positive total weight")
    result[y == 1] *= background / signal
    return result

def _scaled_inputs(model_name, X_train, X_test, X_data, X_val=None):
    if model_name == "random_forest":
        return None, X_train, X_test, X_data, X_val
    scaler = StandardScaler().fit(X_train)
    return (scaler, scaler.transform(X_train), scaler.transform(X_test),
            scaler.transform(X_data) if len(X_data) else X_data.copy(),
            scaler.transform(X_val) if X_val is not None else None)

def _fit_fold(name, X_full, y_full, w_full, X_test, X_data, seed, fold):
    needs_val = name in {"xgboost", "lightgbm", "mlp"}
    if needs_val:
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X_full, y_full, w_full, test_size=.20, stratify=y_full,
            random_state=seed)
    else:
        X_train, y_train, w_train = X_full, y_full, w_full
        X_val = y_val = w_val = None

    scaler, X_train, X_test, X_data, X_val = _scaled_inputs(
        name, X_train, X_test, X_data, X_val)
    details = {}

    if name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError('Install XGBoost with: python -m pip install ".[boosters]"') from exc
        spw = w_train[y_train == 0].sum() / w_train[y_train == 1].sum()
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=.05,
            subsample=.8, colsample_bytree=.8, min_child_weight=1,
            gamma=0, reg_lambda=1, reg_alpha=0, scale_pos_weight=spw,
            eval_metric="auc", random_state=seed, n_jobs=-1,
            tree_method="hist", early_stopping_rounds=20)
        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_val, y_val)], sample_weight_eval_set=[w_val],
                  verbose=False)
        details["best_iteration"] = getattr(model, "best_iteration", None)

    elif name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError('Install LightGBM with: python -m pip install ".[boosters]"') from exc
        spw = w_train[y_train == 0].sum() / w_train[y_train == 1].sum()
        model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=4, num_leaves=15, learning_rate=.05,
            subsample=.8, subsample_freq=1, colsample_bytree=.8,
            min_child_samples=20, reg_lambda=1., reg_alpha=0.,
            scale_pos_weight=spw, objective="binary", metric="auc",
            random_state=seed, n_jobs=-1, verbose=-1)
        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_val, y_val)], eval_sample_weight=[w_val],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
        details["best_iteration"] = getattr(model, "best_iteration_", None)

    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=20,
            n_jobs=-1, random_state=seed)
        model.fit(X_train, y_train,
                  sample_weight=_balanced_weights(y_train, w_train))

    elif name == "logistic_regression":
        model = LogisticRegression(C=100., max_iter=2000, solver="lbfgs",
                                   random_state=seed)
        model.fit(X_train, y_train,
                  sample_weight=_balanced_weights(y_train, w_train))

    elif name == "gaussian_nb":
        model = GaussianNB()
        model.fit(X_train, y_train,
                  sample_weight=_balanced_weights(y_train, w_train))

    elif name == "qda":
        # QDA has no sample_weight API. Preserve the source's weighted-bootstrap
        # strategy while making it deterministic per fold.
        balanced = _balanced_weights(y_train, w_train)
        probabilities = balanced / balanced.sum()
        rng = np.random.default_rng(seed + fold)
        sampled = rng.choice(len(y_train), size=len(y_train), replace=True,
                             p=probabilities)
        if len(np.unique(y_train[sampled])) != 2:
            raise ValueError("QDA bootstrap did not contain both classes")
        model = QuadraticDiscriminantAnalysis(reg_param=.1)
        model.fit(X_train[sampled], y_train[sampled])

    elif name == "mlp":
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError('Install TensorFlow with: python -m pip install ".[neural]"') from exc
        tf.keras.utils.set_random_seed(seed + fold)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss="binary_crossentropy")
        spw = w_train[y_train == 0].sum() / w_train[y_train == 1].sum()
        w_train_bal = w_train.copy(); w_train_bal[y_train == 1] *= spw
        w_val_bal = w_val.copy(); w_val_bal[y_val == 1] *= spw
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True,
            verbose=0)
        history = model.fit(
            X_train, y_train, sample_weight=w_train_bal,
            validation_data=(X_val, y_val, w_val_bal), epochs=100,
            batch_size=256, callbacks=[callback], verbose=0)
        details["epochs"] = len(history.history["loss"])
    else:
        raise ValueError(f"Unsupported model: {name}")

    if name == "mlp":
        test_scores = model.predict(X_test, batch_size=4096, verbose=0).ravel()
        data_scores = (model.predict(X_data, batch_size=4096, verbose=0).ravel()
                       if len(X_data) else np.empty(0))
    else:
        test_scores = model.predict_proba(X_test)[:, 1]
        data_scores = (model.predict_proba(X_data)[:, 1]
                       if len(X_data) else np.empty(0))
    return test_scores, data_scores, {"scaler": scaler, "model": model}, details

def train_models(frame, config):
    """Return OOF MC scores and mean fold-model scores for observed data."""
    t = config.training
    validate_frame(frame, t.features)
    if t.model not in SUPPORTED_MODELS:
        raise ValueError(f"Choose one of: {', '.join(SUPPORTED_MODELS)}")
    mc = frame[frame.role != "data"].copy()
    data = frame[frame.role == "data"].copy()
    X = mc[list(t.features)].to_numpy(float)
    y = mc.label.to_numpy(int)
    w = mc.weight.to_numpy(float)
    X_data = data[list(t.features)].to_numpy(float)
    if (w < 0).any():
        raise ValueError("Negative MC weights need a reviewed model-specific policy")
    if any(np.count_nonzero((y == label) & (w > 0)) < t.folds for label in (0, 1)):
        raise ValueError("Each MC class needs at least folds positive-weight events")

    oof = np.full(len(mc), np.nan)
    fold_ids = np.full(len(mc), -1, int)
    data_per_fold = np.zeros((t.folds, len(data)))
    models, metrics = [], []
    cv = StratifiedKFold(n_splits=t.folds, shuffle=True, random_state=t.seed)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        if any(w[train_idx][y[train_idx] == label].sum() <= 0 or
               w[test_idx][y[test_idx] == label].sum() <= 0
               for label in (0, 1)):
            raise ValueError("Every fold needs positive total weight in each class")
        scores, data_scores, pipeline, details = _fit_fold(
            t.model, X[train_idx], y[train_idx], w[train_idx], X[test_idx],
            X_data, t.seed, fold)
        oof[test_idx] = scores
        fold_ids[test_idx] = fold
        if len(data): data_per_fold[fold] = data_scores
        metrics.append({"fold": fold, "n_train_full": len(train_idx),
                        "n_test": len(test_idx),
                        "weighted_auc": float(roc_auc_score(
                            y[test_idx], scores, sample_weight=w[test_idx])),
                        **details})
        # Keep the original package API: (scaler, fitted model) per fold.
        models.append((pipeline["scaler"], pipeline["model"]))

    mc["score"], mc["fold"], mc["score_kind"] = oof, fold_ids, "out_of_fold"
    data["score"] = data_per_fold.mean(axis=0) if len(data) else np.empty(0)
    data["fold"], data["score_kind"] = -1, "fold_ensemble_mean"
    predictions = pd.concat([mc, data], ignore_index=True)
    return TrainingResult(predictions, metrics,
                          float(roc_auc_score(y, oof, sample_weight=w)), models)
