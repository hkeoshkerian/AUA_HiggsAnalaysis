"""Five-model OOF training migrated from the supplied analysis."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
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
    tuning: dict | None = None

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

def _xgboost_estimator(xgb, seed, max_depth, min_child_weight, scale_pos_weight):
    return xgb.XGBClassifier(
        n_estimators=500, max_depth=max_depth, learning_rate=.05,
        subsample=.8, colsample_bytree=.8,
        min_child_weight=min_child_weight, gamma=0,
        reg_lambda=1, reg_alpha=0, scale_pos_weight=scale_pos_weight,
        eval_metric="auc", random_state=seed, n_jobs=-1,
        tree_method="hist", early_stopping_rounds=20)


def _lightgbm_estimator(lgb, seed, max_depth, num_leaves,
                        min_child_samples, scale_pos_weight):
    return lgb.LGBMClassifier(
        n_estimators=500, max_depth=max_depth, num_leaves=num_leaves,
        learning_rate=.05, subsample=.8, subsample_freq=1,
        colsample_bytree=.8, min_child_samples=min_child_samples,
        reg_lambda=1., reg_alpha=0., scale_pos_weight=scale_pos_weight,
        objective="binary", metric="auc", random_state=seed,
        n_jobs=-1, verbose=-1)


def _random_forest_estimator(seed, max_depth, min_samples_leaf):
    return RandomForestClassifier(
        n_estimators=300, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features="sqrt",
        bootstrap=True, n_jobs=-1, random_state=seed)


def _logistic_estimator(seed, c_value):
    return LogisticRegression(
        C=c_value, max_iter=2000, solver="lbfgs", random_state=seed)


def _build_mlp(tf, input_dim, architecture, learning_rate, l2_value, dropout):
    regularizer = (tf.keras.regularizers.l2(l2_value)
                   if l2_value > 0 else None)
    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    for units in architecture:
        layers.append(tf.keras.layers.Dense(
            units, activation="relu", kernel_regularizer=regularizer))
        if dropout > 0:
            layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy")
    return model


def _fit_weighted_qda(X, y, w, seed, reg_param):
    """Fit QDA on one deterministic physical-weight bootstrap sample."""
    probabilities = w / w.sum()
    for attempt in range(20):
        rng = np.random.default_rng(seed + attempt)
        sampled = rng.choice(
            len(y), size=len(y), replace=True, p=probabilities)
        if set(np.unique(y[sampled])) == {0, 1}:
            scaler = StandardScaler().fit(X[sampled])
            model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
            model.fit(scaler.transform(X[sampled]), y[sampled])
            return scaler, model
    raise ValueError("QDA weighted bootstrap did not contain both classes")

def _checked_group_splits(X, y, groups, folds, seed, context):
    """Return deterministic stratified splits without sharing groups.

    Highly unequal physics-source sizes can make one shuffled assignment omit a
    class from a test fold even when a valid assignment exists. Search a fixed,
    reproducible sequence of seeds and accept only a complete valid partition.
    """
    groups = np.asarray(groups)
    for label in (0, 1):
        count = np.unique(groups[y == label]).size
        if count < folds:
            raise ValueError(
                f"{context} needs at least {folds} distinct groups in class {label}; "
                f"found {count}. Prepare data with per-source group identifiers or "
                "reduce the number of folds.")
    # MC source groups are class-pure. Assign their largest sources first and
    # balance each class independently; this guarantees both classes in every
    # fold whenever each class has at least `folds` groups. StratifiedGroupKFold
    # can otherwise leave an empty class when source sizes differ by orders of
    # magnitude (as they do for ggH versus rare production modes).
    unique_groups = np.unique(groups)
    group_labels = {group: np.unique(y[groups == group]) for group in unique_groups}
    if all(labels.size == 1 for labels in group_labels.values()):
        rng = np.random.default_rng(seed)
        fold_groups = [set() for _ in range(folds)]
        fold_priority = rng.permutation(folds)
        for label in (0, 1):
            candidates = []
            for group in unique_groups:
                if group_labels[group][0] == label:
                    candidates.append((group, int(np.count_nonzero(groups == group))))
            candidates.sort(key=lambda item: (-item[1], str(item[0])))
            for index, (group, count) in enumerate(candidates):
                # Round-robin guarantees comparable numbers of independent
                # sources per fold. Event counts cannot be balanced when a
                # single indivisible source dominates the class.
                fold = int(fold_priority[index % folds])
                fold_groups[fold].add(group)
        splits = []
        for held_out in fold_groups:
            test_idx = np.flatnonzero(np.isin(groups, list(held_out)))
            train_idx = np.flatnonzero(~np.isin(groups, list(held_out)))
            splits.append((train_idx, test_idx))
        return splits

    # General fallback for any future groups containing both classes.
    for attempt in range(100):
        candidate_seed = (seed + attempt) % 2**32
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=candidate_seed)
        splits = list(splitter.split(X, y, groups))
        valid = True
        for train_idx, test_idx in splits:
            overlap = set(groups[train_idx]) & set(groups[test_idx])
            if overlap:
                raise RuntimeError(
                    f"{context} leaked groups across a fold: {sorted(overlap)[:3]}")
            if (set(np.unique(y[train_idx])) != {0, 1} or
                    set(np.unique(y[test_idx])) != {0, 1}):
                valid = False
                break
        if valid:
            return splits
    raise ValueError(
        f"{context} could not construct {folds} zero-overlap folds containing "
        "both classes after 100 deterministic assignments. The source groups "
        "are too sparse or imbalanced for this split.")


def _validation_indices(X, y, groups, seed):
    """Approximately 20% group-preserving validation split."""
    groups = np.asarray(groups)
    validation_groups = set()
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        label_groups, counts = np.unique(groups[y == label], return_counts=True)
        if label_groups.size < 2:
            raise ValueError(
                "Group-aware early stopping needs at least two groups per class")
        # Pick a deterministic subset close to 20% of this class. Starting
        # from the best single group guarantees that validation has the class;
        # greedily add groups only while the target match improves.
        order = rng.permutation(label_groups.size)
        target = .20 * counts.sum()
        best = min(order, key=lambda i: (abs(counts[i] - target), str(label_groups[i])))
        chosen = [best]
        total = counts[best]
        remaining = [i for i in order if i != best]
        while len(chosen) < label_groups.size - 1 and remaining:
            candidate = min(
                remaining,
                key=lambda i: (abs(total + counts[i] - target), str(label_groups[i])))
            if abs(total + counts[candidate] - target) >= abs(total - target):
                break
            chosen.append(candidate)
            remaining.remove(candidate)
            total += counts[candidate]
        validation_groups.update(label_groups[chosen])
    validation = np.flatnonzero(np.isin(groups, list(validation_groups)))
    training = np.flatnonzero(~np.isin(groups, list(validation_groups)))
    if set(np.unique(y[training])) != {0, 1} or set(np.unique(y[validation])) != {0, 1}:
        raise ValueError("Group-aware early-stopping split lacks a class")
    if set(groups[training]) & set(groups[validation]):
        raise RuntimeError("Early-stopping split leaked a group")
    return training, validation


def _outer_splits(X, y, groups, folds, seed, group_aware):
    if group_aware:
        return _checked_group_splits(X, y, groups, folds, seed, "Outer CV")
    return list(StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed).split(X, y))


def _process_stratified_splits(y, samples, folds, seed):
    """Frozen event folds balanced simultaneously by class and MC process."""
    strata = np.char.add(np.char.add(y.astype(str), ":"), samples.astype(str))
    counts = pd.Series(strata).value_counts()
    if (counts < folds).any():
        sparse = counts[counts < folds].to_dict()
        raise ValueError(
            f"Each class/process stratum needs at least {folds} events; "
            f"too-small strata: {sparse}")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), strata))


def _process_validation_indices(y, samples, seed):
    """Training-only validation split, stratified by class and MC process."""
    strata = np.char.add(np.char.add(y.astype(str), ":"), samples.astype(str))
    indices = np.arange(len(y))
    try:
        fit, validation = train_test_split(
            indices, test_size=.20, stratify=strata, random_state=seed)
    except ValueError as exc:
        raise ValueError(
            "Training-side calibration split needs at least two events in "
            "every class/process stratum") from exc
    return fit, validation


def _data_fold_ids(size, folds, seed):
    """Deterministic, balanced fold assignment for otherwise unlabeled data."""
    order = np.random.default_rng(seed).permutation(size)
    assigned = np.empty(size, dtype=int)
    assigned[order] = np.arange(size) % folds
    return assigned


def _weighted_signal_cdf(reference_scores, reference_weights, values):
    """Map raw scores to a fold-local weighted signal percentile."""
    scores = np.asarray(reference_scores, float)
    weights = np.asarray(reference_weights, float)
    values = np.asarray(values, float)
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    if not valid.any() or weights[valid].sum() <= 0:
        raise ValueError("Fold calibration needs positive-weight validation signal")
    order = np.argsort(scores[valid], kind="mergesort")
    sorted_scores = scores[valid][order]
    cumulative = np.cumsum(weights[valid][order])
    cumulative /= cumulative[-1]
    positions = np.searchsorted(sorted_scores, values, side="right") - 1
    result = np.zeros(len(values), dtype=float)
    present = positions >= 0
    result[present] = cumulative[positions[present]]
    return result


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    order = np.argsort(values[valid], kind="mergesort")
    ordered_values = values[valid][order]
    cumulative = np.cumsum(weights[valid][order])
    if not len(ordered_values) or cumulative[-1] <= 0:
        raise ValueError("Weighted quantile needs positive finite weights")
    index = np.searchsorted(cumulative, quantile*cumulative[-1], side="left")
    return float(ordered_values[min(index, len(ordered_values)-1)])


def _tune_xgboost(X, y, w, groups, seed, folds, group_aware):
    """Tune only within one outer training fold."""
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError('Install XGBoost with: python -m pip install ".[boosters]"') from exc
    rows = []
    if group_aware:
        cv = _checked_group_splits(X, y, groups, folds, seed, "Inner tuning CV")
    else:
        cv = list(StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed).split(X, y))
    print(f"Tuning XGBoost: 9 combinations x {folds} folds", flush=True)
    for max_depth in (3, 4, 5):
        for min_child_weight in (1, 5, 10):
            aucs = []
            for inner_fold, (train_idx, test_idx) in enumerate(cv):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                w_train, w_test = w[train_idx], w[test_idx]
                if group_aware:
                    inner_groups = groups[train_idx]
                    fit_idx, val_idx = _validation_indices(
                        X_train, y_train, inner_groups, seed + inner_fold)
                    X_train, X_val = X_train[fit_idx], X_train[val_idx]
                    y_train, y_val = y_train[fit_idx], y_train[val_idx]
                    w_train, w_val = w_train[fit_idx], w_train[val_idx]
                else:
                    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
                        X_train, y_train, w_train, test_size=.20,
                        random_state=seed, stratify=y_train)
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_val = scaler.transform(X_val)
                X_test = scaler.transform(X_test)
                spw = w_train[y_train == 0].sum() / w_train[y_train == 1].sum()
                model = _xgboost_estimator(
                    xgb, seed, max_depth, min_child_weight, spw)
                model.fit(X_train, y_train, sample_weight=w_train,
                          eval_set=[(X_val, y_val)],
                          sample_weight_eval_set=[w_val], verbose=False)
                scores = model.predict_proba(X_test)[:, 1]
                aucs.append(roc_auc_score(y_test, scores, sample_weight=w_test))
            rows.append({"max_depth": max_depth,
                         "min_child_weight": min_child_weight,
                         "mean_auc": float(np.mean(aucs)),
                         "std_auc": float(np.std(aucs))})
            print(f"  max_depth={max_depth}, min_child_weight={min_child_weight}: "
                  f"AUC={np.mean(aucs):.4f} +/- {np.std(aucs):.4f}", flush=True)
    best = max(rows, key=lambda row: row["mean_auc"])
    print(f"Selected max_depth={best['max_depth']}, "
          f"min_child_weight={best['min_child_weight']}", flush=True)
    return {"grid": rows,
            "best_max_depth": best["max_depth"],
            "best_min_child_weight": best["min_child_weight"],
            "best_mean_auc": best["mean_auc"],
            "best_std_auc": best["std_auc"]}


def _tune_lightgbm(X, y, w, groups, seed, folds, group_aware):
    """Select LightGBM complexity using only one outer training fold."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            'Install LightGBM with: python -m pip install ".[boosters]"') from exc
    if group_aware:
        cv = _checked_group_splits(
            X, y, groups, folds, seed, "LightGBM inner tuning CV")
    else:
        cv = list(StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed).split(X, y))

    # Only structurally sensible depth/leaf pairs are considered. Searching
    # all independent combinations would include num_leaves values that a
    # shallow tree cannot use and would waste hundreds of fits.
    structures = ((3, 7), (4, 7), (4, 15), (5, 15), (5, 31))
    combinations = [(depth, leaves, child)
                    for depth, leaves in structures
                    for child in (10, 20, 40)]
    print(f"Tuning LightGBM: {len(combinations)} combinations x {folds} folds",
          flush=True)
    rows = []
    for max_depth, num_leaves, min_child_samples in combinations:
        aucs, iterations = [], []
        for inner_fold, (train_idx, test_idx) in enumerate(cv):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train, w_test = w[train_idx], w[test_idx]
            if group_aware:
                fit_idx, val_idx = _validation_indices(
                    X_train, y_train, groups[train_idx], seed + inner_fold)
                X_train, X_val = X_train[fit_idx], X_train[val_idx]
                y_train, y_val = y_train[fit_idx], y_train[val_idx]
                w_train, w_val = w_train[fit_idx], w_train[val_idx]
            else:
                X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
                    X_train, y_train, w_train, test_size=.20,
                    random_state=seed, stratify=y_train)
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            spw = w_train[y_train == 0].sum() / w_train[y_train == 1].sum()
            model = _lightgbm_estimator(
                lgb, seed, max_depth, num_leaves, min_child_samples, spw)
            model.fit(
                X_train, y_train, sample_weight=w_train,
                eval_set=[(X_val, y_val)], eval_sample_weight=[w_val],
                callbacks=[lgb.early_stopping(20, verbose=False)])
            scores = model.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, scores, sample_weight=w_test))
            iterations.append(int(model.best_iteration_))
        row = {
            "max_depth": max_depth, "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "mean_best_iteration": float(np.mean(iterations)),
        }
        rows.append(row)
        print(f"  depth={max_depth}, leaves={num_leaves}, "
              f"min_child_samples={min_child_samples}: "
              f"AUC={row['mean_auc']:.4f} +/- {row['std_auc']:.4f}", flush=True)
    best = max(rows, key=lambda row: row["mean_auc"])
    print(f"Selected depth={best['max_depth']}, leaves={best['num_leaves']}, "
          f"min_child_samples={best['min_child_samples']}", flush=True)
    return {
        "grid": rows,
        "best_max_depth": best["max_depth"],
        "best_num_leaves": best["num_leaves"],
        "best_min_child_samples": best["min_child_samples"],
        "best_mean_auc": best["mean_auc"],
        "best_std_auc": best["std_auc"],
    }


def _tune_random_forest(X, y, w, groups, seed, folds, group_aware):
    """Select forest regularization using only one outer training fold."""
    if group_aware:
        cv = _checked_group_splits(
            X, y, groups, folds, seed, "Random Forest inner tuning CV")
    else:
        cv = list(StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed).split(X, y))
    combinations = [(depth, leaf)
                    for depth in (4, 5, 7)
                    for leaf in (5, 10, 20)]
    print(f"Tuning Random Forest: {len(combinations)} combinations x {folds} folds",
          flush=True)
    rows = []
    for max_depth, min_samples_leaf in combinations:
        aucs = []
        for inner_fold, (train_idx, test_idx) in enumerate(cv):
            model = _random_forest_estimator(
                seed + inner_fold, max_depth, min_samples_leaf)
            model.fit(
                X[train_idx], y[train_idx],
                sample_weight=_balanced_weights(y[train_idx], w[train_idx]))
            scores = model.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(
                y[test_idx], scores, sample_weight=w[test_idx]))
        row = {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": "sqrt",
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
        }
        rows.append(row)
        print(f"  depth={max_depth}, min_samples_leaf={min_samples_leaf}: "
              f"AUC={row['mean_auc']:.4f} +/- {row['std_auc']:.4f}", flush=True)
    best = max(rows, key=lambda row: row["mean_auc"])
    print(f"Selected depth={best['max_depth']}, "
          f"min_samples_leaf={best['min_samples_leaf']}", flush=True)
    return {
        "grid": rows,
        "best_max_depth": best["max_depth"],
        "best_min_samples_leaf": best["min_samples_leaf"],
        "best_max_features": best["max_features"],
        "best_mean_auc": best["mean_auc"],
        "best_std_auc": best["std_auc"],
    }


def _tune_logistic_regression(X, y, w, groups, seed, group_aware):
    """Paper-compatible six-point C scan inside one outer training fold."""
    inner_folds = 3
    if group_aware:
        cv = _checked_group_splits(
            X, y, groups, inner_folds, seed,
            "Logistic Regression inner tuning CV")
    else:
        cv = list(StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=seed).split(X, y))
    values = (1e-3, 1e-2, 1e-1, 1., 10., 100.)
    print(f"Tuning Logistic Regression: {len(values)} C values x "
          f"{inner_folds} folds", flush=True)
    rows = []
    for c_value in values:
        aucs = []
        for inner_fold, (train_idx, test_idx) in enumerate(cv):
            scaler = StandardScaler().fit(X[train_idx])
            X_train = scaler.transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            model = _logistic_estimator(seed + inner_fold, c_value)
            model.fit(
                X_train, y[train_idx],
                sample_weight=_balanced_weights(y[train_idx], w[train_idx]))
            scores = model.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(
                y[test_idx], scores, sample_weight=w[test_idx]))
        row = {"C": c_value, "mean_auc": float(np.mean(aucs)),
               "std_auc": float(np.std(aucs))}
        rows.append(row)
        print(f"  C={c_value:g}: AUC={row['mean_auc']:.4f} +/- "
              f"{row['std_auc']:.4f}", flush=True)
    best = max(rows, key=lambda row: row["mean_auc"])
    print(f"Selected C={best['C']:g}", flush=True)
    return {"grid": rows, "inner_folds": inner_folds,
            "best_C": best["C"], "best_mean_auc": best["mean_auc"],
            "best_std_auc": best["std_auc"]}


def _tune_mlp(X, y, w, groups, seed, group_aware):
    """Compact nested architecture/regularization search for one outer fold."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            'Install TensorFlow with: python -m pip install ".[neural]"') from exc
    inner_folds = 3
    if group_aware:
        cv = _checked_group_splits(
            X, y, groups, inner_folds, seed, "MLP inner tuning CV")
    else:
        cv = list(StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=seed).split(X, y))
    candidates = (
        {"architecture": (64, 32), "learning_rate": 1e-3,
         "l2": 0., "dropout": 0.},
        {"architecture": (128, 64), "learning_rate": 1e-3,
         "l2": 0., "dropout": 0.},
        {"architecture": (128, 64, 32), "learning_rate": 1e-3,
         "l2": 0., "dropout": 0.},
        {"architecture": (128, 64, 32), "learning_rate": 3e-4,
         "l2": 0., "dropout": 0.},
        {"architecture": (128, 64, 32), "learning_rate": 1e-3,
         "l2": 1e-4, "dropout": 0.},
        {"architecture": (128, 64, 32), "learning_rate": 1e-3,
         "l2": 0., "dropout": .1},
    )
    print(f"Tuning MLP: {len(candidates)} configurations x {inner_folds} folds",
          flush=True)
    rows = []
    for candidate_index, candidate in enumerate(candidates):
        aucs, epochs = [], []
        for inner_fold, (train_idx, test_idx) in enumerate(cv):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train, w_test = w[train_idx], w[test_idx]
            if group_aware:
                fit_idx, val_idx = _validation_indices(
                    X_train, y_train, groups[train_idx], seed + inner_fold)
                X_train, X_val = X_train[fit_idx], X_train[val_idx]
                y_train, y_val = y_train[fit_idx], y_train[val_idx]
                w_train, w_val = w_train[fit_idx], w_train[val_idx]
            else:
                X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
                    X_train, y_train, w_train, test_size=.20,
                    random_state=seed, stratify=y_train)
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            spw = w_train[y_train == 0].sum() / w_train[y_train == 1].sum()
            w_train_bal = w_train.copy(); w_train_bal[y_train == 1] *= spw
            w_val_bal = w_val.copy(); w_val_bal[y_val == 1] *= spw
            model_seed = seed + 100 * candidate_index + inner_fold
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(model_seed)
            model = _build_mlp(
                tf, X_train.shape[1], candidate["architecture"],
                candidate["learning_rate"], candidate["l2"],
                candidate["dropout"])
            callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True,
                verbose=0)
            history = model.fit(
                X_train, y_train, sample_weight=w_train_bal,
                validation_data=(X_val, y_val, w_val_bal), epochs=100,
                batch_size=256, callbacks=[callback], verbose=0)
            scores = model.predict(X_test, batch_size=4096, verbose=0).ravel()
            aucs.append(roc_auc_score(y_test, scores, sample_weight=w_test))
            epochs.append(len(history.history["loss"]))
        row = {
            "architecture": list(candidate["architecture"]),
            "learning_rate": candidate["learning_rate"],
            "l2": candidate["l2"], "dropout": candidate["dropout"],
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "mean_epochs": float(np.mean(epochs)),
        }
        rows.append(row)
        print(f"  architecture={row['architecture']}, lr={row['learning_rate']:g}, "
              f"l2={row['l2']:g}, dropout={row['dropout']:g}: "
              f"AUC={row['mean_auc']:.4f} +/- {row['std_auc']:.4f}", flush=True)
    best = max(rows, key=lambda row: row["mean_auc"])
    print(f"Selected architecture={best['architecture']}, "
          f"lr={best['learning_rate']:g}, l2={best['l2']:g}, "
          f"dropout={best['dropout']:g}", flush=True)
    return {
        "grid": rows, "inner_folds": inner_folds,
        "best_architecture": best["architecture"],
        "best_learning_rate": best["learning_rate"],
        "best_l2": best["l2"], "best_dropout": best["dropout"],
        "best_mean_auc": best["mean_auc"],
        "best_std_auc": best["std_auc"],
    }


def _tune_qda(X, y, w, groups, seed, folds, group_aware):
    """Nested covariance-regularization search for weighted-bootstrap QDA."""
    if group_aware:
        cv = _checked_group_splits(
            X, y, groups, folds, seed, "QDA inner tuning CV")
    else:
        cv = list(StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed).split(X, y))
    values = (0., .01, .05, .1, .2, .5)
    print(f"Tuning QDA: {len(values)} reg_param values x {folds} folds",
          flush=True)
    rows = []
    for reg_param in values:
        aucs = []
        for inner_fold, (train_idx, test_idx) in enumerate(cv):
            # Use the same resample seed for every regularization candidate so
            # candidates differ only in reg_param, not bootstrap fluctuations.
            scaler, model = _fit_weighted_qda(
                X[train_idx], y[train_idx], w[train_idx],
                seed + inner_fold, reg_param)
            scores = model.predict_proba(scaler.transform(X[test_idx]))[:, 1]
            if not np.isfinite(scores).all():
                raise ValueError(
                    f"QDA produced nonfinite inner-fold scores at reg_param={reg_param}")
            aucs.append(roc_auc_score(
                y[test_idx], scores, sample_weight=w[test_idx]))
        row = {"reg_param": reg_param, "mean_auc": float(np.mean(aucs)),
               "std_auc": float(np.std(aucs))}
        rows.append(row)
        print(f"  reg_param={reg_param:g}: AUC={row['mean_auc']:.4f} +/- "
              f"{row['std_auc']:.4f}", flush=True)
    best = max(rows, key=lambda row: (row["mean_auc"],
                                      -abs(row["reg_param"] - .1)))
    print(f"Selected reg_param={best['reg_param']:g}", flush=True)
    return {"grid": rows, "best_reg_param": best["reg_param"],
            "best_mean_auc": best["mean_auc"],
            "best_std_auc": best["std_auc"]}


def _tune_gaussian_nb(X, y, w, groups, seed, folds, group_aware):
    """Nested var_smoothing search using native physical sample weights."""
    if group_aware:
        cv = _checked_group_splits(
            X, y, groups, folds, seed, "GaussianNB inner tuning CV")
    else:
        cv = list(StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed).split(X, y))
    values = (1e-11, 1e-10, 1e-9, 1e-8, 1e-7)
    print(f"Tuning GaussianNB: {len(values)} var_smoothing values x {folds} folds",
          flush=True)
    rows = []
    for var_smoothing in values:
        aucs = []
        for train_idx, test_idx in cv:
            scaler = StandardScaler().fit(X[train_idx])
            model = GaussianNB(var_smoothing=var_smoothing)
            model.fit(
                scaler.transform(X[train_idx]), y[train_idx],
                sample_weight=w[train_idx])
            scores = model.predict_proba(scaler.transform(X[test_idx]))[:, 1]
            aucs.append(roc_auc_score(
                y[test_idx], scores, sample_weight=w[test_idx]))
        row = {"var_smoothing": var_smoothing,
               "mean_auc": float(np.mean(aucs)),
               "std_auc": float(np.std(aucs))}
        rows.append(row)
        print(f"  var_smoothing={var_smoothing:g}: "
              f"AUC={row['mean_auc']:.4f} +/- {row['std_auc']:.4f}", flush=True)
    best = max(rows, key=lambda row: (
        row["mean_auc"], -abs(np.log10(row["var_smoothing"]) + 9)))
    print(f"Selected var_smoothing={best['var_smoothing']:g}", flush=True)
    return {"grid": rows, "best_var_smoothing": best["var_smoothing"],
            "best_mean_auc": best["mean_auc"],
            "best_std_auc": best["std_auc"]}

def _fit_fold(name, X_full, y_full, w_full, samples_full, X_test, X_data,
              seed, fold, tuned=None):
    fit_idx, val_idx = _process_validation_indices(
        y_full, samples_full, seed + fold)
    X_train, X_val = X_full[fit_idx], X_full[val_idx]
    y_train, y_val = y_full[fit_idx], y_full[val_idx]
    w_train, w_val = w_full[fit_idx], w_full[val_idx]

    details = {}

    # The reference QDA first performs a physical-weight bootstrap and then
    # fits the scaler on that resampled training population.
    if name == "qda":
        reg_param = tuned["best_reg_param"] if tuned else .1
        scaler, model = _fit_weighted_qda(
            X_train, y_train, w_train, seed + fold, reg_param)
        test_scores = model.predict_proba(scaler.transform(X_test))[:, 1]
        data_scores = (model.predict_proba(scaler.transform(X_data))[:, 1]
                       if len(X_data) else np.empty(0))
        details["reg_param"] = reg_param
        val_scores = model.predict_proba(scaler.transform(X_val))[:, 1]
        return (test_scores, data_scores, val_scores, y_val, w_val,
                {"scaler": scaler, "model": model}, details)

    scaler, X_train, X_test, X_data, X_val = _scaled_inputs(
        name, X_train, X_test, X_data, X_val)

    if name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError('Install XGBoost with: python -m pip install ".[boosters]"') from exc
        spw = w_train[y_train == 0].sum() / w_train[y_train == 1].sum()
        max_depth = tuned["best_max_depth"] if tuned else 4
        min_child_weight = tuned["best_min_child_weight"] if tuned else 1
        model = _xgboost_estimator(
            xgb, seed, max_depth, min_child_weight, spw)
        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_val, y_val)], sample_weight_eval_set=[w_val],
                  verbose=False)
        details["best_iteration"] = getattr(model, "best_iteration", None)
        details["max_depth"] = max_depth
        details["min_child_weight"] = min_child_weight

    elif name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError('Install LightGBM with: python -m pip install ".[boosters]"') from exc
        spw = w_train[y_train == 0].sum() / w_train[y_train == 1].sum()
        max_depth = tuned["best_max_depth"] if tuned else 4
        num_leaves = tuned["best_num_leaves"] if tuned else 15
        min_child_samples = tuned["best_min_child_samples"] if tuned else 20
        model = _lightgbm_estimator(
            lgb, seed, max_depth, num_leaves, min_child_samples, spw)
        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_val, y_val)], eval_sample_weight=[w_val],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
        details["best_iteration"] = getattr(model, "best_iteration_", None)
        details["max_depth"] = max_depth
        details["num_leaves"] = num_leaves
        details["min_child_samples"] = min_child_samples

    elif name == "random_forest":
        max_depth = tuned["best_max_depth"] if tuned else 5
        min_samples_leaf = tuned["best_min_samples_leaf"] if tuned else 5
        model = _random_forest_estimator(
            seed + fold, max_depth, min_samples_leaf)
        model.fit(X_train, y_train,
                  sample_weight=_balanced_weights(y_train, w_train))
        details["max_depth"] = max_depth
        details["min_samples_leaf"] = min_samples_leaf
        details["max_features"] = "sqrt"

    elif name == "logistic_regression":
        c_value = tuned["best_C"] if tuned else 100.
        model = _logistic_estimator(seed + fold, c_value)
        model.fit(X_train, y_train,
                  sample_weight=_balanced_weights(y_train, w_train))
        details["C"] = c_value

    elif name == "gaussian_nb":
        var_smoothing = tuned["best_var_smoothing"] if tuned else 1e-9
        model = GaussianNB(var_smoothing=var_smoothing)
        model.fit(X_train, y_train, sample_weight=w_train)
        details["var_smoothing"] = var_smoothing

    elif name == "mlp":
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError('Install TensorFlow with: python -m pip install ".[neural]"') from exc
        architecture = tuple(tuned["best_architecture"]) if tuned else (128, 64, 32)
        learning_rate = tuned["best_learning_rate"] if tuned else 1e-3
        l2_value = tuned["best_l2"] if tuned else 0.
        dropout = tuned["best_dropout"] if tuned else 0.
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed + fold)
        model = _build_mlp(
            tf, X_train.shape[1], architecture, learning_rate, l2_value, dropout)
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
        details["architecture"] = list(architecture)
        details["learning_rate"] = learning_rate
        details["l2"] = l2_value
        details["dropout"] = dropout
    else:
        raise ValueError(f"Unsupported model: {name}")

    if name == "mlp":
        test_scores = model.predict(X_test, batch_size=4096, verbose=0).ravel()
        data_scores = (model.predict(X_data, batch_size=4096, verbose=0).ravel()
                       if len(X_data) else np.empty(0))
        val_scores = model.predict(X_val, batch_size=4096, verbose=0).ravel()
    else:
        test_scores = model.predict_proba(X_test)[:, 1]
        data_scores = (model.predict_proba(X_data)[:, 1]
                       if len(X_data) else np.empty(0))
        val_scores = model.predict_proba(X_val)[:, 1]
    return (test_scores, data_scores, val_scores, y_val, w_val,
            {"scaler": scaler, "model": model}, details)

def train_models(frame, config):
    """Cross-fit MC and data with frozen process-stratified folds.

    MC events are scored only by their held-out model.  Data events receive a
    deterministic fold and are scored only by that fold's model.  Raw outputs
    are converted to fold-local signal percentiles using training-side
    validation signal, making the configured cut comparable across folds.
    """
    t = config.training
    validate_frame(frame, t.features)
    if t.model not in SUPPORTED_MODELS:
        raise ValueError(f"Choose one of: {', '.join(SUPPORTED_MODELS)}")
    # Match build_ml_dataset_with_mass() in the reference: signal rows first,
    # followed by the background samples in their prepared order. Fold
    # membership is deterministic but depends on this row order.
    mc = pd.concat([frame[frame.role == "signal"],
                    frame[frame.role == "background"]]).copy()
    data = frame[frame.role == "data"].copy()
    X = mc[list(t.features)].to_numpy(float)
    y = mc.label.to_numpy(int)
    w = mc.weight.to_numpy(float)
    if "sample" not in mc or mc["sample"].isna().any():
        raise ValueError("Process-stratified CV requires nonmissing prepared 'sample'")
    samples = mc["sample"].astype(str).to_numpy()
    groups = np.arange(len(mc))
    X_data = data[list(t.features)].to_numpy(float)
    if (w < 0).any():
        raise ValueError("Negative MC weights need a reviewed model-specific policy")
    if any(np.count_nonzero((y == label) & (w > 0)) < t.folds for label in (0, 1)):
        raise ValueError("Each MC class needs at least folds positive-weight events")

    tuning = ([] if t.model in {"xgboost", "lightgbm", "random_forest",
                                "logistic_regression", "mlp", "qda",
                                "gaussian_nb"}
              and t.nested_tuning
              else None)
    oof = np.full(len(mc), np.nan)
    fold_ids = np.full(len(mc), -1, int)
    raw_oof = np.full(len(mc), np.nan)
    raw_data = np.full(len(data), np.nan)
    data_folds = _data_fold_ids(len(data), t.folds, t.seed)
    models, metrics = [], []
    cv = _process_stratified_splits(y, samples, t.folds, t.seed)
    for fold, (train_idx, test_idx) in enumerate(cv):
        if any(w[train_idx][y[train_idx] == label].sum() <= 0 or
               w[test_idx][y[test_idx] == label].sum() <= 0
               for label in (0, 1)):
            raise ValueError("Every fold needs positive total weight in each class")
        fold_tuning = None
        if t.model == "xgboost":
            if t.nested_tuning:
                fold_tuning = _tune_xgboost(
                    X[train_idx], y[train_idx], w[train_idx], groups[train_idx],
                    t.seed + fold, t.folds, t.group_aware)
                tuning.append({"outer_fold": fold, **fold_tuning})
            else:
                fold_tuning = {"best_max_depth": 4, "best_min_child_weight": 1,
                               "mode": "fixed; nested tuning disabled"}
        elif t.model == "lightgbm":
            if t.nested_tuning:
                fold_tuning = _tune_lightgbm(
                    X[train_idx], y[train_idx], w[train_idx], groups[train_idx],
                    t.seed + fold, t.folds, t.group_aware)
                tuning.append({"outer_fold": fold, **fold_tuning})
            else:
                fold_tuning = {
                    "best_max_depth": 4, "best_num_leaves": 15,
                    "best_min_child_samples": 20,
                    "mode": "fixed; nested tuning disabled"}
        elif t.model == "random_forest":
            if t.nested_tuning:
                fold_tuning = _tune_random_forest(
                    X[train_idx], y[train_idx], w[train_idx], groups[train_idx],
                    t.seed + fold, t.folds, t.group_aware)
                tuning.append({"outer_fold": fold, **fold_tuning})
            else:
                fold_tuning = {
                    "best_max_depth": 5, "best_min_samples_leaf": 5,
                    "best_max_features": "sqrt",
                    "mode": "fixed; nested tuning disabled"}
        elif t.model == "logistic_regression":
            if t.nested_tuning:
                fold_tuning = _tune_logistic_regression(
                    X[train_idx], y[train_idx], w[train_idx], groups[train_idx],
                    t.seed + fold, t.group_aware)
                tuning.append({"outer_fold": fold, **fold_tuning})
            else:
                fold_tuning = {"best_C": 100.,
                               "mode": "fixed; nested tuning disabled"}
        elif t.model == "mlp":
            if t.nested_tuning:
                fold_tuning = _tune_mlp(
                    X[train_idx], y[train_idx], w[train_idx], groups[train_idx],
                    t.seed + fold, t.group_aware)
                tuning.append({"outer_fold": fold, **fold_tuning})
            else:
                fold_tuning = {
                    "best_architecture": [128, 64, 32],
                    "best_learning_rate": 1e-3, "best_l2": 0.,
                    "best_dropout": 0.,
                    "mode": "fixed; nested tuning disabled"}
        elif t.model == "qda":
            if t.nested_tuning:
                fold_tuning = _tune_qda(
                    X[train_idx], y[train_idx], w[train_idx], groups[train_idx],
                    t.seed + fold, t.folds, t.group_aware)
                tuning.append({"outer_fold": fold, **fold_tuning})
            else:
                fold_tuning = {"best_reg_param": .1,
                               "mode": "fixed; nested tuning disabled"}
        elif t.model == "gaussian_nb":
            if t.nested_tuning:
                fold_tuning = _tune_gaussian_nb(
                    X[train_idx], y[train_idx], w[train_idx], groups[train_idx],
                    t.seed + fold, t.folds, t.group_aware)
                tuning.append({"outer_fold": fold, **fold_tuning})
            else:
                fold_tuning = {"best_var_smoothing": 1e-9,
                               "mode": "fixed; nested tuning disabled"}
        data_idx = np.flatnonzero(data_folds == fold)
        (scores, data_scores, val_scores, y_val, w_val,
         pipeline, details) = _fit_fold(
            t.model, X[train_idx], y[train_idx], w[train_idx], samples[train_idx],
            X[test_idx], X_data[data_idx], t.seed, fold, fold_tuning)
        signal_val = y_val == 1
        calibrated = _weighted_signal_cdf(
            val_scores[signal_val], w_val[signal_val], scores)
        oof[test_idx] = calibrated
        raw_oof[test_idx] = scores
        fold_ids[test_idx] = fold
        if len(data_idx):
            raw_data[data_idx] = data_scores
            raw_signal = val_scores[signal_val]
            signal_weights = w_val[signal_val]
            data.loc[data.index[data_idx], "score"] = _weighted_signal_cdf(
                raw_signal, signal_weights, data_scores)
        raw_threshold = _weighted_quantile(
            val_scores[signal_val], w_val[signal_val],
            1-t.target_signal_efficiency)
        metrics.append({"fold": fold, "n_train_full": len(train_idx),
                        "n_test": len(test_idx),
                        "n_data": len(data_idx),
                        "fold_raw_threshold": raw_threshold,
                        "target_signal_efficiency": t.target_signal_efficiency,
                        "weighted_auc": float(roc_auc_score(
                            y[test_idx], scores, sample_weight=w[test_idx])),
                        **details})
        # Keep the original package API: (scaler, fitted model) per fold.
        models.append((pipeline["scaler"], pipeline["model"]))

    mc["raw_score"] = raw_oof
    mc["score"], mc["fold"] = oof, fold_ids
    mc["score_kind"] = "out_of_fold_calibrated"
    data["raw_score"] = raw_data
    data["fold"] = data_folds
    data["score_kind"] = "fold_assigned_calibrated"
    predictions = pd.concat([mc, data], ignore_index=True)
    fold_weight = np.asarray([
        w[fold_ids == fold].sum() for fold in range(t.folds)], float)
    fold_auc = np.asarray([row["weighted_auc"] for row in metrics], float)
    return TrainingResult(predictions, metrics,
                          float(np.average(fold_auc, weights=fold_weight)), models,
                          tuning)
