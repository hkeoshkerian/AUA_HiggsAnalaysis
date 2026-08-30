"""Training-only feature ranking and correlation pruning."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .features import FEATURE_KEYS, validate_frame
from .provenance import new_output, sha256, write_json


def separation_power(signal, background, signal_weight, background_weight, n_bins=50):
    """Paper Eq. 3, using weighted, normalized class histograms."""
    signal = np.asarray(signal, dtype=float)
    background = np.asarray(background, dtype=float)
    low = min(signal.min(), background.min())
    high = max(signal.max(), background.max())
    if low == high:
        return 0.0
    bins = np.linspace(low, high, n_bins + 1)
    s = np.histogram(signal, bins=bins, weights=signal_weight)[0]
    b = np.histogram(background, bins=bins, weights=background_weight)[0]
    s = s / (s.sum() + 1e-12)
    b = b / (b.sum() + 1e-12)
    denominator = s + b
    terms = np.divide((s-b)**2, denominator, out=np.zeros_like(s), where=denominator > 0)
    return float(0.5 * terms.sum())


def rank_features(frame, seed=42, train_fraction=0.60):
    """Rank all reconstructed features using only a stratified MC training split."""
    validate_frame(frame, FEATURE_KEYS)
    mc = frame[frame.role != "data"].copy()
    if (mc.weight < 0).any():
        raise ValueError("Feature ranking does not support negative MC weights")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    indices = np.arange(len(mc))
    train_indices, _ = train_test_split(
        indices, train_size=train_fraction, stratify=mc.label, random_state=seed)
    train = mc.iloc[train_indices]
    y = train.label.to_numpy(int)
    weights = train.weight.to_numpy(float)
    signal = y == 1
    background = y == 0
    if weights[signal].sum() <= 0 or weights[background].sum() <= 0:
        raise ValueError("Both classes need positive total training weight")

    rows = []
    X = train[list(FEATURE_KEYS)].to_numpy(float)
    for column, name in enumerate(FEATURE_KEYS):
        values = X[:, column]
        auc = roc_auc_score(y, values, sample_weight=weights)
        rows.append({
            "feature": name,
            "separation_power": separation_power(
                values[signal], values[background], weights[signal], weights[background]),
            "single_feature_auc": float(max(auc, 1-auc)),
        })

    balanced = weights.copy()
    balanced[signal] *= weights[background].sum() / weights[signal].sum()
    forest = RandomForestClassifier(
        n_estimators=300, max_depth=5, n_jobs=-1, random_state=seed)
    forest.fit(X, y, sample_weight=balanced)
    for row, importance in zip(rows, forest.feature_importances_):
        row["rf_importance"] = float(importance)

    ranking = pd.DataFrame(rows)
    for metric in ("separation_power", "single_feature_auc", "rf_importance"):
        ranking[f"{metric}_rank"] = ranking[metric].rank(method="min", ascending=False).astype(int)
    rank_columns = [c for c in ranking if c.endswith("_rank")]
    ranking["mean_rank"] = ranking[rank_columns].mean(axis=1)
    ranking = ranking.sort_values(["mean_rank", "separation_power"], ascending=[True, False]).reset_index(drop=True)
    return ranking, train


def prune_correlations(training_frame, scores, threshold=0.70):
    """Drop the lower-ranked feature from each highly correlated signal pair."""
    if not 0 < threshold <= 1:
        raise ValueError("correlation threshold must be in (0, 1]")
    names = list(scores)
    signal = training_frame[training_frame.label == 1]
    correlations = signal[names].corr().abs().fillna(0.0)
    pairs = []
    for left_index, left in enumerate(names):
        for right in names[left_index+1:]:
            value = float(correlations.loc[left, right])
            if value >= threshold:
                pairs.append((left, right, value))
    dropped = set()
    decisions = []
    for left, right, value in sorted(pairs, key=lambda item: item[2], reverse=True):
        if left in dropped or right in dropped:
            continue
        keep, drop = (left, right) if scores[left] >= scores[right] else (right, left)
        dropped.add(drop)
        decisions.append({"feature_1": left, "feature_2": right,
                          "absolute_correlation": value, "keep": keep, "drop": drop})
    columns = ["feature_1", "feature_2", "absolute_correlation", "keep", "drop"]
    return ([name for name in names if name not in dropped], sorted(dropped),
            pd.DataFrame(decisions, columns=columns))


def _save_metric_plot(ranking, metric, path, title):
    import matplotlib.pyplot as plt
    ordered = ranking.sort_values(metric)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(ordered.feature, ordered[metric], color="steelblue", alpha=.85)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_features(prepared, output, seed=42, correlation_threshold=.70):
    """Analyze a verified prepared cache and write rankings and plots."""
    prepared = Path(prepared)
    manifest = json.loads((prepared/"manifest.json").read_text())
    csv_path = prepared/"features.csv"
    if sha256(csv_path) != manifest.get("features_sha256"):
        raise ValueError("Prepared feature checksum mismatch")
    frame = pd.read_csv(csv_path)
    ranking, training = rank_features(frame, seed=seed)
    output = new_output(output)
    ranking.to_csv(output/"feature_ranking.csv", index=False)

    pruning = {}
    decisions = []
    for metric in ("separation_power", "single_feature_auc", "rf_importance"):
        scores = dict(zip(ranking.feature, ranking[metric]))
        kept, dropped, table = prune_correlations(training, scores, correlation_threshold)
        pruning[metric] = {"kept": kept, "dropped": dropped}
        if len(table):
            table.insert(0, "method", metric)
            decisions.append(table)
        _save_metric_plot(ranking, metric, output/f"{metric}.png",
                          f"{metric.replace('_', ' ').title()} — training split only")
    pd.concat(decisions, ignore_index=True).to_csv(
        output/"correlation_pruning.csv", index=False) if decisions else pd.DataFrame(
            columns=["method", "feature_1", "feature_2", "absolute_correlation", "keep", "drop"]
        ).to_csv(output/"correlation_pruning.csv", index=False)
    write_json(output/"feature_analysis.json", {
        "seed": seed, "train_fraction": .60,
        "training_events": len(training),
        "correlation_threshold": correlation_threshold,
        "pruning": pruning,
        "note": "Ranking and correlation decisions use the MC training split only."
    })
    return output
