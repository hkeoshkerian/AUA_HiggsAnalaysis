"""Compare independently saved model prediction tables."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .inference import observed_methods
from .provenance import new_output, write_json
from .thresholds import scan_thresholds


def bootstrap_auc(frame, repeats=1000, seed=42):
    mc = frame[frame.role != "data"]
    y = mc.label.to_numpy(int); score = mc.score.to_numpy(float); weight = mc.weight.to_numpy(float)
    point = float(roc_auc_score(y, score, sample_weight=weight))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = rng.integers(0, len(mc), len(mc))
        if len(np.unique(y[indices])) == 2:
            values.append(roc_auc_score(y[indices], score[indices], sample_weight=weight[indices]))
    if not values:
        raise ValueError("Bootstrap samples did not contain both classes")
    low, high = np.quantile(values, [.025, .975])
    return point, float(low), float(high)


def compare_prediction_frames(frames, thresholds, mass_window=(110., 135.),
                              systematic=.30, sidebands=(90., 105., 140., 155.),
                              bootstrap_repeats=1000, seed=42, include_observed=True):
    rows, observed_rows = [], []
    for offset, (name, frame) in enumerate(frames.items()):
        threshold = thresholds[name]
        point, low, high = bootstrap_auc(frame, bootstrap_repeats, seed+offset)
        scan = scan_thresholds(frame, [threshold], mass_window, systematic).iloc[0]
        rows.append({"model": name, "threshold": threshold, "weighted_oof_auc": point,
                     "auc_ci_low": low, "auc_ci_high": high,
                     "S": scan.S, "B": scan.B, "Z_proxy": scan.Z_proxy,
                     "sigma_Z_proxy": scan.sigma_Z_proxy,
                     "signal_efficiency": scan.signal_efficiency,
                     "background_rejection": scan.background_rejection,
                     "weighted_accuracy": scan.weighted_accuracy})
        if include_observed:
            observed = observed_methods(frame, threshold, mass_window, sidebands, systematic)
            observed.insert(0, "model", name)
            observed_rows.append(observed)
    return pd.DataFrame(rows), (pd.concat(observed_rows, ignore_index=True)
                                if observed_rows else pd.DataFrame())


def _forest_plot(results, metric, low, high, xlabel, path):
    import matplotlib.pyplot as plt
    ordered = results.sort_values(metric)
    values = ordered[metric].to_numpy(float)
    xerr = np.vstack([values-ordered[low].to_numpy(float), ordered[high].to_numpy(float)-values])
    fig, ax = plt.subplots(figsize=(8, max(4, len(ordered)*.65)))
    ax.errorbar(values, ordered.model, xerr=xerr, fmt="o", capsize=4)
    ax.set(xlabel=xlabel, title=f"Model comparison: {xlabel}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def compare_models(prediction_paths, thresholds, output, config,
                   sidebands=(90., 105., 140., 155.), bootstrap_repeats=1000):
    frames = {name: pd.read_csv(path) for name, path in prediction_paths.items()}
    results, observed = compare_prediction_frames(
        frames, thresholds, config.statistics.mass_window_gev,
        config.statistics.background_fractional_systematic, sidebands,
        bootstrap_repeats, config.training.seed, include_observed=config.data.fraction == 1)
    output = new_output(output)
    results.to_csv(output/"model_comparison.csv", index=False)
    from .plots import save_classifier_metrics
    save_classifier_metrics(results, output)
    if len(observed): observed.to_csv(output/"observed_model_comparison.csv", index=False)
    _forest_plot(results, "weighted_oof_auc", "auc_ci_low", "auc_ci_high",
                 "Weighted OOF AUC (bootstrap 95% CI)", output/"auc_forest.png")
    z = results.dropna(subset=["Z_proxy", "sigma_Z_proxy"]).copy()
    if len(z):
        z["z_low"] = z.Z_proxy-z.sigma_Z_proxy; z["z_high"] = z.Z_proxy+z.sigma_Z_proxy
        _forest_plot(z, "Z_proxy", "z_low", "z_high", "Expected MC Z proxy",
                     output/"significance_forest.png")
    write_json(output/"comparison_summary.json", {
        "models": list(frames), "bootstrap_repeats": bootstrap_repeats,
        "observed_included": config.data.fraction == 1,
        "warning": "Comparisons are meaningful only when runs use compatible prepared data and predefined choices."
    })
    return output
