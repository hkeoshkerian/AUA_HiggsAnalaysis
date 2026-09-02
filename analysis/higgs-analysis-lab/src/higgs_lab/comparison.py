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
    for name, frame in frames.items():
        threshold = thresholds[name]
        # The reference restarts the same deterministic bootstrap for each model.
        point, low, high = bootstrap_auc(frame, bootstrap_repeats, seed)
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
    from .plots import (save_all_model_roc, save_classifier_metrics,
                        save_model_significance, save_observed_model_significance)
    from .statistics import summarize_mc
    save_classifier_metrics(results, output)
    save_all_model_roc(frames, results, output/"roc_all_models.png")
    if len(observed):
        observed.to_csv(output/"observed_model_comparison.csv", index=False)
        save_observed_model_significance(
            observed, output/"observed_significance_comparison.png")
        retained = ["XGBoost", "LightGBM", "Random Forest", "MLP",
                    "Logistic Regression"]
        after = observed[(observed.stage == "after_ml") &
                         observed.model.isin(retained)]
        table = after.pivot(index="model", columns="method",
                            values=["background", "Z", "sigma_Z"])
        table = table.reindex([name for name in retained if name in table.index])
        table.columns = [f"{method}_{metric}" for metric, method in table.columns]
        baseline = observed[(observed.stage == "before_ml") &
                            (observed.method == "mc_prediction")].iloc[0]
        baseline_row = pd.DataFrame({
            "mc_prediction_background": [baseline.background],
            "mc_prediction_Z": [baseline.Z],
            "mc_prediction_sigma_Z": [baseline.sigma_Z],
            "sideband_background": [baseline.background],
            "sideband_Z": [baseline.Z],
            "sideband_sigma_Z": [baseline.sigma_Z]}, index=["No ML (baseline)"])
        pd.concat([baseline_row, table]).to_csv(output/"table_vii.csv", index_label="classifier")
    _forest_plot(results, "weighted_oof_auc", "auc_ci_low", "auc_ci_high",
                 "Weighted OOF AUC (bootstrap 95% CI)", output/"auc_forest.png")
    baseline = summarize_mc(
        next(iter(frames.values())), config.statistics.mass_window_gev,
        config.statistics.background_fractional_systematic, config.data.fraction)
    z = results.dropna(subset=["Z_proxy", "sigma_Z_proxy"]).copy()
    if len(z) and baseline.get("Z_proxy") is not None:
        save_model_significance(z, baseline, output/"significance_forest.png")
    write_json(output/"comparison_summary.json", {
        "models": list(frames), "bootstrap_repeats": bootstrap_repeats,
        "observed_included": config.data.fraction == 1,
        "no_ml_baseline": baseline,
        "warning": "Comparisons are meaningful only when runs use compatible prepared data and predefined choices."
    })
    return output
