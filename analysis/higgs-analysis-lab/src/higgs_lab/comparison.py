"""Compare independently saved model prediction tables."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .inference import observed_methods
from .provenance import new_output, write_json
from .statistics import summarize_mc
from .thresholds import scan_thresholds


def bootstrap_auc(frame, repeats=1000, seed=42):
    mc = frame[frame.role != "data"]
    if "fold" not in mc or not mc.score_kind.eq("out_of_fold_calibrated").all():
        raise ValueError("AUC comparison requires calibrated OOF scores and fold IDs")
    folds = sorted(mc.fold.unique())
    def combined_auc(table):
        aucs, totals = [], []
        for fold in folds:
            part = table[table.fold == fold]
            if part.label.nunique() != 2:
                continue
            aucs.append(roc_auc_score(part.label, part.score,
                                      sample_weight=part.weight))
            totals.append(part.weight.sum())
        if not aucs:
            raise ValueError("No fold contains both classes")
        return float(np.average(aucs, weights=totals))
    point = combined_auc(mc)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        pieces = []
        for fold in folds:
            part = mc[mc.fold == fold]
            indices = rng.integers(0, len(part), len(part))
            pieces.append(part.iloc[indices])
        sample = pd.concat(pieces, ignore_index=True)
        try:
            values.append(combined_auc(sample))
        except ValueError:
            pass
    if not values:
        raise ValueError("Bootstrap samples did not contain both classes")
    low, high = np.quantile(values, [.025, .975])
    return point, float(low), float(high)


def compare_prediction_frames(frames, thresholds, mass_window=(118., 130.),
                              systematic=.30, sidebands=(90., 105., 140., 155.),
                              bootstrap_repeats=1000, seed=42, include_observed=True):
    rows, observed_rows = [], []
    reference_assignment = None
    for name, frame in frames.items():
        required = {"event_id", "role", "fold", "score_kind"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} is missing comparison columns: {sorted(missing)}")
        assignment = frame[["event_id", "role", "fold"]].sort_values(
            ["role", "event_id"]).reset_index(drop=True)
        if reference_assignment is None:
            reference_assignment = assignment
        elif not assignment.equals(reference_assignment):
            raise ValueError(
                f"{name} does not use the same event-to-fold assignment as "
                "the other models")
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


def compare_selection_strategies(frames, calibrated_threshold,
                                 mass_window=(118., 130.), systematic=.30,
                                 raw_threshold=.65):
    """Compare the historical raw cut with a fixed-efficiency operating point."""
    rows = []
    for model, frame in frames.items():
        required = {"role", "weight", "mass", "raw_score", "score", "score_kind"}
        missing = required - set(frame)
        if missing:
            raise ValueError(
                f"{model} cannot compare selection strategies; missing {sorted(missing)}")
        mc = frame[frame.role.isin(["signal", "background"])]
        if not mc.score_kind.eq("out_of_fold_calibrated").all():
            raise ValueError(f"{model} strategy comparison requires calibrated OOF MC")
        total_signal = mc.loc[mc.role == "signal", "weight"].sum()
        total_background = mc.loc[mc.role == "background", "weight"].sum()
        for strategy, column, threshold in (
                ("global raw score > 0.65", "raw_score", raw_threshold),
                ("fold-local 80% signal efficiency", "score",
                 calibrated_threshold)):
            selected = mc[mc[column] > threshold]
            selected_signal = selected[selected.role == "signal"]
            selected_background = selected[selected.role == "background"]
            signal_efficiency = (selected_signal.weight.sum()/total_signal
                                 if total_signal > 0 else np.nan)
            background_efficiency = (selected_background.weight.sum()/total_background
                                     if total_background > 0 else np.nan)
            summary = summarize_mc(selected, mass_window, systematic, fraction=1.)
            rows.append({
                "model": model,
                "selection_strategy": strategy,
                "score_column": column,
                "threshold": threshold,
                "oof_signal_efficiency": signal_efficiency,
                "oof_background_rejection": 1-background_efficiency,
                "signal_yield_mass_window": summary["S"],
                "background_yield_mass_window": summary["B"],
                "background_constraint_sigma": summary.get(
                    "background_constraint_sigma"),
                "expected_profile_Z": summary.get("expected_profile_Z"),
            })
    return pd.DataFrame(rows)


def _write_strategy_markdown(table, path):
    columns = ["model", "selection_strategy", "oof_signal_efficiency",
               "oof_background_rejection", "signal_yield_mass_window",
               "background_yield_mass_window", "expected_profile_Z"]
    labels = ["Model", "Selection", "Signal eff.", "Bkg rejection",
              "S (118–130)", "B (118–130)", "Expected profile Z"]
    lines = ["| " + " | ".join(labels) + " |",
             "|" + "|".join(["---"]*len(labels)) + "|"]
    for row in table[columns].itertuples(index=False):
        values = [row[0], row[1]] + [f"{float(value):.4f}" for value in row[2:]]
        lines.append("| " + " | ".join(values) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


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
    strategies = compare_selection_strategies(
        frames, config.training.threshold,
        config.statistics.mass_window_gev,
        config.statistics.background_fractional_systematic)
    strategies.to_csv(output/"selection_strategy_comparison.csv", index=False)
    _write_strategy_markdown(
        strategies, output/"selection_strategy_comparison.md")
    from .plots import (save_all_model_roc, save_classifier_metrics,
                        save_model_significance, save_observed_model_significance,
                        save_observed_profile_significance)
    save_classifier_metrics(results, output)
    save_all_model_roc(frames, results, output/"roc_all_models.png")
    if len(observed):
        observed.to_csv(output/"observed_model_comparison.csv", index=False)
        save_observed_model_significance(
            observed, output/"observed_significance_comparison.png")
        save_observed_profile_significance(
            observed, output/"observed_profile_likelihood_comparison.png")
        retained = ["XGBoost", "LightGBM", "Random Forest", "MLP",
                    "Logistic Regression"]
        after = observed[(observed.stage == "after_ml") &
                         observed.model.isin(retained)]
        table = after.pivot(index="model", columns="method",
                            values=["background", "Z", "sigma_Z", "profile_Z",
                                    "profile_p_value_one_sided"])
        table = table.reindex([name for name in retained if name in table.index])
        table.columns = [f"{method}_{metric}" for metric, method in table.columns]
        baseline_mc = observed[(observed.stage == "before_ml") &
                               (observed.method == "mc_prediction")].iloc[0]
        baseline_sideband = observed[(observed.stage == "before_ml") &
                                     (observed.method == "sideband")].iloc[0]
        baseline_row = pd.DataFrame({
            "mc_prediction_background": [baseline_mc.background],
            "mc_prediction_Z": [baseline_mc.Z],
            "mc_prediction_sigma_Z": [baseline_mc.sigma_Z],
            "mc_prediction_profile_Z": [baseline_mc.profile_Z],
            "mc_prediction_profile_p_value_one_sided": [
                baseline_mc.profile_p_value_one_sided],
            "sideband_background": [baseline_sideband.background],
            "sideband_Z": [baseline_sideband.Z],
            "sideband_sigma_Z": [baseline_sideband.sigma_Z],
            "sideband_profile_Z": [baseline_sideband.profile_Z],
            "sideband_profile_p_value_one_sided": [
                baseline_sideband.profile_p_value_one_sided]},
            index=["No ML (baseline)"])
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
