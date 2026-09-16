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
                     "S": scan.S, "B": scan.B,
                     "expected_profile_Z": scan.expected_profile_Z,
                     "sigma_expected_profile_Z": scan.sigma_expected_profile_Z,
                     "background_constraint_sigma": scan.background_constraint_sigma,
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
                "sigma_expected_profile_Z": summary.get(
                    "sigma_expected_profile_Z"),
            })
    return pd.DataFrame(rows)


def scan_signal_efficiencies(frames, efficiencies, mass_window=(118., 130.),
                             systematic=.30):
    """Evaluate common signal-quantile operating points using OOF MC only."""
    efficiencies = np.asarray(list(efficiencies), dtype=float)
    if (efficiencies.size == 0 or not np.isfinite(efficiencies).all()
            or np.any((efficiencies <= 0) | (efficiencies >= 1))):
        raise ValueError("Signal efficiencies must be finite values in (0, 1)")
    if len(np.unique(efficiencies)) != len(efficiencies):
        raise ValueError("Signal efficiencies must be unique")

    aggregate, by_fold = [], []
    for model, frame in frames.items():
        mc = frame[frame.role.isin(["signal", "background"])]
        if mc.empty or not mc.score_kind.eq("out_of_fold_calibrated").all():
            raise ValueError(f"{model} efficiency scan requires calibrated OOF MC")
        for target in sorted(efficiencies):
            threshold = round(1.-float(target), 12)
            row = scan_thresholds(
                frame, [threshold], mass_window, systematic).iloc[0]
            aggregate.append({
                "model": model, "target_signal_efficiency": target,
                "calibrated_threshold": threshold,
                "achieved_signal_efficiency": row.signal_efficiency,
                "background_rejection": row.background_rejection,
                "signal_yield": row.S, "background_yield": row.B,
                "background_constraint_sigma": row.background_constraint_sigma,
                "expected_profile_Z": row.expected_profile_Z,
                "sigma_expected_profile_Z": row.sigma_expected_profile_Z,
            })
            for fold, part in mc.groupby("fold", sort=True):
                fold_scan = scan_thresholds(
                    part, [threshold], mass_window, systematic).iloc[0]
                by_fold.append({
                    "model": model, "fold": int(fold),
                    "target_signal_efficiency": target,
                    "calibrated_threshold": threshold,
                    "achieved_signal_efficiency": fold_scan.signal_efficiency,
                    "background_rejection": fold_scan.background_rejection,
                    "signal_yield": fold_scan.S,
                    "background_yield": fold_scan.B,
                    "expected_profile_Z": fold_scan.expected_profile_Z,
                    "sigma_expected_profile_Z": fold_scan.sigma_expected_profile_Z,
                })
    aggregate = pd.DataFrame(aggregate)
    by_fold = pd.DataFrame(by_fold)
    fold_stability = by_fold.groupby(
        ["model", "target_signal_efficiency"], as_index=False).agg(
            fold_signal_efficiency_std=("achieved_signal_efficiency", "std"),
            fold_background_rejection_std=("background_rejection", "std"),
            fold_profile_Z_std=("expected_profile_Z", "std"))
    aggregate = aggregate.merge(
        fold_stability, on=["model", "target_signal_efficiency"], how="left")
    summary = aggregate.groupby("target_signal_efficiency", as_index=False).agg(
        mean_expected_profile_Z=("expected_profile_Z", "mean"),
        model_spread_profile_Z=("expected_profile_Z", "std"),
        mean_fold_profile_Z_std=("fold_profile_Z_std", "mean"))
    statistical = aggregate.assign(
        variance=lambda table: table.sigma_expected_profile_Z**2).groupby(
            "target_signal_efficiency", as_index=False).agg(
                summed_variance=("variance", "sum"), model_count=("model", "count"))
    summary = summary.merge(statistical, on="target_signal_efficiency")
    summary["sigma_mean_expected_profile_Z"] = (
        np.sqrt(summary.summed_variance)/summary.model_count)
    summary = summary.drop(columns=["summed_variance", "model_count"])
    chosen = summary.sort_values(
        ["mean_expected_profile_Z", "target_signal_efficiency"],
        ascending=[False, True]).iloc[0]
    return aggregate, by_fold, summary, chosen.to_dict()


def save_efficiency_scan(aggregate, by_fold, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for model, group in aggregate.groupby("model", sort=False):
        group = group.sort_values("target_signal_efficiency")
        axes[0, 0].errorbar(group.target_signal_efficiency,
                            group.expected_profile_Z,
                            yerr=group.sigma_expected_profile_Z,
                            marker="o", capsize=3, label=model)
        axes[0, 1].plot(group.target_signal_efficiency,
                        group.signal_yield, marker="o", label=f"{model} S")
        axes[0, 1].plot(group.target_signal_efficiency,
                        group.background_yield, marker="s", linestyle="--",
                        label=f"{model} B")
        axes[1, 0].plot(group.target_signal_efficiency,
                        group.background_rejection, marker="o", label=model)
    for (model, fold), group in by_fold.groupby(["model", "fold"], sort=False):
        group = group.sort_values("target_signal_efficiency")
        axes[1, 1].plot(group.target_signal_efficiency,
                        group.expected_profile_Z, alpha=.28, linewidth=1)
    axes[0, 0].plot(summary.target_signal_efficiency,
                    summary.mean_expected_profile_Z, color="black", linewidth=3,
                    marker="D", label="Across-model mean")
    axes[0, 0].set(ylabel=r"Expected profile $Z_A$",
                   title="Expected sensitivity (OOF MC only)")
    axes[0, 1].set(ylabel="Weighted yield in 118–130 GeV",
                   title="Signal and background yields")
    axes[1, 0].set(xlabel=r"Target signal efficiency $\epsilon_S$",
                   ylabel="Background rejection", title="Background rejection")
    axes[1, 1].set(xlabel=r"Target signal efficiency $\epsilon_S$",
                   ylabel=r"Fold contribution to expected $Z_A$",
                   title="Outer-fold stability")
    axes[0, 0].legend(fontsize=9); axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 0].legend(fontsize=9)
    for ax in axes.flat: ax.grid(alpha=.25)
    fig.suptitle("Common signal-efficiency operating-point study", weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)


def _write_strategy_markdown(table, path):
    columns = ["model", "selection_strategy", "oof_signal_efficiency",
               "oof_background_rejection", "signal_yield_mass_window",
               "background_yield_mass_window", "expected_profile_Z",
               "sigma_expected_profile_Z"]
    labels = ["Model", "Selection", "Signal eff.", "Bkg rejection",
              "S (118–130)", "B (118–130)", "Expected profile Z",
              "Stat. error on Z"]
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
                   sidebands=(90., 105., 140., 155.), bootstrap_repeats=1000,
                   efficiencies=(.60, .65, .70, .75, .80, .85, .90)):
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
    efficiency_scan, efficiency_folds, efficiency_summary, chosen_efficiency = (
        scan_signal_efficiencies(
            frames, efficiencies, config.statistics.mass_window_gev,
            config.statistics.background_fractional_systematic))
    efficiency_scan.to_csv(output/"signal_efficiency_scan.csv", index=False)
    efficiency_folds.to_csv(output/"signal_efficiency_scan_by_fold.csv", index=False)
    efficiency_summary.to_csv(output/"signal_efficiency_scan_summary.csv", index=False)
    save_efficiency_scan(
        efficiency_scan, efficiency_folds, efficiency_summary,
        output/"signal_efficiency_scan.png")
    write_json(output/"chosen_signal_efficiency.json", {
        "selection_source": "calibrated out-of-fold MC predictions only",
        "candidate_efficiencies": list(map(float, efficiencies)),
        "selection_rule": "maximize the across-model mean expected Asimov one-bin profile-likelihood Z; ties prefer lower efficiency",
        "chosen": chosen_efficiency,
        "observed_data_used": False,
        "warning": "Treat the chosen point as established only if the scan and fold diagnostics show a stable sensitivity plateau."
    })
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
                            values=["background", "profile_Z", "sigma_profile_Z",
                                    "profile_p_value_one_sided"])
        table = table.reindex([name for name in retained if name in table.index])
        table.columns = [f"{method}_{metric}" for metric, method in table.columns]
        baseline_mc = observed[(observed.stage == "before_ml") &
                               (observed.method == "mc_prediction")].iloc[0]
        baseline_sideband = observed[(observed.stage == "before_ml") &
                                     (observed.method == "sideband")].iloc[0]
        baseline_row = pd.DataFrame({
            "mc_prediction_background": [baseline_mc.background],
            "mc_prediction_profile_Z": [baseline_mc.profile_Z],
            "mc_prediction_sigma_profile_Z": [baseline_mc.sigma_profile_Z],
            "mc_prediction_profile_p_value_one_sided": [
                baseline_mc.profile_p_value_one_sided],
            "sideband_background": [baseline_sideband.background],
            "sideband_profile_Z": [baseline_sideband.profile_Z],
            "sideband_sigma_profile_Z": [baseline_sideband.sigma_profile_Z],
            "sideband_profile_p_value_one_sided": [
                baseline_sideband.profile_p_value_one_sided]},
            index=["No ML (baseline)"])
        pd.concat([baseline_row, table]).to_csv(output/"table_vii.csv", index_label="classifier")
    _forest_plot(results, "weighted_oof_auc", "auc_ci_low", "auc_ci_high",
                 "Weighted OOF AUC (bootstrap 95% CI)", output/"auc_forest.png")
    baseline = summarize_mc(
        next(iter(frames.values())), config.statistics.mass_window_gev,
        config.statistics.background_fractional_systematic, config.data.fraction)
    z = results.dropna(subset=["expected_profile_Z", "sigma_expected_profile_Z"]).copy()
    if len(z) and baseline.get("expected_profile_Z") is not None:
        save_model_significance(z, baseline, output/"significance_forest.png")
    write_json(output/"comparison_summary.json", {
        "models": list(frames), "bootstrap_repeats": bootstrap_repeats,
        "observed_included": config.data.fraction == 1,
        "no_ml_baseline": baseline,
        "warning": "Comparisons are meaningful only when runs use compatible prepared data and predefined choices."
    })
    return output
