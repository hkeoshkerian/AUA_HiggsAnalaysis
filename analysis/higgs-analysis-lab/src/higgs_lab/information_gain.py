"""Controlled study of information added by an OOF classifier score."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .provenance import new_output
from .shape_likelihood import (
    MassShapeLikelihood, PUBLISHED_ZZ_UNCERTAINTIES, TemplateCategory,
    build_templates, _histogram,
)
from .training import _weighted_quantile


def _score_categories(frame, mass_edges, score_edges, prefix):
    score_edges = np.asarray(score_edges, dtype=float)
    if (len(score_edges) < 3 or score_edges[0] != 0 or score_edges[-1] != 1
            or np.any(np.diff(score_edges) <= 0)):
        raise ValueError("Score edges must increase strictly from zero to one")
    categories = []
    for index, (low, high) in enumerate(zip(score_edges[:-1], score_edges[1:])):
        mask = ((frame.score >= low) &
                ((frame.score < high) if index < len(score_edges)-2
                 else (frame.score <= high)))
        part = frame.loc[mask]
        observed, _ = _histogram(part, mass_edges, "data")
        signal, signal_stat = _histogram(part, mass_edges, "signal")
        background, background_stat = _histogram(part, mass_edges, "background")
        if signal.sum() <= 0 or background.sum() <= 0:
            raise ValueError(
                f"Score category [{low:g}, {high:g}] has nonpositive MC yield")
        if np.any(background <= 0):
            raise ValueError(
                f"Score category [{low:g}, {high:g}] has an empty background mass bin")
        categories.append(TemplateCategory(
            f"{prefix}_{index}_{low:g}_{high:g}", observed, signal,
            background, background_stat, signal_stat))
    return categories


def _minimum_effective_background(categories):
    values = []
    for category in categories:
        variance = np.square(category.background_mc_stat)
        valid = variance > 0
        values.extend(np.square(category.background[valid])/variance[valid])
    return float(np.min(values)) if values else float("nan")


def _evaluate(model, strategy, categories, repeats, seed):
    likelihood = MassShapeLikelihood(categories, PUBLISHED_ZZ_UNCERTAINTIES)
    observations = [category.observed for category in categories]
    asimov = [category.signal + category.background for category in categories]
    expected = likelihood.discovery(asimov)
    observed = likelihood.discovery(observations)
    expected_interval = likelihood.profile_mu_interval(asimov, 1.)
    observed_interval = likelihood.profile_mu_interval(observations, 1.)
    statistical = (likelihood.statistical_z_ensemble(repeats, seed)
                   if repeats else None)
    return {
        "model": model,
        "strategy": strategy,
        "categories": len(categories),
        "mass_bins_per_category": len(categories[0].signal),
        "total_likelihood_bins": sum(len(item.signal) for item in categories),
        "signal_yield": sum(item.signal.sum() for item in categories),
        "background_yield": sum(item.background.sum() for item in categories),
        "observed_events": sum(item.observed.sum() for item in categories),
        "minimum_background_effective_events_per_bin":
            _minimum_effective_background(categories),
        "expected_Z": expected["Z"],
        "expected_mu_hat": expected["mu_hat"],
        "expected_mu_68_low": expected_interval["low"],
        "expected_mu_68_high": expected_interval["high"],
        "expected_sigma_mu": .5*(expected_interval["high"]-expected_interval["low"]),
        "observed_Z": observed["Z"],
        "observed_mu_hat": observed["mu_hat"],
        "observed_mu_68_low": observed_interval["low"],
        "observed_mu_68_high": observed_interval["high"],
        "expected_Z_stat_error": (None if statistical is None else
                                  statistical["expected"]["sigma"]),
        "observed_Z_stat_error": (None if statistical is None else
                                  statistical["observed"]["sigma"]),
    }


def _plot_metric(table, metric, path, ylabel, title, error=None):
    strategies = ["mass_only", "fixed_80pct_cut", "three_score_categories"]
    models = list(dict.fromkeys(table.model))
    x = np.arange(len(models), dtype=float)
    width = .19
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for index, strategy in enumerate(strategies):
        part = table[table.strategy == strategy].set_index("model").reindex(models)
        positions = x+(index-1)*width
        errors = None if error is None else part[error].to_numpy(float)
        ax.bar(positions, part[metric], width, yerr=errors, capsize=3,
               label=strategy.replace("_", " ").title())
    ax.set_xticks(x, models, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def compare_information_strategies(paths, output, fit_range=(105., 140.),
                                   mass_bin_width=5., target_efficiency=.80,
                                   statistical_repeats=0, seed=42):
    """Compare mass-only, hard-cut, categorized and joint mass-score fits."""
    if not paths:
        raise ValueError("At least one model prediction table is required")
    if not 0 < target_efficiency < 1:
        raise ValueError("Target signal efficiency must lie in (0, 1)")
    span = fit_range[1]-fit_range[0]
    bins = int(round(span/mass_bin_width))
    if bins < 5 or not math.isclose(bins*mass_bin_width, span, abs_tol=1e-8):
        raise ValueError("Fit range must contain an integer number of at least five bins")
    mass_edges = np.linspace(fit_range[0], fit_range[1], bins+1)
    output = new_output(output)
    rows = []
    operating_points = []
    for model_index, (model, path) in enumerate(paths.items()):
        frame = pd.read_csv(path)
        required = {"role", "mass", "weight", "score", "score_kind"}
        if required-set(frame):
            raise ValueError(f"{model} prediction table is missing required columns")
        signal_window = frame[(frame.role == "signal") &
                              (frame.mass >= 118.) & (frame.mass < 130.)]
        threshold = _weighted_quantile(
            signal_window.score.to_numpy(float),
            signal_window.weight.to_numpy(float), 1-target_efficiency)
        achieved = (signal_window.loc[signal_window.score > threshold, "weight"].sum()
                    / signal_window.weight.sum())
        operating_points.append({
            "model": model, "target_signal_efficiency": target_efficiency,
            "threshold": threshold, "achieved_signal_efficiency": achieved})

        strategies = {
            "mass_only": build_templates(frame, mass_edges, threshold=None),
            "fixed_80pct_cut": [item for item in build_templates(
                frame, mass_edges, threshold=threshold)
                if item.name == "score_pass"],
            "three_score_categories": _score_categories(
                frame, mass_edges, [0., .2, .6, 1.], "score_category"),
        }
        for strategy_index, (strategy, categories) in enumerate(strategies.items()):
            rows.append(_evaluate(
                model, strategy, categories, statistical_repeats,
                seed + 100*model_index + strategy_index))

    table = pd.DataFrame(rows)
    baselines = table[table.strategy == "mass_only"].set_index("model")
    table["expected_Z_relative_gain_percent"] = [
        100.*(row.expected_Z/baselines.loc[row.model, "expected_Z"]-1.)
        for row in table.itertuples()]
    table["expected_sigma_mu_improvement_percent"] = [
        100.*(1.-row.expected_sigma_mu/
              baselines.loc[row.model, "expected_sigma_mu"])
        for row in table.itertuples()]
    table.to_csv(output/"information_gain_comparison.csv", index=False)
    pd.DataFrame(operating_points).to_csv(
        output/"information_gain_operating_points.csv", index=False)
    _plot_metric(
        table, "expected_Z", output/"expected_significance_comparison.png",
        r"Expected local significance $Z_A$ [$\sigma$]",
        r"Information beyond $m_{4\ell}$: expected significance",
        "expected_Z_stat_error" if statistical_repeats else None)
    _plot_metric(
        table, "observed_Z", output/"observed_significance_comparison.png",
        r"Observed local significance $Z$ [$\sigma$]",
        r"Observed-data effect of retaining ML-score information",
        "observed_Z_stat_error" if statistical_repeats else None)
    _plot_metric(
        table, "expected_sigma_mu", output/"expected_mu_precision_comparison.png",
        r"Expected $\sigma_\mu$ (68% profile interval half-width)",
        r"Information beyond $m_{4\ell}$: signal-strength precision")
    with open(output/"information_gain_method.json", "w") as stream:
        json.dump({
            "question": "How much information does the OOF ML score add beyond m4l?",
            "fit_range_gev": list(fit_range),
            "mass_bin_width_gev": mass_bin_width,
            "target_signal_efficiency": target_efficiency,
            "strategy_definitions": {
                "mass_only": "inclusive m4l template",
                "fixed_80pct_cut": "m4l template after a model-specific OOF signal-efficiency cut",
                "three_score_categories": "simultaneous m4l templates in score [0,.2), [.2,.6), [.6,1]",
            },
            "score_boundaries": "fixed before observed-data evaluation; calibrated OOF signal-percentile scale",
            "primary_decision_metrics": ["expected_Z", "expected_sigma_mu"],
            "observed_results_used_for_method_selection": False,
            "normalization_uncertainties": PUBLISHED_ZZ_UNCERTAINTIES,
            "statistical_repeats": statistical_repeats,
        }, stream, indent=2, allow_nan=False)
    return output
