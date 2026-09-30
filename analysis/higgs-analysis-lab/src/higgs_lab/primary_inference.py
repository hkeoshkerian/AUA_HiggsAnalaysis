"""Primary score-pass binned likelihood in the Higgs mass window."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import new_output
from .shape_likelihood import (
    MassShapeLikelihood, build_templates,
    _save_mass_plot, _save_scan_plot, _save_significance_forest,
    _save_signal_strength_forest,
)
from .training import _weighted_quantile


def _slug(name):
    return name.lower().replace(" ", "-")


def fixed_signal_efficiency_thresholds(paths, target=.80,
                                       mass_window=(118., 130.)):
    """Derive model-specific cuts from weighted OOF signal MC only."""
    if not 0 < target < 1:
        raise ValueError("Target signal efficiency must lie in (0, 1)")
    thresholds = {}
    for name, path in paths.items():
        frame = pd.read_csv(path)
        signal = frame[(frame.role == "signal") &
                       (frame.mass >= mass_window[0]) &
                       (frame.mass < mass_window[1])]
        if signal.empty:
            raise ValueError(f"{name} has no signal MC in the operating-point window")
        if not signal.score_kind.eq("out_of_fold_calibrated").all():
            raise ValueError(f"{name} operating point requires OOF calibrated MC scores")
        quantile = _weighted_quantile(
            signal.score.to_numpy(float), signal.weight.to_numpy(float), 1-target)
        # The inference selection is expressed everywhere as score > cut. Move
        # the cut one representable float below the quantile so that a tied
        # score block at the boundary is included rather than discarded in
        # full. Exact target efficiency is generally impossible with ties.
        thresholds[name] = float(np.nextafter(quantile, -np.inf))
    return thresholds


def _fit_one(name, categories, edges, output, statistical_repeats, seed,
             uncertainties, selection):
    likelihood = MassShapeLikelihood(categories, uncertainties)
    observations = [category.observed for category in categories]
    observed = likelihood.discovery(observations)
    expected = likelihood.expected()
    interval68 = likelihood.profile_mu_interval(observations, 1.)
    interval95 = likelihood.profile_mu_interval(observations, 3.841458820694124)
    statistical = likelihood.statistical_z_ensemble(
        statistical_repeats, seed) if statistical_repeats else None
    scan_max = max(3., observed["mu_hat"]*2.2, expected["mu_hat"]*2.2)
    scan = likelihood.scan_mu(observations, np.linspace(0., scan_max, 121))
    slug = _slug(name)
    scan.to_csv(output/f"{slug}_likelihood_scan.csv", index=False)
    _save_scan_plot(scan, output/f"{slug}_likelihood_scan.png",
                    f"{name}: score-pass profile likelihood")
    fit = {**observed, "profiled_backgrounds": likelihood.profiled_backgrounds(
        observed["mu_hat"], observed["background_norm_pull_hat"], observations)}
    _save_mass_plot(categories, edges, fit, output/f"{slug}_score_pass_mass_fit.png",
                    f"{name}: score-pass primary mass fit")
    return {
        "model": name, "selection": selection,
        "observed_events": sum(c.observed.sum() for c in categories),
        "signal_yield": sum(c.signal.sum() for c in categories),
        "background_yield": sum(c.background.sum() for c in categories),
        "observed_q0": observed["q0"], "observed_Z": observed["Z"],
        "observed_p_value": observed["p_value_one_sided"],
        "expected_q0": expected["q0"], "expected_Z": expected["Z"],
        "expected_p_value": expected["p_value_one_sided"],
        "mu_hat": observed["mu_hat"],
        "mu_68_low": interval68["low"], "mu_68_high": interval68["high"],
        "mu_95_low": interval95["low"], "mu_95_high": interval95["high"],
        "mu_68_lower_at_boundary": interval68["lower_at_physical_boundary"],
        "mu_95_lower_at_boundary": interval95["lower_at_physical_boundary"],
        "background_norm_pull_hat": observed["background_norm_pull_hat"],
        "observed_Z_stat_low": None if statistical is None else statistical["observed"]["low"],
        "observed_Z_stat_high": None if statistical is None else statistical["observed"]["high"],
        "sigma_observed_Z_stat": None if statistical is None else statistical["observed"]["sigma"],
        "expected_Z_stat_low": None if statistical is None else statistical["expected"]["low"],
        "expected_Z_stat_high": None if statistical is None else statistical["expected"]["high"],
        "sigma_expected_Z_stat": None if statistical is None else statistical["expected"]["sigma"],
    }


def run_primary_inference(paths, thresholds, output, fit_range=(115., 130.),
                          bin_width=5., background_systematic=.30,
                          statistical_repeats=1000, seed=42,
                          include_no_ml=True):
    """Fit score-pass templates in an explicitly fixed mass window."""
    if not paths:
        raise ValueError("At least one model prediction table is required")
    if fit_range[1] <= fit_range[0] or bin_width <= 0:
        raise ValueError("Fit range and bin width must be positive and increasing")
    span = fit_range[1]-fit_range[0]
    count = int(round(span/bin_width))
    if count < 3 or not math.isclose(count*bin_width, span, abs_tol=1e-8):
        raise ValueError("Fit range must contain an integer number of at least three bins")
    if not math.isfinite(background_systematic) or background_systematic < 0:
        raise ValueError("Background systematic must be finite and nonnegative")
    edges = np.linspace(fit_range[0], fit_range[1], count+1)
    output = new_output(output)
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    nominal_rows = []
    operating_rows = []
    for name, frame in frames.items():
        signal = frame[frame.role == "signal"]
        in_window = signal[(signal.mass >= fit_range[0]) &
                           (signal.mass < fit_range[1])]
        threshold = thresholds[name]
        def efficiency(part):
            total = part.weight.sum()
            return (part.loc[part.score > threshold, "weight"].sum()/total
                    if total > 0 else np.nan)
        operating_rows.append({
            "model": name, "threshold": threshold,
            "all_signal_oof_efficiency": efficiency(signal),
            "signal_window_oof_efficiency": efficiency(in_window),
            "signal_window_low_gev": fit_range[0],
            "signal_window_high_gev": fit_range[1],
        })
    pd.DataFrame(operating_rows).to_csv(
        output/"primary_operating_points.csv", index=False)

    uncertainties = {"background_systematic": float(background_systematic)}
    if include_no_ml:
        reference = next(iter(frames.values()))
        inclusive = build_templates(reference, edges, threshold=None)
        nominal_rows.append(_fit_one(
            "No ML", inclusive, edges, output, statistical_repeats,
            seed+10_000, uncertainties, "inclusive reference"))
    for index, (name, frame) in enumerate(frames.items()):
        categories = build_templates(frame, edges, thresholds[name])
        score_pass = [c for c in categories if c.name == "score_pass"]
        nominal = _fit_one(
            name, score_pass, edges, output, statistical_repeats,
            seed+10_001+index, uncertainties,
            f"score pass only at {thresholds[name]:g}")
        nominal_rows.append(nominal)

    nominal = pd.DataFrame(nominal_rows)
    nominal.to_csv(output/"primary_score_pass_inference.csv", index=False)
    _save_significance_forest(
        nominal, "expected_Z", "expected_p_value",
        output/"primary_expected_significance_forest.png",
        r"Primary score-pass expected significance ($m_{4\ell}$ shape)",
        r"Expected local profile-likelihood significance $Z$ [$\sigma$]")
    _save_significance_forest(
        nominal, "observed_Z", "observed_p_value",
        output/"primary_observed_significance_forest.png",
        r"Primary score-pass observed significance ($m_{4\ell}$ shape)",
        r"Observed local profile-likelihood significance $Z$ [$\sigma$]",
        include_mu=True)
    _save_signal_strength_forest(
        nominal, output/"primary_signal_strength_forest.png")
    with open(output/"primary_inference_method.json", "w") as stream:
        json.dump({
            "primary_models": list(paths), "excluded_models": ["Logistic Regression"],
            "fit_range_gev": list(fit_range), "bin_width_gev": bin_width,
            "primary_category": "score_pass only",
            "score_fail_category": "excluded from primary inference",
            "thresholds": thresholds,
            "normalization_uncertainties": uncertainties,
            "statistical_repeats": statistical_repeats,
        }, stream, indent=2, allow_nan=False)
    return output
