"""Guarded observed-count and sideband summaries from saved predictions."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import new_output, write_json
from .statistics import (_symmetric_stat_error, profile_likelihood_discovery,
                         weighted_yield)


def observed_count_result(n_observed, background, sigma_background,
                          fractional_systematic):
    values = [n_observed, background, sigma_background, fractional_systematic]
    if not all(math.isfinite(x) and x >= 0 for x in values) or background <= 0:
        return {"signal_excess": None, "profile_Z": None,
                "sigma_profile_Z": None,
                "profile_p_value_one_sided": None,
                "status": "undefined: nonpositive background or invalid inputs"}
    excess = n_observed - background
    sigma_observed = math.sqrt(n_observed)
    total_background_sigma = math.hypot(sigma_background,
                                       fractional_systematic*background)
    def evaluate(n, b):
        constraint = math.hypot(sigma_background,
                                fractional_systematic*b)
        return profile_likelihood_discovery(n, b, constraint)["profile_Z"]
    profile = profile_likelihood_discovery(
        n_observed, background, total_background_sigma)
    return {"signal_excess": excess,
            "background_constraint_sigma": total_background_sigma,
            **profile,
            "sigma_profile_Z": _symmetric_stat_error(
                evaluate, (n_observed, background),
                (sigma_observed, sigma_background)),
            "status": "one-bin profile likelihood; Z error propagates count statistics"}


def observed_methods(predictions, threshold, mass_window=(118., 130.),
                     sidebands=(90., 105., 140., 155.), systematic=.30):
    required = {"role", "weight", "mass", "score", "score_kind"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    data = predictions[predictions.role == "data"]
    mc_background = predictions[predictions.role == "background"]
    if data.empty or mc_background.empty:
        raise ValueError("Observed inference requires data and MC background rows")
    if not data.score_kind.eq("fold_assigned_calibrated").all():
        raise ValueError("Observed data must use fold-assigned calibrated scores")
    if not mc_background.score_kind.eq("out_of_fold_calibrated").all():
        raise ValueError("MC background must use calibrated out-of-fold scores")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    lo, hi = mass_window
    left_lo, left_hi, right_lo, right_hi = sidebands
    if not (left_lo < left_hi <= lo < hi <= right_lo < right_hi):
        raise ValueError("Sidebands must not overlap and must bracket the signal window")

    rows = []
    for stage, data_pass, mc_pass in (
        ("before_ml", np.ones(len(data), bool), np.ones(len(mc_background), bool)),
        ("after_ml", data.score.to_numpy() > threshold,
         mc_background.score.to_numpy() > threshold)):
        data_mass = data.mass.to_numpy(float)
        observed = int(np.count_nonzero(data_pass & (data_mass >= lo) & (data_mass < hi)))
        selected_mc = mc_background[mc_pass]
        b_mc, db_mc = weighted_yield(selected_mc.mass, selected_mc.weight, mass_window)
        rows.append({"stage": stage, "method": "mc_prediction", "N_observed": observed,
                     "background": b_mc, "sigma_background": db_mc,
                     **observed_count_result(observed, b_mc, db_mc, systematic)})

        sideband_count = int(np.count_nonzero(data_pass & (
            ((data_mass >= left_lo) & (data_mass < left_hi)) |
            ((data_mass >= right_lo) & (data_mass < right_hi)))))
        scale = (hi-lo) / ((left_hi-left_lo) + (right_hi-right_lo))
        b_sideband = sideband_count * scale
        db_sideband = math.sqrt(sideband_count) * scale
        rows.append({"stage": stage, "method": "sideband", "N_observed": observed,
                     "sideband_events": sideband_count, "background": b_sideband,
                     "sigma_background": db_sideband,
                     **observed_count_result(observed, b_sideband, db_sideband, systematic)})
    return pd.DataFrame(rows)


def analyze_observed(predictions_path, output, threshold, mass_window,
                     sidebands, systematic, fraction=1.0):
    if fraction != 1:
        raise ValueError("Observed inference is disabled for partial prepared inputs")
    predictions = pd.read_csv(predictions_path)
    results = observed_methods(predictions, threshold, mass_window, sidebands, systematic)
    output = new_output(output)
    results.to_csv(output/"observed_methods.csv", index=False)
    from .plots import save_observed_method_comparison
    save_observed_method_comparison(results, output/"observed_method_comparison.png",
                                    threshold, mass_window)
    from .plots import save_observed_profile_significance
    profile_display = results.copy()
    profile_display.insert(0, "model", "Classifier")
    save_observed_profile_significance(
        profile_display, output/"observed_profile_likelihood_comparison.png")
    write_json(output/"observed_summary.json", {
        "threshold": threshold, "mass_window_gev": list(mass_window),
        "sidebands_gev": list(sidebands),
        "background_fractional_systematic": systematic,
        # Convert to object dtype first; otherwise pandas keeps NaN in numeric
        # columns even when ``None`` is supplied to ``where``.  Strict JSON
        # serialization correctly rejects those non-finite values.
        "results": results.astype(object).where(
            pd.notna(results), None
        ).to_dict(orient="records"),
        "warnings": [
            "Z uses a one-bin profile likelihood; this is not a mass-shape fit.",
            "The sideband background estimate still requires closure validation.",
            "Data and MC use the same fold-local calibration; validate residual data/MC response differences before publication."
        ]})
    return output
