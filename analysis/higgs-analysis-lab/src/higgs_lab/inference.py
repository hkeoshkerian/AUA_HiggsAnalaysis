"""Guarded observed-count and sideband summaries from saved predictions."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import new_output, write_json
from .statistics import weighted_yield


def observed_count_result(n_observed, background, sigma_background,
                          fractional_systematic):
    values = [n_observed, background, sigma_background, fractional_systematic]
    if not all(math.isfinite(x) and x >= 0 for x in values) or background <= 0:
        return {"signal_excess": None, "Z": None, "sigma_Z": None,
                "p_value_one_sided": None, "status": "undefined: nonpositive background or invalid inputs"}
    excess = n_observed - background
    sigma_observed = math.sqrt(n_observed)
    k = fractional_systematic
    denominator = background + (k*background)**2
    z = excess / math.sqrt(denominator)
    variance = (sigma_observed**2 / denominator +
                excess**2 * (1 + 2*k*k*background)**2 * sigma_background**2 /
                (4*denominator**3))
    return {"signal_excess": excess, "Z": z, "sigma_Z": math.sqrt(variance),
            "p_value_one_sided": .5*math.erfc(z/math.sqrt(2)),
            "status": "counting approximation; not a calibrated likelihood result"}


def observed_methods(predictions, threshold, mass_window=(110., 135.),
                     sidebands=(90., 105., 140., 155.), systematic=.30):
    required = {"role", "weight", "mass", "score", "score_kind"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    data = predictions[predictions.role == "data"]
    mc_background = predictions[predictions.role == "background"]
    if data.empty or mc_background.empty:
        raise ValueError("Observed inference requires data and MC background rows")
    if not data.score_kind.eq("fold_ensemble_mean").all():
        raise ValueError("Observed data must use fold-ensemble scores")
    if not mc_background.score_kind.eq("out_of_fold").all():
        raise ValueError("MC background must use out-of-fold scores")
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
    write_json(output/"observed_summary.json", {
        "threshold": threshold, "mass_window_gev": list(mass_window),
        "sidebands_gev": list(sidebands),
        "background_fractional_systematic": systematic,
        "results": results.where(pd.notna(results), None).to_dict(orient="records"),
        "warnings": [
            "Counting and sideband approximations are not a calibrated likelihood fit.",
            "Fold-ensemble data scores and OOF MC scores require response validation before publication."
        ]})
    return output
