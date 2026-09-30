"""Blinded robustness checks for separately retrained classifier predictions.

This module does not claim that sparse reducible MC is repaired by retraining.
It enforces a validation contract and reports when no defensible operating point
exists.  Observed events inside the signal window are never used.
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd

from .features import validate_frame
from .provenance import new_output, write_json
from .thresholds import scan_thresholds


ROBUST_FEATURES = (
    "dR_z2", "Phi1", "eta_z1_l2", "Phi", "theta1", "theta2",
    "met_phi", "eta_z2_l1", "Theta",
)


def effective_events(weights):
    weights = np.asarray(weights, float)
    total = weights.sum(); sumw2 = np.square(weights).sum()
    return float(total*total/sumw2) if sumw2 > 0 else 0.


def _yield(table):
    return float(table.weight.sum()), float(np.sqrt(np.square(table.weight).sum()))


def _ratio(a, da, b, db):
    if a <= 0 or b <= 0:
        return np.nan, np.nan
    value = a/b
    return value, abs(value)*math.sqrt((da/a)**2+(db/b)**2)


def _closure(frame, threshold, ranges):
    region = (((frame.mass >= ranges[0]) & (frame.mass < ranges[1])) |
              ((frame.mass >= ranges[2]) & (frame.mass < ranges[3])))
    table = frame[region]
    values = {}
    for source, role in (("data", "data"), ("background", "background"),
                         ("signal", "signal")):
        for category, passed in (("pass", True), ("fail", False)):
            part = table[(table.role == role) & ((table.score > threshold) == passed)]
            values[(source, category)] = _yield(part)
    # Subtract predicted signal leakage before comparing data to background.
    corrected = {}
    for category in ("pass", "fail"):
        nd, dd = values[("data", category)]
        ns, ds = values[("signal", category)]
        corrected[category] = (max(0., nd-ns), math.hypot(dd, ds))
    data_pf = _ratio(*corrected["pass"], *corrected["fail"])
    mc_pf = _ratio(*values[("background", "pass")],
                   *values[("background", "fail")])
    kappa = _ratio(*data_pf, *mc_pf)
    data_total = (corrected["pass"][0]+corrected["fail"][0],
                  math.hypot(corrected["pass"][1], corrected["fail"][1]))
    mc_total = (values[("background", "pass")][0]+values[("background", "fail")][0],
                math.hypot(values[("background", "pass")][1],
                           values[("background", "fail")][1]))
    norm = _ratio(*data_total, *mc_total)
    return {"sideband_normalization": norm[0],
            "sideband_normalization_error": norm[1],
            "pass_fail_kappa": kappa[0], "pass_fail_kappa_error": kappa[1],
            "data_pass_corrected": corrected["pass"][0],
            "data_fail_corrected": corrected["fail"][0],
            "mc_pass": values[("background", "pass")][0],
            "mc_fail": values[("background", "fail")][0]}


def audit_training_input(frame, features=ROBUST_FEATURES,
                         fit_range=(105., 140.), signal_window=(118., 130.)):
    validate_frame(frame, features)
    rows = []
    regions = {
        "all": np.ones(len(frame), bool),
        "fit": (frame.mass >= fit_range[0]) & (frame.mass < fit_range[1]),
        "signal_window": ((frame.mass >= signal_window[0]) &
                          (frame.mass < signal_window[1])),
        "sidebands": (((frame.mass >= fit_range[0]) &
                       (frame.mass < signal_window[0])) |
                      ((frame.mass >= signal_window[1]) &
                       (frame.mass < fit_range[1]))),
    }
    for region, mask in regions.items():
        for (role, sample), part in frame[mask].groupby(["role", "sample"]):
            total, error = _yield(part)
            rows.append({"region": region, "role": role, "sample": sample,
                         "rows": len(part), "yield": total,
                         "mc_stat_error": error,
                         "effective_events": effective_events(part.weight)})
    return pd.DataFrame(rows)


def validate_robust_models(predictions, output, efficiencies=(.6, .65, .7, .75,
                           .8, .85, .9), mass_window=(118., 130.),
                           sidebands=(105., 118., 130., 140.), systematic=.30,
                           minimum_background_neff=10., maximum_fold_cv=.25):
    """Validate OOF models and select working points without signal-region data."""
    output = new_output(output)
    scan_rows, process_rows, fold_rows, audit_rows = [], [], [], []
    choices = []
    for model, path in predictions.items():
        frame = pd.read_csv(path)
        validate_frame(frame, ROBUST_FEATURES)
        mc = frame[frame.role.isin(["signal", "background"])]
        if not mc.score_kind.eq("out_of_fold_calibrated").all():
            raise ValueError(f"{model} requires calibrated OOF MC scores")
        audit = audit_training_input(frame)
        audit.insert(0, "model", model); audit_rows.append(audit)
        for efficiency in efficiencies:
            threshold = round(1-float(efficiency), 12)
            metric = scan_thresholds(frame, [threshold], mass_window,
                                     systematic).iloc[0].to_dict()
            passed = mc.score > threshold
            selected_background = mc[(mc.role == "background") & passed &
                (mc.mass >= mass_window[0]) & (mc.mass < mass_window[1])]
            neff = effective_events(selected_background.weight)
            closure = _closure(frame, threshold, sidebands)
            fold_z = []
            for fold, part in mc.groupby("fold", sort=True):
                fold_metric = scan_thresholds(part, [threshold], mass_window,
                                              systematic).iloc[0]
                z = fold_metric.expected_profile_Z
                fold_z.append(z)
                fold_rows.append({"model": model, "fold": int(fold),
                                  "target_signal_efficiency": efficiency,
                                  "expected_profile_Z": z,
                                  "background_yield": fold_metric.B})
            finite = np.asarray([x for x in fold_z
                                 if x is not None and np.isfinite(x)], float)
            fold_cv = (float(finite.std(ddof=1)/finite.mean())
                       if len(finite) > 1 and finite.mean() > 0 else np.inf)
            kappa, dkappa = closure["pass_fail_kappa"], closure["pass_fail_kappa_error"]
            closure_ok = (np.isfinite(kappa) and np.isfinite(dkappa) and
                          abs(kappa-1.) <= 2*dkappa)
            eligible = bool(closure_ok and neff >= minimum_background_neff and
                            fold_cv <= maximum_fold_cv)
            row = {"model": model, "target_signal_efficiency": efficiency,
                   "calibrated_threshold": threshold, **metric,
                   "background_effective_events": neff,
                   "fold_Z_coefficient_of_variation": fold_cv,
                   "closure_compatible_2sigma": closure_ok,
                   "eligible": eligible, **closure}
            scan_rows.append(row)
            for sample, part in mc[passed].groupby("sample"):
                denominator = mc[mc["sample"] == sample]
                in_window = part[(part.mass >= mass_window[0]) &
                                 (part.mass < mass_window[1])]
                process_rows.append({"model": model,
                    "target_signal_efficiency": efficiency, "sample": sample,
                    "role": part.role.iloc[0], "selected_rows": len(part),
                    "total_weight": denominator.weight.sum(),
                    "selected_weight": part.weight.sum(),
                    "selection_efficiency": (part.weight.sum()/denominator.weight.sum()
                                             if denominator.weight.sum() > 0 else np.nan),
                    "mass_window_yield": in_window.weight.sum(),
                    "mass_window_effective_events": effective_events(in_window.weight)})
        options = [row for row in scan_rows if row["model"] == model and row["eligible"]]
        if options:
            chosen = sorted(options, key=lambda x:(-x["expected_profile_Z"],
                                                   x["target_signal_efficiency"]))[0]
            choices.append({"model": model, "status": "validated",
                            "target_signal_efficiency": chosen["target_signal_efficiency"],
                            "calibrated_threshold": chosen["calibrated_threshold"],
                            "selection_rule": "maximum expected OOF-MC Z among validated points"})
        else:
            choices.append({"model": model,
                            "status": "no operating point passed closure, effective-statistics, and fold-stability requirements",
                            "target_signal_efficiency": None,
                            "calibrated_threshold": None,
                            "selection_rule": "no fallback chosen"})
    scan = pd.DataFrame(scan_rows)
    pd.concat(audit_rows, ignore_index=True).to_csv(output/"training_input_audit.csv", index=False)
    scan.to_csv(output/"robust_operating_point_scan.csv", index=False)
    pd.DataFrame(process_rows).to_csv(output/"process_level_performance.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output/"fold_stability.csv", index=False)
    pd.DataFrame(choices).to_csv(output/"chosen_operating_points.csv", index=False)
    write_json(output/"robust_method.json", {
        "observed_signal_region_used": False,
        "features": list(ROBUST_FEATURES),
        "mass_window": list(mass_window), "sidebands": list(sidebands),
        "minimum_background_effective_events": minimum_background_neff,
        "maximum_fold_Z_coefficient_of_variation": maximum_fold_cv,
        "closure_requirement": "abs(kappa-1) <= 2 sigma_kappa",
        "background_contract": "Current MC benchmark only. A loose-not-tight data control sample is still required to validate reducible background inference.",
    })
    _plot_robust_scan(scan, output/"robust_operating_point_scan.png")
    return output


def _plot_robust_scan(scan, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for model, part in scan.groupby("model", sort=False):
        axes[0].plot(part.target_signal_efficiency, part.expected_profile_Z,
                     marker="o", label=model)
        axes[1].plot(part.target_signal_efficiency,
                     part.background_effective_events, marker="o", label=model)
        axes[2].errorbar(part.target_signal_efficiency, part.pass_fail_kappa,
                         yerr=part.pass_fail_kappa_error, marker="o", label=model)
    axes[0].set(ylabel="Expected profile Z", xlabel="Target signal efficiency")
    axes[1].axhline(10, color="black", linestyle="--", label="Minimum $N_{eff}$")
    axes[1].set(ylabel="Selected background $N_{eff}$ in 118–130 GeV",
                xlabel="Target signal efficiency", yscale="log")
    axes[2].axhline(1, color="black", linestyle="--")
    axes[2].set(ylabel="Sideband pass/fail closure $\\kappa$",
                xlabel="Target signal efficiency")
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes: ax.grid(alpha=.25)
    fig.suptitle("Robust retraining validation — signal region blinded")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
