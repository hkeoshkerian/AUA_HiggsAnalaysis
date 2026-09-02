"""MC-only threshold diagnostics for out-of-fold classifier scores."""
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import new_output, write_json
from .statistics import expected_count_proxy, weighted_yield


def scan_thresholds(predictions, thresholds, mass_window=(110.0, 135.0),
                    background_systematic=.30):
    required = {"role", "label", "weight", "mass", "score", "score_kind"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    mc = predictions[predictions.role != "data"].copy()
    if mc.empty or not mc.score_kind.eq("out_of_fold").all():
        raise ValueError("Threshold selection requires MC out-of-fold predictions")
    numeric = mc[["label", "weight", "mass", "score"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (mc.weight < 0).any():
        raise ValueError("Threshold scanning requires finite, nonnegative MC inputs")
    if set(mc.label.unique()) != {0, 1}:
        raise ValueError("Threshold scanning requires both signal and background")
    thresholds = np.asarray(list(thresholds), dtype=float)
    if thresholds.size == 0 or not np.isfinite(thresholds).all() or np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("Thresholds must be finite values in (0, 1)")

    total_signal = mc.loc[mc.label == 1, "weight"].sum()
    total_background = mc.loc[mc.label == 0, "weight"].sum()
    rows = []
    for threshold in thresholds:
        passed = mc.score > threshold
        signal = mc[(mc.label == 1) & passed]
        background = mc[(mc.label == 0) & passed]
        s, ds = weighted_yield(signal.mass, signal.weight, mass_window)
        b, db = weighted_yield(background.mass, background.weight, mass_window)
        significance = expected_count_proxy(s, b, ds, db, background_systematic)
        tp = mc.loc[(mc.label == 1) & passed, "weight"].sum()
        tn = mc.loc[(mc.label == 0) & ~passed, "weight"].sum()
        rows.append({
            "threshold": float(threshold), "S": s, "B": b,
            "sigma_S_mc": ds, "sigma_B_mc": db,
            "Z_proxy": significance["Z_proxy"],
            "sigma_Z_proxy": significance["sigma_Z_proxy"],
            "signal_efficiency": float(tp / total_signal),
            "background_rejection": float(tn / total_background),
            "weighted_accuracy": float((tp + tn) / (total_signal + total_background)),
            "selected_mc_events": int(passed.sum()),
        })
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def optimal_threshold(scan):
    valid = scan.dropna(subset=["Z_proxy"])
    if valid.empty:
        return None
    # Deterministic tie-break: prefer the lower threshold.
    return valid.sort_values(["Z_proxy", "threshold"], ascending=[False, True]).iloc[0].to_dict()


def _save_plots(scan, predictions, output, working_point=.65):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = scan.dropna(subset=["Z_proxy", "sigma_Z_proxy"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(5., color="green", linestyle="--", linewidth=1.5,
               alpha=.7, label="Discovery (5σ)")
    ax.axhline(3., color="orange", linestyle="--", linewidth=1.5,
               alpha=.7, label="Evidence (3σ)")
    ax.axvline(working_point, color="red", linestyle=":", linewidth=1.5,
               alpha=.7, label=f"Optimal: {working_point:.2f}")
    ax.errorbar(finite.threshold, finite.Z_proxy,
                yerr=finite.sigma_Z_proxy, fmt="o-", color="blue",
                capsize=3, capthick=1, markersize=6, linewidth=2,
                label="Significance Z (MC)")
    if len(finite):
        selected = finite.iloc[(finite.threshold-working_point).abs().argmin()]
        ax.plot(working_point, selected.Z_proxy, "o", color="red", markersize=10)
        upper = max(float(finite.Z_proxy.max())*1.2, .5)
    else:
        upper = 1.
    ax.set(xlim=(0, 1), ylim=(0, upper),
           xlabel="BDT Score Threshold", ylabel="Significance Z (σ)",
           title="MC Validation: Significance vs. BDT Threshold")
    ax.grid(True, alpha=.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout(); fig.savefig(output/"significance_vs_threshold.png",
                                    dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scan.threshold, scan.signal_efficiency, label="Signal efficiency")
    ax.plot(scan.threshold, scan.background_rejection, label="Background rejection")
    ax.plot(scan.threshold, scan.weighted_accuracy, label="Weighted accuracy")
    ax.set(xlabel="Classifier threshold", ylabel="Fraction", ylim=(0, 1.03),
           title="Threshold diagnostics — OOF MC only")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output/"efficiency_rejection.png", dpi=150); plt.close(fig)

    mc = predictions[predictions.role != "data"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for role, color in (("background", "tab:blue"), ("signal", "tab:red")):
        part = mc[mc.role == role]
        ax.hist(part.score, bins=np.linspace(0, 1, 41), weights=part.weight,
                histtype="step", linewidth=1.8, label=role.title(), color=color)
    ax.set(xlabel="Out-of-fold score", ylabel="Weighted MC events",
           title="Classifier score distribution")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output/"score_distribution.png", dpi=150); plt.close(fig)


def analyze_thresholds(predictions_path, output, thresholds, mass_window,
                       background_systematic, working_point=.65):
    predictions = pd.read_csv(predictions_path)
    scan = scan_thresholds(predictions, thresholds, mass_window, background_systematic)
    best = optimal_threshold(scan)
    output = new_output(output)
    scan.to_csv(output/"threshold_scan.csv", index=False)
    write_json(output/"threshold_summary.json", {
        "selection_source": "MC out-of-fold predictions only",
        "mass_window_gev": list(mass_window),
        "background_fractional_systematic": background_systematic,
        "working_point": working_point,
        "optimal": best,
        "warning": "Do not optimize a threshold on observed data or final test data."
    })
    _save_plots(scan, predictions, output, working_point)
    return output
