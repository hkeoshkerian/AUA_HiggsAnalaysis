"""Seed and fold-count stability studies."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .features import validate_frame
from .provenance import new_output, sha256, write_json
from .statistics import summarize_mc
from .training import train_models


def run_stability(frame, config, seeds, folds):
    validate_frame(frame, config.training.features)
    seeds, folds = list(seeds), list(folds)
    if not seeds or not folds or any(type(x) is not int for x in seeds + folds):
        raise ValueError("Seeds and folds must be nonempty integer lists")
    if any(x < 0 or x >= 2**32 for x in seeds) or any(x < 2 for x in folds):
        raise ValueError("Seeds must be unsigned 32-bit values and folds must be >= 2")
    rows = []
    for fold_count in folds:
        for seed in seeds:
            current = replace(config, training=replace(
                config.training, seed=seed, folds=fold_count))
            result = train_models(frame, current)
            selected = result.predictions[
                result.predictions.score > current.training.threshold]
            summary = summarize_mc(
                selected, current.statistics.mass_window_gev,
                current.statistics.background_fractional_systematic,
                current.data.fraction)
            rows.append({"seed": seed, "folds": fold_count,
                         "weighted_oof_auc": result.oof_auc,
                         "threshold": current.training.threshold,
                         "S": summary["S"], "B": summary["B"],
                         "expected_profile_Z": summary["expected_profile_Z"],
                         "sigma_expected_profile_Z": summary["sigma_expected_profile_Z"]})
    return pd.DataFrame(rows)


def _plot_stability(results, output):
    import matplotlib.pyplot as plt
    for metric, ylabel in (("weighted_oof_auc", "Weighted OOF AUC"),
                           ("expected_profile_Z", "Expected profile-likelihood Z")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for folds, group in results.groupby("folds"):
            ordered = group.sort_values("seed")
            ax.plot(ordered.seed, ordered[metric], marker="o", label=f"{folds} folds")
        ax.set(xlabel="Random seed", ylabel=ylabel, title=f"{ylabel} stability")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(output/f"{metric}_stability.png", dpi=150); plt.close(fig)


def analyze_stability(prepared, output, config, seeds, folds):
    prepared = Path(prepared)
    manifest = json.loads((prepared/"manifest.json").read_text())
    csv_path = prepared/"features.csv"
    if sha256(csv_path) != manifest.get("features_sha256"):
        raise ValueError("Prepared feature checksum mismatch")
    frame = pd.read_csv(csv_path)
    results = run_stability(frame, config, seeds, folds)
    output = new_output(output)
    results.to_csv(output/"stability.csv", index=False)
    write_json(output/"stability_summary.json", {
        "model": config.training.model, "seeds": list(seeds), "folds": list(folds),
        "threshold": config.training.threshold,
        "warning": "Stability diagnostics are not a substitute for an untouched final evaluation set."
    })
    _plot_stability(results, output)
    return output
