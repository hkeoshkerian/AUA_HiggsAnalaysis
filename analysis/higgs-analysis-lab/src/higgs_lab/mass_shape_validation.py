"""Background and fit validation for the optional mass-shape likelihood."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .provenance import new_output
from .shape_likelihood import (MassShapeLikelihood, TemplateCategory,
                               PUBLISHED_ZZ_UNCERTAINTIES, build_templates,
                               _profile_bin_background)


def _slug(name):
    return name.lower().replace(" ", "-")


def _poisson_deviance(observed, expected):
    n = np.asarray(observed, float); lam = np.maximum(np.asarray(expected, float), 1e-12)
    terms = lam-n
    positive = n > 0
    terms[positive] += n[positive]*np.log(n[positive]/lam[positive])
    return float(2.*terms.sum())


def _slice_categories(categories, mask, names=None):
    result = []
    for category in categories:
        if names is not None and category.name not in names:
            continue
        signal_stat = (None if category.signal_mc_stat is None else
                       category.signal_mc_stat[mask])
        result.append(TemplateCategory(
            category.name, category.observed[mask], category.signal[mask],
            category.background[mask], category.background_mc_stat[mask],
            signal_stat))
    return result


def _gof(categories, norm_uncertainties, toys, seed):
    likelihood = MassShapeLikelihood(categories, norm_uncertainties)
    observations = [c.observed for c in categories]
    fit = likelihood.discovery(observations)
    backgrounds = likelihood.profiled_backgrounds(
        fit["mu_hat"], fit["background_norm_pull_hat"], observations)
    means = [fit["mu_hat"]*c.signal+b for c, b in zip(categories, backgrounds)]
    q_obs = sum(_poisson_deviance(n, lam) for n, lam in zip(observations, means))
    if toys <= 0:
        return q_obs, None, None
    rng = np.random.default_rng(seed)
    q_toys = np.empty(toys)
    for repeat in range(toys):
        toy = [rng.poisson(np.maximum(x, 0.)).astype(float) for x in means]
        toy_fit = likelihood.discovery(toy)
        toy_b = likelihood.profiled_backgrounds(
            toy_fit["mu_hat"], toy_fit["background_norm_pull_hat"], toy)
        toy_means = [toy_fit["mu_hat"]*c.signal+b
                     for c, b in zip(categories, toy_b)]
        q_toys[repeat] = sum(_poisson_deviance(n, lam)
                             for n, lam in zip(toy, toy_means))
    p = (np.count_nonzero(q_toys >= q_obs)+1.)/(toys+1.)
    return q_obs, float(p), q_toys


def _ratio(value_a, error_a, value_b, error_b):
    if value_a <= 0 or value_b <= 0:
        return np.nan, np.nan
    value = value_a/value_b
    error = value*math.hypot(error_a/value_a, error_b/value_b)
    return value, error


def _sideband_closure(frame, threshold, sidebands):
    mass = frame.mass.to_numpy(float)
    sb = (((mass >= sidebands[0]) & (mass < sidebands[1])) |
          ((mass >= sidebands[2]) & (mass < sidebands[3])))
    rows = []
    totals = {}
    for source, role in (("data", "data"), ("mc", "background")):
        role_mask = frame.role.eq(role).to_numpy()
        for region, score_mask in (("fail", frame.score.to_numpy(float) <= threshold),
                                   ("pass", frame.score.to_numpy(float) > threshold)):
            chosen = sb & role_mask & score_mask
            if source == "data":
                value = float(chosen.sum()); error = math.sqrt(value)
            else:
                weights = frame.weight.to_numpy(float)[chosen]
                value = float(weights.sum()); error = float(np.sqrt(np.square(weights).sum()))
            totals[(source, region)] = (value, error)
            rows.append({"source": source, "region": region,
                         "yield": value, "stat_error": error})
    data_ratio = _ratio(*totals[("data", "pass")], *totals[("data", "fail")])
    mc_ratio = _ratio(*totals[("mc", "pass")], *totals[("mc", "fail")])
    kappa = _ratio(*data_ratio, *mc_ratio)
    migration = (max(abs(kappa[0]-1.), kappa[1])
                 if np.isfinite(kappa[0]) and np.isfinite(kappa[1]) else np.nan)
    return rows, {"data_pass_fail": data_ratio[0], "data_pass_fail_error": data_ratio[1],
                  "mc_pass_fail": mc_ratio[0], "mc_pass_fail_error": mc_ratio[1],
                  "kappa": kappa[0], "kappa_error": kappa[1],
                  "migration_fractional_uncertainty": migration}


def _adaptive_edges(frame, threshold, base_edges, minimum_pass_background=2.):
    background = frame[(frame.role == "background") & (frame.score > threshold)]
    yields = np.histogram(background.mass, base_edges, weights=background.weight)[0]
    edges = [float(base_edges[0])]; running = 0.
    for index, value in enumerate(yields):
        running += max(0., float(value))
        if running >= minimum_pass_background:
            edges.append(float(base_edges[index+1])); running = 0.
    if edges[-1] != float(base_edges[-1]):
        if len(edges) > 1:
            edges[-1] = float(base_edges[-1])
        else:
            edges.append(float(base_edges[-1]))
    return np.asarray(sorted(set(edges)), float)


class _MigrationLikelihood(MassShapeLikelihood):
    """Pass/fail likelihood with an anti-correlated migration nuisance."""
    def __init__(self, categories, norm_uncertainties, migration_uncertainty):
        super().__init__(categories, norm_uncertainties)
        self.migration_uncertainty = max(0., float(migration_uncertainty))

    def _nll_vector(self, mu, eta, xi, observations):
        if mu < 0 or not all(math.isfinite(x) for x in (mu, eta, xi)):
            return float("inf")
        total = .5*eta*eta + .5*xi*xi
        global_scale = math.exp(self.relative_background_norm*eta)
        for category, observed in zip(self.categories, observations):
            direction = 1. if category.name == "score_pass" else -1.
            migration_scale = math.exp(direction*self.migration_uncertainty*xi)
            scale = global_scale*migration_scale
            for n, s, b0, db0 in zip(observed, category.signal,
                                      category.background,
                                      category.background_mc_stat):
                signal = mu*max(0., s); center = max(0., b0*scale)
                sigma = max(0., db0*scale)
                b = _profile_bin_background(float(n), signal, center, sigma)
                lam = max(1e-12, signal+b)
                total += lam-(float(n)*math.log(lam) if n > 0 else 0.)
                if sigma > 0: total += .5*((b-center)/sigma)**2
        return total

    def discovery(self, observations):
        observations = [np.asarray(x, float) for x in observations]
        null = minimize(lambda x: self._nll_vector(0., x[0], x[1], observations),
                        [0., 0.], method="L-BFGS-B", bounds=[(-8., 8.), (-8., 8.)])
        free = minimize(lambda x: self._nll_vector(x[0], x[1], x[2], observations),
                        [1., 0., 0.], method="L-BFGS-B",
                        bounds=[(0., 20.), (-8., 8.), (-8., 8.)])
        if not null.success or not free.success:
            raise RuntimeError("Migration-nuisance likelihood fit failed")
        q0 = max(0., 2.*(null.fun-free.fun)) if free.x[0] > 0 else 0.
        return {"Z": math.sqrt(q0), "mu_hat": float(free.x[0]),
                "global_pull": float(free.x[1]), "migration_pull": float(free.x[2])}

    def expected(self):
        return self.discovery([c.signal+c.background for c in self.categories])


def _residual_table(categories, edges, relative_norm):
    rows = []
    centers = .5*(edges[:-1]+edges[1:])
    for category in categories:
        for index, center in enumerate(centers):
            n = category.observed[index]; b = category.background[index]
            db = category.background_mc_stat[index]
            denominator = math.sqrt(max(0., n)+db*db+(relative_norm*b)**2)
            rows.append({"category": category.name, "bin_low": edges[index],
                         "bin_high": edges[index+1], "bin_center": center,
                         "observed": n, "background": b,
                         "background_mc_stat": db, "residual": n-b,
                         "pull": (n-b)/denominator if denominator > 0 else np.nan})
    return pd.DataFrame(rows)


def _plot_residuals(table, signal_window, path, title):
    categories = table.category.unique()
    fig, axes = plt.subplots(len(categories), 1, figsize=(11, 3.8*len(categories)),
                             sharex=True, squeeze=False)
    for ax, category in zip(axes[:, 0], categories):
        part = table[table.category == category]
        width = part.bin_high-part.bin_low
        ax.axhline(0., color="black", linewidth=1)
        ax.axhspan(-2., 2., color="#56B4E9", alpha=.12)
        ax.axvspan(*signal_window, color="#F0E442", alpha=.16,
                   label="Signal window excluded from closure")
        ax.bar(part.bin_center, part.pull, width=.85*width,
               color=np.where(np.abs(part.pull) >= 2., "#D55E00", "#0072B2"))
        ax.set_ylabel("Pre-fit pull")
        ax.set_title(category.replace("_", " ").title())
        ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend(fontsize=9)
    axes[-1, 0].set_xlabel(r"$m_{4\ell}$ [GeV]")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_score(frame, sidebands, path, title):
    mass = frame.mass.to_numpy(float)
    sb = (((mass >= sidebands[0]) & (mass < sidebands[1])) |
          ((mass >= sidebands[2]) & (mass < sidebands[3])))
    data = frame[sb & frame.role.eq("data")]
    mc = frame[sb & frame.role.eq("background")]
    edges = np.linspace(0., 1., 11); centers = .5*(edges[:-1]+edges[1:])
    d = np.histogram(data.score, edges)[0].astype(float)
    b = np.histogram(mc.score, edges, weights=mc.weight)[0]
    db = np.sqrt(np.histogram(mc.score, edges, weights=mc.weight**2)[0])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].stairs(b, edges, color="#D55E00", linewidth=2, label="Background MC")
    axes[0].errorbar(centers, d, yerr=np.sqrt(d), fmt="o", color="black", label="Data")
    axes[0].set_ylabel("Events / score bin"); axes[0].legend(); axes[0].grid(alpha=.2)
    ratio = np.divide(d, b, out=np.full_like(d, np.nan), where=b > 0)
    ratio_error = np.divide(np.sqrt(d), b, out=np.full_like(d, np.nan), where=b > 0)
    axes[1].axhline(1., color="black", linewidth=1)
    axes[1].errorbar(centers, ratio, yerr=ratio_error, fmt="o", color="#0072B2")
    axes[1].set(xlabel="Calibrated classifier score", ylabel="Data / MC")
    axes[1].grid(alpha=.2); fig.suptitle(title, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_components(frame, threshold, edges, path, title):
    selected = frame[(frame.role == "background") & (frame.score > threshold)]
    samples = sorted(selected["sample"].astype(str).unique())
    values = [np.histogram(selected.loc[selected["sample"].astype(str).eq(sample), "mass"],
                           edges, weights=selected.loc[selected["sample"].astype(str).eq(sample), "weight"])[0]
              for sample in samples]
    fig, ax = plt.subplots(figsize=(11, 6))
    if values:
        ax.hist([.5*(edges[:-1]+edges[1:])]*len(values), bins=edges,
                weights=values, stacked=True, label=samples)
    ax.set(xlabel=r"$m_{4\ell}$ [GeV]", ylabel="Weighted background MC",
           title=title); ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def validate_mass_shapes(paths, thresholds, output, fit_range=(105., 140.),
                         bin_width=2.5, signal_window=(118., 130.),
                         sidebands=(105., 118., 130., 140.), gof_toys=300,
                         seed=42):
    output = new_output(output)
    span = fit_range[1]-fit_range[0]
    count = int(round(span/bin_width))
    edges = np.linspace(fit_range[0], fit_range[1], count+1)
    relative_norm = math.sqrt(sum(x*x for x in PUBLISHED_ZZ_UNCERTAINTIES.values()))
    rows_gof, rows_closure, rows_ablation, rows_binning = [], [], [], []
    for model_index, (name, path) in enumerate(paths.items()):
        frame = pd.read_csv(path)
        categories = build_templates(frame, edges, thresholds[name])
        slug = _slug(name)
        residuals = _residual_table(categories, edges, relative_norm)
        residuals.to_csv(output/f"{slug}_prefit_residuals.csv", index=False)
        _plot_residuals(residuals, signal_window,
                        output/f"{slug}_prefit_residuals.png",
                        f"{name}: pre-fit data minus background MC")
        _plot_score(frame, sidebands, output/f"{slug}_sideband_score_validation.png",
                    f"{name}: classifier score in background sidebands")
        _plot_components(frame, thresholds[name], edges,
                         output/f"{slug}_background_components.png",
                         f"{name}: score-pass MC background composition")

        closure = None
        closure_definitions = {
            "combined": sidebands,
            "lower": (sidebands[0], sidebands[1], sidebands[3], sidebands[3]),
            "upper": (sidebands[0], sidebands[0], sidebands[2], sidebands[3]),
        }
        for sideband_name, definition in closure_definitions.items():
            closure_yields, closure_result = _sideband_closure(
                frame, thresholds[name], definition)
            for item in closure_yields:
                rows_closure.append({"model": name,
                                     "sideband_region": sideband_name, **item})
            rows_closure.append({"model": name, "sideband_region": sideband_name,
                                 "source": "closure", "region": "combined",
                                 "yield": np.nan, "stat_error": np.nan,
                                 **closure_result})
            if sideband_name == "combined":
                closure = closure_result

        centers = .5*(edges[:-1]+edges[1:])
        sideband_mask = ((centers < signal_window[0]) | (centers >= signal_window[1]))
        gof_jobs = {
            "full": categories,
            "pass": _slice_categories(categories, np.ones(count, bool), {"score_pass"}),
            "fail": _slice_categories(categories, np.ones(count, bool), {"score_fail"}),
            "sidebands_pass_fail": _slice_categories(categories, sideband_mask),
            "sidebands_pass": _slice_categories(categories, sideband_mask, {"score_pass"}),
            "sidebands_fail": _slice_categories(categories, sideband_mask, {"score_fail"}),
        }
        for job_index, (region, cats) in enumerate(gof_jobs.items()):
            q, p, toy_values = _gof(cats, PUBLISHED_ZZ_UNCERTAINTIES,
                                    gof_toys, seed+100*model_index+job_index)
            rows_gof.append({"model": name, "region": region,
                             "gof_deviance": q, "toy_p_value": p,
                             "toy_repeats": gof_toys})
            if toy_values is not None:
                pd.DataFrame({"q_gof": toy_values}).to_csv(
                    output/f"{slug}_{region}_gof_toys.csv", index=False)

        for width in (2.5, 5., 10.):
            if not math.isclose(span/width, round(span/width)):
                continue
            local_edges = np.linspace(fit_range[0], fit_range[1], int(round(span/width))+1)
            local = build_templates(frame, local_edges, thresholds[name])
            likelihood = MassShapeLikelihood(local, PUBLISHED_ZZ_UNCERTAINTIES)
            observed = likelihood.discovery([c.observed for c in local])
            expected = likelihood.expected()
            rows_binning.append({"model": name, "binning": f"fixed_{width:g}_gev",
                                 "observed_Z": observed["Z"],
                                 "expected_Z": expected["Z"],
                                 "mu_hat": observed["mu_hat"]})
        adaptive_edges = _adaptive_edges(frame, thresholds[name], edges)
        if len(adaptive_edges) >= 3:
            local = build_templates(frame, adaptive_edges, thresholds[name])
            likelihood = MassShapeLikelihood(local, PUBLISHED_ZZ_UNCERTAINTIES)
            observed = likelihood.discovery([c.observed for c in local])
            expected = likelihood.expected()
            rows_binning.append({"model": name,
                                 "binning": "adaptive_min_2_pass_background",
                                 "observed_Z": observed["Z"],
                                 "expected_Z": expected["Z"],
                                 "mu_hat": observed["mu_hat"],
                                 "edges": ";".join(f"{x:g}" for x in adaptive_edges)})

        mass_pass = (frame.mass >= signal_window[0]) & (frame.mass < signal_window[1])
        pass_sel = frame.score > thresholds[name]
        one_bin_edges = np.asarray(signal_window, float)
        one_bin = build_templates(frame, one_bin_edges, thresholds[name])
        pass_one = [c for c in one_bin if c.name == "score_pass"]
        variants = [
            ("A_one_bin_pass_30pct", pass_one, {"background": .30}),
            ("B_one_bin_pass_6p7pct", pass_one, PUBLISHED_ZZ_UNCERTAINTIES),
            ("C_binned_pass_6p7pct", [c for c in categories if c.name == "score_pass"], PUBLISHED_ZZ_UNCERTAINTIES),
            ("D_binned_pass_fail_6p7pct", categories, PUBLISHED_ZZ_UNCERTAINTIES),
        ]
        for variant, cats, uncertainties in variants:
            likelihood = MassShapeLikelihood(cats, uncertainties)
            rows_ablation.append({"model": name, "variant": variant,
                                  "expected_Z": likelihood.expected()["Z"],
                                  "observed_Z": likelihood.discovery([c.observed for c in cats])["Z"],
                                  "migration_fractional_uncertainty": 0.})
        migration = closure["migration_fractional_uncertainty"]
        if np.isfinite(migration):
            migration_likelihood = _MigrationLikelihood(
                categories, PUBLISHED_ZZ_UNCERTAINTIES, migration)
            expected = migration_likelihood.expected()
            observed = migration_likelihood.discovery([c.observed for c in categories])
            rows_ablation.append({"model": name,
                                  "variant": "E_binned_pass_fail_with_sideband_migration",
                                  "expected_Z": expected["Z"], "observed_Z": observed["Z"],
                                  "migration_fractional_uncertainty": migration,
                                  "migration_pull": observed["migration_pull"]})

    gof = pd.DataFrame(rows_gof); closure = pd.DataFrame(rows_closure)
    ablation = pd.DataFrame(rows_ablation); binning = pd.DataFrame(rows_binning)
    gof.to_csv(output/"goodness_of_fit_summary.csv", index=False)
    closure.to_csv(output/"sideband_pass_fail_closure.csv", index=False)
    ablation.to_csv(output/"likelihood_ablation.csv", index=False)
    binning.to_csv(output/"binning_stability.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, part in ablation.groupby("model", sort=False):
        ax.plot(part.variant, part.expected_Z, "o-", label=name)
    ax.set(ylabel=r"Expected profile $Z_A$", xlabel="Likelihood variant",
           title="Likelihood-choice ablation")
    ax.tick_params(axis="x", rotation=28); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output/"likelihood_ablation.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, part in binning.groupby("model", sort=False):
        ax.plot(part.binning, part.observed_Z, "o-", label=name)
    ax.set(ylabel="Observed profile Z", xlabel="Binning",
           title="Mass-binning stability")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output/"binning_stability.png", dpi=220); plt.close(fig)

    closure_summary = closure[(closure.source == "closure") &
                              (closure.sideband_region == "combined")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(np.arange(len(closure_summary)), closure_summary.kappa,
                yerr=closure_summary.kappa_error, fmt="o", capsize=5)
    ax.axhline(1., color="black", linestyle="--")
    ax.set_xticks(np.arange(len(closure_summary)), closure_summary.model,
                  rotation=25, ha="right")
    ax.set(ylabel=r"$\kappa=(P/F)_{data}/(P/F)_{MC}$",
           title="Sideband pass/fail closure")
    ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(output/"sideband_pass_fail_closure.png", dpi=220); plt.close(fig)

    metadata = {"fit_range": list(fit_range), "signal_window": list(signal_window),
                "sidebands": list(sidebands), "bin_width": bin_width,
                "gof_toys": gof_toys, "seed": seed,
                "migration_rule": "max(abs(kappa-1), statistical error on kappa); signal-region data excluded",
                "caution": "Validation suite; no range, binning, or nuisance is selected by observed signal significance."}
    with open(output/"validation_method.json", "w") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    return output
