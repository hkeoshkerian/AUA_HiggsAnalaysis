"""ATLAS-inspired binned mass-shape profile-likelihood inference.

This is deliberately independent of the legacy one-bin significance code.  It
uses observed m4l bins, signal/background MC templates, a correlated log-normal
ZZ* normalization nuisance, and Gaussian-constrained weighted-MC statistical
uncertainties profiled independently in every bin.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import norm

from .provenance import new_output


PUBLISHED_ZZ_UNCERTAINTIES = {
    "qcd_scale": 0.04,
    "pdf": 0.02,
    "parton_shower": 0.05,
}


@dataclass(frozen=True)
class TemplateCategory:
    name: str
    observed: np.ndarray
    signal: np.ndarray
    background: np.ndarray
    background_mc_stat: np.ndarray
    signal_mc_stat: np.ndarray | None = None


def _histogram(frame, edges, role):
    selected = frame[frame.role == role]
    if role == "data":
        counts = np.histogram(selected.mass.to_numpy(float), bins=edges)[0]
        return counts.astype(float), np.sqrt(counts.astype(float))
    weights = selected.weight.to_numpy(float)
    mass = selected.mass.to_numpy(float)
    counts = np.histogram(mass, bins=edges, weights=weights)[0]
    sumw2 = np.histogram(mass, bins=edges, weights=weights * weights)[0]
    return counts.astype(float), np.sqrt(sumw2.astype(float))


def build_templates(frame, edges, threshold=None):
    """Build inclusive or pass/fail templates from one prediction table."""
    required = {"role", "mass", "weight", "score", "score_kind"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if not frame.role.isin(["signal", "background", "data"]).all():
        raise ValueError("Prediction table contains an unknown role")
    if (frame.loc[frame.role != "data", "weight"] < 0).any():
        raise ValueError("Mass-shape likelihood currently requires nonnegative MC weights")
    data = frame[frame.role == "data"]
    mc = frame[frame.role != "data"]
    if data.empty or mc.empty:
        raise ValueError("Mass-shape inference requires observed data and MC")
    if not data.score_kind.eq("fold_assigned_calibrated").all():
        raise ValueError("Observed data must use fold-assigned calibrated scores")
    if not mc.score_kind.eq("out_of_fold_calibrated").all():
        raise ValueError("MC must use calibrated out-of-fold scores")

    selections = [("inclusive", np.ones(len(frame), dtype=bool))]
    if threshold is not None:
        if not 0 < threshold < 1:
            raise ValueError("Threshold must lie strictly between zero and one")
        score = frame.score.to_numpy(float)
        selections = [("score_fail", score <= threshold),
                      ("score_pass", score > threshold)]

    categories = []
    for name, mask in selections:
        part = frame.loc[mask]
        observed, _ = _histogram(part, edges, "data")
        signal, signal_stat = _histogram(part, edges, "signal")
        background, background_stat = _histogram(part, edges, "background")
        if signal.sum() <= 0 or background.sum() <= 0:
            raise ValueError(f"Category {name} has nonpositive signal or background yield")
        categories.append(TemplateCategory(
            name, observed, signal, background, background_stat, signal_stat))
    return categories


def _profile_bin_background(n, signal_expectation, center, sigma):
    """Analytic Gaussian-constrained background profile for one Poisson bin."""
    if sigma <= 0:
        return max(0.0, center)
    # Solve dNLL/db=0 after writing lambda=signal+b.  If the stationary point
    # violates b>=0, the constrained optimum is the boundary.
    coefficient = sigma * sigma - signal_expectation - center
    discriminant = coefficient * coefficient + 4.0 * n * sigma * sigma
    lam = 0.5 * (-coefficient + math.sqrt(max(0.0, discriminant)))
    return max(0.0, lam - signal_expectation)


class MassShapeLikelihood:
    def __init__(self, categories, normalization_uncertainties=None):
        self.categories = list(categories)
        components = normalization_uncertainties or PUBLISHED_ZZ_UNCERTAINTIES
        if not components or any(v < 0 or not math.isfinite(v) for v in components.values()):
            raise ValueError("Normalization uncertainties must be finite and nonnegative")
        self.normalization_components = dict(components)
        self.relative_background_norm = math.sqrt(sum(v*v for v in components.values()))

    def _nll(self, mu, eta, observations):
        if mu < 0 or not math.isfinite(mu) or not math.isfinite(eta):
            return float("inf")
        scale = math.exp(self.relative_background_norm * eta)
        total = 0.5 * eta * eta
        for category, observed in zip(self.categories, observations):
            for n, s, b0, db0 in zip(observed, category.signal,
                                      category.background,
                                      category.background_mc_stat):
                signal_expectation = mu * max(0.0, s)
                center = max(0.0, b0 * scale)
                sigma = max(0.0, db0 * scale)
                b = _profile_bin_background(float(n), signal_expectation,
                                            center, sigma)
                expectation = max(1e-12, signal_expectation + b)
                total += expectation
                if n > 0:
                    total -= float(n) * math.log(expectation)
                if sigma > 0:
                    total += 0.5 * ((b - center) / sigma) ** 2
        return total

    def _fit_null(self, observations):
        result = minimize_scalar(lambda eta: self._nll(0.0, eta, observations),
                                 bounds=(-8.0, 8.0), method="bounded")
        if not result.success:
            raise RuntimeError(f"Null likelihood fit failed: {result.message}")
        return float(result.fun), 0.0, float(result.x)

    def _fit_free(self, observations):
        signal_total = sum(c.signal.sum() for c in self.categories)
        background_total = sum(c.background.sum() for c in self.categories)
        observed_total = sum(np.asarray(x).sum() for x in observations)
        start_mu = max(0.0, (observed_total-background_total)/max(signal_total, 1e-9))
        result = minimize(lambda x: self._nll(float(x[0]), float(x[1]), observations),
                          x0=np.array([start_mu, 0.0]), method="L-BFGS-B",
                          bounds=[(0.0, 20.0), (-8.0, 8.0)])
        if not result.success:
            raise RuntimeError(f"Signal-plus-background likelihood fit failed: {result.message}")
        return float(result.fun), float(result.x[0]), float(result.x[1])

    def discovery(self, observations):
        observations = [np.asarray(x, dtype=float) for x in observations]
        null_nll, _, eta_null = self._fit_null(observations)
        free_nll, mu_hat, eta_hat = self._fit_free(observations)
        q0 = max(0.0, 2.0 * (null_nll-free_nll)) if mu_hat > 0 else 0.0
        z = math.sqrt(q0)
        return {
            "q0": q0,
            "Z": z,
            "p_value_one_sided": float(norm.sf(z)),
            "mu_hat": mu_hat,
            "background_norm_pull_hat": eta_hat,
            "background_norm_pull_null": eta_null,
        }

    def expected(self):
        asimov = [c.signal+c.background for c in self.categories]
        return self.discovery(asimov)

    def profiled_backgrounds(self, mu, eta, observations):
        """Return the fitted per-bin background yields for plotting."""
        scale = math.exp(self.relative_background_norm * eta)
        fitted = []
        for category, observed in zip(self.categories, observations):
            values = []
            for n, s, b0, db0 in zip(observed, category.signal,
                                      category.background,
                                      category.background_mc_stat):
                values.append(_profile_bin_background(
                    float(n), float(mu)*max(0.0, float(s)),
                    max(0.0, float(b0))*scale,
                    max(0.0, float(db0))*scale))
            fitted.append(np.asarray(values))
        return fitted

    def scan_mu(self, observations, values):
        observations = [np.asarray(x, dtype=float) for x in observations]
        best_nll, _, _ = self._fit_free(observations)
        rows = []
        for mu in values:
            result = minimize_scalar(
                lambda eta: self._nll(float(mu), eta, observations),
                bounds=(-8.0, 8.0), method="bounded")
            if not result.success:
                raise RuntimeError(f"Likelihood scan failed at mu={mu:g}")
            rows.append({"mu": float(mu), "minus2_delta_log_likelihood":
                         max(0.0, 2.0*(float(result.fun)-best_nll)),
                         "background_norm_pull": float(result.x)})
        return pd.DataFrame(rows)

    def profile_mu_interval(self, observations, delta_q):
        """Profile-likelihood interval for the nonnegative signal strength."""
        if delta_q <= 0:
            raise ValueError("delta_q must be positive")
        observations = [np.asarray(x, dtype=float) for x in observations]
        best_nll, mu_hat, _ = self._fit_free(observations)

        def q_mu(mu):
            result = minimize_scalar(
                lambda eta: self._nll(float(mu), eta, observations),
                bounds=(-8., 8.), method="bounded")
            if not result.success:
                raise RuntimeError(f"Likelihood interval fit failed at mu={mu:g}")
            return max(0., 2.*(float(result.fun)-best_nll))

        if mu_hat <= 1e-10 or q_mu(0.) <= delta_q:
            lower = 0.
            lower_at_boundary = True
        else:
            lower = float(brentq(lambda mu: q_mu(mu)-delta_q, 0., mu_hat))
            lower_at_boundary = False

        upper_bracket = max(1., mu_hat + .5)
        while upper_bracket < 20. and q_mu(upper_bracket) < delta_q:
            upper_bracket = min(20., upper_bracket*1.6 + .25)
        if q_mu(upper_bracket) < delta_q:
            upper = float("nan")
        else:
            upper = float(brentq(
                lambda mu: q_mu(mu)-delta_q, mu_hat, upper_bracket))
        return {"low": lower, "high": upper,
                "lower_at_physical_boundary": lower_at_boundary}

    def conditional_toys(self, repeats, seed=42):
        """Background-only conditional toys for an asymptotic cross-check."""
        if repeats <= 0:
            return None
        rng = np.random.default_rng(seed)
        nominal = [c.background for c in self.categories]
        observed = self.discovery([c.observed for c in self.categories])
        q_observed = observed["q0"]
        values = np.empty(repeats, dtype=float)
        for index in range(repeats):
            toy = [rng.poisson(np.maximum(x, 0.0)).astype(float) for x in nominal]
            values[index] = self.discovery(toy)["q0"]
        exceedances = int(np.count_nonzero(values >= q_observed))
        p = (exceedances+1.0)/(repeats+1.0)
        return {"q0": values, "exceedances": exceedances,
                "p_value": p, "Z": float(norm.isf(p)),
                "resolution_limit_Z": float(norm.isf(1.0/(repeats+1.0)))}

    def statistical_z_ensemble(self, repeats, seed=42):
        """Propagate finite data/MC statistics through complete refits.

        Weighted MC bin yields are fluctuated with their sumw2 standard
        deviations and constrained to remain nonnegative. Observed bin counts
        are Poisson-fluctuated. The reported intervals are the central 68%
        ensemble ranges, not systematic or total uncertainty intervals.
        """
        if repeats <= 0:
            return None
        rng = np.random.default_rng(seed)
        expected_values = np.empty(repeats, dtype=float)
        observed_values = np.empty(repeats, dtype=float)
        for repeat in range(repeats):
            varied = []
            varied_observations = []
            for category in self.categories:
                signal_sigma = (category.signal_mc_stat
                                if category.signal_mc_stat is not None
                                else np.zeros_like(category.signal))
                signal = np.maximum(
                    0., rng.normal(category.signal, signal_sigma))
                background = np.maximum(
                    0., rng.normal(category.background,
                                   category.background_mc_stat))
                observed = rng.poisson(np.maximum(category.observed, 0.)).astype(float)
                varied.append(TemplateCategory(
                    category.name, observed, signal, background,
                    category.background_mc_stat, signal_sigma))
                varied_observations.append(observed)
            varied_likelihood = MassShapeLikelihood(
                varied, self.normalization_components)
            expected_values[repeat] = varied_likelihood.expected()["Z"]
            observed_values[repeat] = varied_likelihood.discovery(
                varied_observations)["Z"]

        def interval(values):
            low, median, high = np.quantile(values, [.16, .5, .84])
            return {
                "median": float(median),
                "low": float(low), "high": float(high),
                "sigma": float(.5*(high-low)),
            }
        return {"expected": interval(expected_values),
                "observed": interval(observed_values)}


def _save_mass_plot(categories, edges, fit, path, title):
    centers = 0.5*(edges[:-1]+edges[1:])
    width = np.diff(edges)
    fig, axes = plt.subplots(len(categories), 1, figsize=(10, 4.2*len(categories)),
                             sharex=True, squeeze=False)
    for ax, category, background in zip(
            axes[:, 0], categories, fit["profiled_backgrounds"]):
        signal = fit["mu_hat"]*category.signal
        ax.stairs(background, edges, color="#d62728", linewidth=2,
                  label="Post-fit MC background")
        ax.stairs(background+signal, edges, color="#1f77b4", linewidth=2,
                  label=r"Post-fit signal + background")
        ax.errorbar(centers, category.observed, yerr=np.sqrt(category.observed),
                    xerr=width/2, fmt="o", color="black", capsize=2,
                    label="Observed data")
        ax.set_ylabel(f"Events / {width[0]:g} GeV")
        ax.set_title(category.name.replace("_", " ").title())
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    axes[-1, 0].set_xlabel(r"$m_{4\ell}$ [GeV]")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_scan_plot(scan, path, title):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(scan.mu, scan.minus2_delta_log_likelihood, color="#1f77b4", linewidth=2)
    ax.axhline(1, color="#ffb000", linestyle="--", label=r"68% interval ($\Delta q=1$)")
    ax.axhline(3.84, color="#d62728", linestyle="--", label=r"95% interval ($\Delta q=3.84$)")
    ax.set(xlabel=r"Signal strength $\mu$", ylabel=r"$-2\Delta\ln L$", title=title)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _save_summary_plot(summary, path):
    ordered = summary.sort_values("expected_Z", ascending=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, max(5, .75*len(ordered))))
    ax.scatter(ordered.expected_Z, y, marker="s", s=75, label="Expected (Asimov)")
    ax.scatter(ordered.observed_Z, y, marker="o", s=75, label="Observed")
    for index, row in enumerate(ordered.itertuples()):
        ax.plot([row.expected_Z, row.observed_Z], [index, index], color="0.72", zorder=0)
    ax.set_yticks(y, ordered.model)
    ax.set_xlabel(r"Local profile-likelihood significance $Z$ [$\sigma$]")
    ax.set_title(r"Binned $m_{4\ell}$ mass-shape significance (MC background)",
                 fontweight="bold")
    ax.grid(axis="x", alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _save_significance_forest(summary, metric, p_metric, path, title,
                              xlabel, include_mu=False):
    """Save one MC-background-only forest for a likelihood result column.

    These plots intentionally do not reuse the propagated errors from the
    legacy one-bin counting model.  A confidence interval for the binned-shape
    result requires its own likelihood construction or ensemble calculation.
    """
    ordered = summary.sort_values(metric, ascending=False).reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = {
        "XGBoost": "#0072B2", "LightGBM": "#009E3B",
        "Random Forest": "#E69F00", "MLP": "#E51D2A",
        "Logistic Regression": "#9564CC", "No ML": "black",
    }
    values = ordered[metric].to_numpy(float)
    prefix = "expected" if metric == "expected_Z" else "observed"
    xmax = max(5.35, float(values.max()) + 1.35)
    fig, ax = plt.subplots(figsize=(11, max(6, .9*len(ordered))))
    for index, row in ordered.iterrows():
        model = str(row["model"])
        value = float(row[metric])
        p_value = float(row[p_metric])
        def finite_or_nan(column):
            value = row.get(column, np.nan)
            return np.nan if value is None else float(value)
        low = finite_or_nan(f"{prefix}_Z_stat_low")
        high = finite_or_nan(f"{prefix}_Z_stat_high")
        sigma = finite_or_nan(f"sigma_{prefix}_Z_stat")
        if np.isfinite(low) and np.isfinite(high):
            xerr = np.array([[max(0., value-low)], [max(0., high-value)]])
            ax.errorbar(value, index, xerr=xerr, fmt="o",
                        color=colors.get(model, "#666666"), markersize=10,
                        capsize=5, elinewidth=2)
        else:
            ax.plot(value, index, "o", color=colors.get(model, "#666666"),
                    markersize=10)
        annotation = (f"Z = {value:.2f} ± {sigma:.2f}\np = {p_value:.2e}"
                      if np.isfinite(sigma)
                      else f"Z = {value:.2f}\np = {p_value:.2e}")
        if include_mu:
            annotation += "\n" + rf"$\hat{{\mu}}$ = {float(row['mu_hat']):.2f}"
        ax.text(value + .08, index, annotation, va="center", fontsize=11,
                bbox={"facecolor": "white", "alpha": .88, "edgecolor": "none"})
    ax.axvline(3., color="#FFB000", linestyle="--", linewidth=1.7,
               label=r"Evidence ($3\sigma$)")
    ax.axvline(5., color="#E51D2A", linestyle="--", linewidth=1.7,
               label=r"Discovery ($5\sigma$)")
    ax.set_yticks(y, ordered.model, fontsize=14)
    ax.set_xlim(0., xmax)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    ax.grid(axis="x", alpha=.25)
    ax.invert_yaxis()
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _save_signal_strength_forest(summary, path):
    ordered = summary.sort_values("mu_hat", ascending=False).reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = {
        "XGBoost": "#0072B2", "LightGBM": "#009E3B",
        "Random Forest": "#E69F00", "MLP": "#E51D2A",
        "Logistic Regression": "#9564CC", "No ML": "black",
    }
    fig, ax = plt.subplots(figsize=(11, max(6, .9*len(ordered))))
    for index, row in ordered.iterrows():
        mu = float(row.mu_hat)
        low68, high68 = float(row.mu_68_low), float(row.mu_68_high)
        low95, high95 = float(row.mu_95_low), float(row.mu_95_high)
        color = colors.get(str(row.model), "#666666")
        ax.errorbar(mu, index,
                    xerr=np.array([[mu-low95], [high95-mu]]),
                    fmt="none", ecolor=color, alpha=.45,
                    capsize=4, elinewidth=2)
        ax.errorbar(mu, index,
                    xerr=np.array([[mu-low68], [high68-mu]]),
                    fmt="o", color=color, capsize=5,
                    markersize=9, elinewidth=4)
        ax.text(high95+.03, index,
                rf"$\hat{{\mu}}={mu:.2f}^{{+{high68-mu:.2f}}}_{{-{mu-low68:.2f}}}$",
                va="center", fontsize=11,
                bbox={"facecolor": "white", "alpha": .88, "edgecolor": "none"})
    ax.axvline(1., color="#0072B2", linestyle="--", linewidth=1.8,
               label=r"Standard Model ($\mu=1$)")
    ax.axvline(1.28, color="#CC79A7", linestyle=":", linewidth=2.,
               label=r"Published reference central value ($\mu=1.28$)")
    ax.set_yticks(y, ordered.model, fontsize=14)
    ax.set_xlim(left=0.)
    ax.set_xlabel(r"Fitted signal strength $\mu$", fontsize=15)
    ax.set_title(r"Observed signal strength from the $m_{4\ell}$ profile likelihood",
                 fontsize=18, fontweight="bold", pad=14)
    ax.grid(axis="x", alpha=.25)
    ax.invert_yaxis()
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def compare_mass_shape_models(paths, thresholds, output, fit_range=(105., 160.),
                              bin_width=2.5, toys=0, seed=42,
                              stat_repeats=0,
                              normalization_uncertainties=None):
    """Run the independent mass-shape workflow for No ML and saved models."""
    if fit_range[1] <= fit_range[0] or bin_width <= 0:
        raise ValueError("Fit range and bin width must be positive and increasing")
    span = fit_range[1]-fit_range[0]
    bins = int(round(span/bin_width))
    if bins < 5 or not math.isclose(bins*bin_width, span, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Fit range must contain an integer number of at least five bins")
    edges = np.linspace(fit_range[0], fit_range[1], bins+1)
    output = new_output(output)

    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    reference = next(iter(frames.values()))
    baseline_categories = build_templates(reference, edges, threshold=None)
    jobs = [("No ML", baseline_categories)]
    for name, frame in frames.items():
        jobs.append((name, build_templates(frame, edges, thresholds[name])))

    rows = []
    toy_tables = []
    uncertainty_model = normalization_uncertainties or PUBLISHED_ZZ_UNCERTAINTIES
    relative_norm = math.sqrt(sum(value*value for value in uncertainty_model.values()))
    for index, (name, categories) in enumerate(jobs):
        likelihood = MassShapeLikelihood(categories, uncertainty_model)
        observed = likelihood.discovery([c.observed for c in categories])
        expected = likelihood.expected()
        interval68 = likelihood.profile_mu_interval(
            [c.observed for c in categories], 1.0)
        interval95 = likelihood.profile_mu_interval(
            [c.observed for c in categories], 3.841458820694124)
        scan_max = max(3.0, observed["mu_hat"]*2.2, expected["mu_hat"]*2.2)
        scan = likelihood.scan_mu([c.observed for c in categories],
                                  np.linspace(0, scan_max, 121))
        slug = name.lower().replace(" ", "-")
        scan.to_csv(output/f"{slug}_likelihood_scan.csv", index=False)
        _save_scan_plot(scan, output/f"{slug}_likelihood_scan.png",
                        f"{name}: observed profile likelihood")
        plot_fit = {**observed,
                    "profiled_backgrounds": likelihood.profiled_backgrounds(
                        observed["mu_hat"], observed["background_norm_pull_hat"],
                        [c.observed for c in categories])}
        _save_mass_plot(categories, edges, plot_fit,
                        output/f"{slug}_mass_fit.png",
                        f"{name}: observed binned mass fit")
        toy = likelihood.conditional_toys(toys, seed+index) if toys else None
        statistical = likelihood.statistical_z_ensemble(
            stat_repeats, seed+10_000+index) if stat_repeats else None
        if toy is not None:
            toy_tables.append(pd.DataFrame({"model": name, "q0": toy.pop("q0")}))
        rows.append({
            "model": name,
            "selection": "inclusive" if name == "No ML" else
                         f"score pass/fail at {thresholds[name]:g}",
            "fit_min_gev": fit_range[0], "fit_max_gev": fit_range[1],
            "bin_width_gev": bin_width,
            "observed_events": sum(c.observed.sum() for c in categories),
            "signal_yield": sum(c.signal.sum() for c in categories),
            "background_yield": sum(c.background.sum() for c in categories),
            "observed_q0": observed["q0"], "observed_Z": observed["Z"],
            "observed_p_value": observed["p_value_one_sided"],
            "mu_hat": observed["mu_hat"],
            "mu_68_low": interval68["low"],
            "mu_68_high": interval68["high"],
            "mu_95_low": interval95["low"],
            "mu_95_high": interval95["high"],
            "mu_68_lower_at_boundary": interval68["lower_at_physical_boundary"],
            "mu_95_lower_at_boundary": interval95["lower_at_physical_boundary"],
            "background_norm_pull_hat": observed["background_norm_pull_hat"],
            "expected_q0": expected["q0"], "expected_Z": expected["Z"],
            "expected_p_value": expected["p_value_one_sided"],
            "observed_Z_stat_low": (None if statistical is None else
                                     statistical["observed"]["low"]),
            "observed_Z_stat_high": (None if statistical is None else
                                      statistical["observed"]["high"]),
            "sigma_observed_Z_stat": (None if statistical is None else
                                       statistical["observed"]["sigma"]),
            "expected_Z_stat_low": (None if statistical is None else
                                     statistical["expected"]["low"]),
            "expected_Z_stat_high": (None if statistical is None else
                                      statistical["expected"]["high"]),
            "sigma_expected_Z_stat": (None if statistical is None else
                                       statistical["expected"]["sigma"]),
            "toy_repeats": toys,
            "toy_exceedances": None if toy is None else toy["exceedances"],
            "toy_p_value": None if toy is None else toy["p_value"],
            "toy_Z": None if toy is None else toy["Z"],
            "toy_resolution_limit_Z": None if toy is None else toy["resolution_limit_Z"],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(output/"mass_shape_significance.csv", index=False)
    _save_summary_plot(summary, output/"mass_shape_significance.png")
    _save_significance_forest(
        summary, "expected_Z", "expected_p_value",
        output/"expected_mc_significance_forest.png",
        r"Expected significance from trained MC ($m_{4\ell}$ shape)",
        r"Expected local profile-likelihood significance $Z$ [$\sigma$]")
    _save_significance_forest(
        summary, "observed_Z", "observed_p_value",
        output/"observed_mc_background_significance_forest.png",
        r"Observed-data significance with MC background ($m_{4\ell}$ shape)",
        r"Observed local profile-likelihood significance $Z$ [$\sigma$]",
        include_mu=True)
    _save_signal_strength_forest(
        summary, output/"observed_signal_strength_forest.png")
    if toy_tables:
        pd.concat(toy_tables, ignore_index=True).to_csv(
            output/"background_only_toy_q0.csv", index=False)
    metadata = {
        "method": "binned m4l profile likelihood with MC background only",
        "fit_range_gev": list(fit_range), "bin_width_gev": bin_width,
        "categories": "No ML inclusive; classifiers use score pass/fail categories",
        "background_normalization_uncertainties": uncertainty_model,
        "combined_relative_background_normalization": relative_norm,
        "mc_statistical_model": "independent Gaussian-constrained weighted-MC yield in every category and mass bin",
        "discovery_test": "q0=-2log[L(mu=0)/L(mu_hat>=0)]; local asymptotic Z=sqrt(q0)",
        "signal_strength_intervals": {
            "68_percent": "-2 delta log L = 1.0",
            "95_percent": "-2 delta log L = 3.841458820694124",
            "construction": "profile likelihood with mu constrained nonnegative and all modeled nuisances re-profiled",
        },
        "toys": {"repeats": toys, "seed": seed,
                 "type": "conditional background-only Poisson toys; auxiliary constraints fixed at nominal"},
        "statistical_error_ensemble": {
            "repeats": stat_repeats, "seed_offset": 10000,
            "interval": "central 68% (16th and 84th percentiles)",
            "mc": "independent Gaussian bin fluctuations using weighted sumw2",
            "data": "independent Poisson bin fluctuations",
            "scope": "statistical only; excludes detector and theory systematics",
        },
        "limitations": [
            "This is ATLAS-inspired, not the unpublished full ATLAS likelihood.",
            "Detector shape variations and the complete ATLAS category scheme are unavailable.",
            "The published inclusive ZZ* QCD/PDF/shower ranges are used as correlated normalization components.",
            "Toy p-values cannot resolve significances beyond their finite repeat limit."
        ],
    }
    with open(output/"mass_shape_method.json", "w") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    return output
