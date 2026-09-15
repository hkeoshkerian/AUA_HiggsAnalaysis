"""Headless figures: plot calls never train models or read ROOT files."""
import numpy as np

def save_mass_plot(frame, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    bins = np.arange(80,252.5,2.5)
    background = frame[frame.role == "background"]
    signal = frame[frame.role == "signal"]
    ax.hist(background.mass,bins=bins,weights=background.weight,label="MC background",alpha=.5)
    ax.hist(signal.mass,bins=bins,weights=signal.weight,label="MC signal",histtype="step",linewidth=1.6)
    observed = frame[frame.role == "data"]
    if len(observed):
        counts,_ = np.histogram(observed.mass,bins=bins)
        ax.errorbar((bins[:-1]+bins[1:])/2,counts,yerr=np.sqrt(counts),fmt=".",color="black",label="Observed data")
    ax.set(xlabel="Four-lepton mass [GeV]",ylabel="Events / 2.5 GeV",title=title)
    ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

def save_roc_plot(frame,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    mc = frame[frame.role != "data"]
    fpr,tpr,_ = roc_curve(mc.label,mc.score,sample_weight=mc.weight)
    fig,ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr,tpr); ax.plot([0,1],[0,1],"--",color="gray")
    ax.set(xlabel="Background efficiency",ylabel="Signal efficiency",title="Weighted out-of-fold ROC")
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

def save_all_model_roc(frames, results, path):
    """Reference all-model weighted OOF ROC plot with bootstrap AUC intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    ordered = results.sort_values("weighted_oof_auc", ascending=False)
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(index) for index, name in enumerate(ordered.model)}
    fig, ax = plt.subplots(figsize=(9, 7))
    for row in ordered.itertuples(index=False):
        mc = frames[row.model][frames[row.model].role != "data"]
        fpr, tpr, _ = roc_curve(
            mc.label.to_numpy(int), mc.score.to_numpy(float),
            sample_weight=mc.weight.to_numpy(float))
        label = (f"{row.model} (AUC = {row.weighted_oof_auc:.3f}, "
                 f"95% CI [{row.auc_ci_low:.3f}, {row.auc_ci_high:.3f}])")
        ax.plot(fpr, tpr, label=label, color=colors[row.model])
    ax.plot([0, 1], [0, 1], "k--", alpha=.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves with Bootstrap 95% CI (sorted by AUC)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def save_feature_plots(frame, output, features):
    """Save reference-style, MC-only kinematics and signal correlations."""
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    output = Path(output); output.mkdir(parents=True, exist_ok=True)

    mc = frame[frame.role != "data"]
    backgrounds = mc[mc.role == "background"]
    signal = mc[mc.role == "signal"]

    def background_parts():
        # Preserve the reference order: minor backgrounds first, then ZZ*.
        groups = list(backgrounds.groupby("sample", sort=False))
        return sorted(groups, key=lambda item: "ZZ" in str(item[0]))

    def draw_stack(ax, feature, bins, xlabel, ylabel, title=None, transform=None):
        transform = transform or (lambda values: values)
        parts = background_parts()
        arrays = [transform(part[feature].to_numpy(float)) for _, part in parts]
        weights = [part.weight.to_numpy(float) for _, part in parts]
        labels = [name for name, _ in parts]
        colors = ["#6555d9" if "ZZ" not in str(name) else "#ff1717"
                  for name, _ in parts]
        if arrays:
            result = ax.hist(arrays, bins=bins, weights=weights, stacked=True,
                             color=colors, label=labels, histtype="stepfilled")
            background_total = result[0][-1]
            all_values = np.hstack(arrays)
            all_weights = np.hstack(weights)
            error = np.sqrt(np.histogram(all_values, bins=bins,
                                         weights=all_weights**2)[0])
            centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(centers, 2*error, bottom=background_total-error,
                   width=np.diff(bins), color="none", edgecolor="black",
                   linewidth=0, hatch="////", label="Stat. Unc.")
        else:
            background_total = np.zeros(len(bins)-1)
        if len(signal):
            ax.hist(transform(signal[feature].to_numpy(float)), bins=bins,
                    weights=signal.weight.to_numpy(float), bottom=background_total,
                    color="#11bce5", label=r"Signal ($m_H$ = 125 GeV)")
        ax.set_xlim(bins[0], bins[-1]); ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel, fontsize=12, x=1, horizontalalignment="right")
        ax.set_ylabel(ylabel, fontsize=11, y=1, horizontalalignment="right")
        if title: ax.set_title(title, fontsize=12)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        handles, legend_labels = ax.get_legend_handles_labels()
        def legend_rank(label):
            if "Signal" in label: return 2
            if "Stat." in label: return 3
            if "ZZ" in label: return 1
            return 0
        ordered = sorted(zip(handles, legend_labels), key=lambda item: legend_rank(item[1]))
        if ordered:
            ax.legend([item[0] for item in ordered], [item[1] for item in ordered],
                      frameon=False, fontsize=9)

    labels = {
        "mz1": r"$m_{Z_1}$ [GeV]", "mz2": r"$m_{Z_2}$ [GeV]",
        "ptz1": r"$p_T^{Z_1}$ [GeV]", "ptz2": r"$p_T^{Z_2}$ [GeV]",
        "pt4l": r"$p_T^{4\ell}$ [GeV]", "met": r"$E_T^{\rm miss}$ [GeV]",
        "jet_n": r"Jet multiplicity $N_{\rm jets}$",
    }
    fixed_bins = {
        "mz1": np.arange(40, 122, 2), "mz2": np.arange(0, 82, 2),
        "ptz1": np.arange(0, 205, 5), "ptz2": np.arange(0, 205, 5),
        "pt4l": np.arange(0, 205, 5), "met": np.arange(0, 205, 5),
        "jet_n": np.arange(-.5, 9.5, 1),
    }
    for feature in features:
        values = mc[feature].to_numpy(float)
        low, high = np.nanquantile(values, [.01, .99])
        if low == high: low, high = low-.5, high+.5
        bins = fixed_bins.get(feature, np.linspace(low, high, 41))
        width = np.diff(bins)[0]
        unit = " GeV" if feature.startswith(("m", "pt")) or feature == "met" else ""
        fig, ax = plt.subplots(figsize=(10, 7))
        draw_stack(ax, feature, bins, labels.get(feature, feature),
                   f"Events / {width:g}{unit}", f"{feature} distribution")
        ax.text(.05, .95, "ATLAS Open Data", transform=ax.transAxes, fontsize=14, va="top")
        ax.text(.05, .90, "for education", transform=ax.transAxes,
                fontsize=11, style="italic", va="top")
        fig.tight_layout(); fig.savefig(output/f"{feature}.png", dpi=150); plt.close(fig)

    def panel_figure(filename, panels, shape, suptitle, bins, ylabel, transform=None):
        if not all(key in features for key, _, _ in panels): return
        fig, axes = plt.subplots(*shape, figsize=(14, 10) if shape == (2, 2) else (16, 10))
        axes = np.asarray(axes).reshape(-1)
        for ax, (key, title, xlabel) in zip(axes, panels):
            panel_bins = bins[key] if isinstance(bins, dict) else bins
            panel_ylabel = ylabel[key] if isinstance(ylabel, dict) else ylabel
            draw_stack(ax, key, panel_bins, xlabel, panel_ylabel, title, transform)
        for ax in axes[len(panels):]: ax.set_visible(False)
        fig.suptitle(suptitle, fontsize=13)
        fig.tight_layout(); fig.savefig(output/filename, dpi=200); plt.close(fig)

    panel_figure("lepton_pt_2x2.png", [
        ("pt_z1_l1", r"$Z_1$ leading lepton", r"$p_T^{Z_1,\,\ell_1}$ [GeV]"),
        ("pt_z1_l2", r"$Z_1$ subleading lepton", r"$p_T^{Z_1,\,\ell_2}$ [GeV]"),
        ("pt_z2_l1", r"$Z_2$ leading lepton", r"$p_T^{Z_2,\,\ell_1}$ [GeV]"),
        ("pt_z2_l2", r"$Z_2$ subleading lepton", r"$p_T^{Z_2,\,\ell_2}$ [GeV]"),
    ], (2, 2),
        r"$H \rightarrow ZZ^* \rightarrow 4\ell$ — Lepton $p_T$    "
        r"$\sqrt{s}=13$ TeV, $\int L\,dt=36.6$ fb$^{-1}$",
        np.arange(0, 153, 3), "Events / 3.0 GeV")

    panel_figure("lepton_eta_2x2.png", [
        ("eta_z1_l1", r"$Z_1$ leading lepton", r"$\eta^{Z_1,\,\ell_1}$"),
        ("eta_z1_l2", r"$Z_1$ subleading lepton", r"$\eta^{Z_1,\,\ell_2}$"),
        ("eta_z2_l1", r"$Z_2$ leading lepton", r"$\eta^{Z_2,\,\ell_1}$"),
        ("eta_z2_l2", r"$Z_2$ subleading lepton", r"$\eta^{Z_2,\,\ell_2}$"),
    ], (2, 2),
        r"$H \rightarrow ZZ^* \rightarrow 4\ell$ — Lepton $\eta$    "
        r"$\sqrt{s}=13$ TeV, $\int L\,dt=36.6$ fb$^{-1}$",
        np.arange(-3, 3.2, .2), "Events / 0.2")

    panel_figure("helicity_angles.png", [
        ("theta1", r"$\theta_1$", r"$\theta_1$ [rad]"),
        ("theta2", r"$\theta_2$", r"$\theta_2$ [rad]"),
        ("Theta", r"$\Theta$", r"$\Theta$ [rad]"),
        ("Phi", r"$\Phi$", r"$\Phi$ [rad]"),
        ("Phi1", r"$\Phi_1$", r"$\Phi_1$ [rad]"),
    ], (2, 3), r"$H \rightarrow ZZ^* \rightarrow 4\ell$ — Helicity angles",
        {key: np.arange(0, np.pi+.1, .1) for key in ("theta1", "theta2", "Theta")} |
        {key: np.arange(-np.pi, np.pi+.2, .2) for key in ("Phi", "Phi1")},
        {key: "Events / 0.1 rad" for key in ("theta1", "theta2", "Theta")} |
        {key: "Events / 0.2 rad" for key in ("Phi", "Phi1")})

    panel_figure("delta_phi_1x2.png", [
        ("dphi_z1", r"$Z_1$ (on-shell)", r"$\Delta\phi_{Z_1}$ [deg]"),
        ("dphi_z2", r"$Z_2$ (off-shell)", r"$\Delta\phi_{Z_2}$ [deg]"),
    ], (1, 2), r"$H \rightarrow ZZ^* \rightarrow 4\ell$ — $\Delta\phi$ between lepton pairs",
        np.arange(0, 190, 10), "Events / 10°", np.degrees)

    panel_figure("scalar_pt_sums_1x2.png", [
        ("scalar_pt_sum_z1", r"$Z_1$ (on-shell)",
         r"$p_T^{\ell_1}+p_T^{\ell_2}$ [GeV] $(Z_1)$"),
        ("scalar_pt_sum_z2", r"$Z_2$ (off-shell)",
         r"$p_T^{\ell_1}+p_T^{\ell_2}$ [GeV] $(Z_2)$"),
    ], (1, 2), r"$H \rightarrow ZZ^* \rightarrow 4\ell$ — Scalar $p_T$ sum of lepton pairs",
        np.arange(0, 205, 5), "Events / 5 GeV")

    panel_figure("delta_r_2x3.png", [
        ("dR_z1", r"$Z_1$ lepton pair (same Z)", r"$\Delta R_{Z_1}$"),
        ("dR_z2", r"$Z_2$ lepton pair (same Z)", r"$\Delta R_{Z_2}$"),
        ("dR_02", r"Cross-Z pair $\ell_1^{Z_1}$ vs $\ell_1^{Z_2}$", r"$\Delta R_{02}$"),
        ("dR_03", r"Cross-Z pair $\ell_1^{Z_1}$ vs $\ell_2^{Z_2}$", r"$\Delta R_{03}$"),
        ("dR_12", r"Cross-Z pair $\ell_2^{Z_1}$ vs $\ell_1^{Z_2}$", r"$\Delta R_{12}$"),
        ("dR_13", r"Cross-Z pair $\ell_2^{Z_1}$ vs $\ell_2^{Z_2}$", r"$\Delta R_{13}$"),
    ], (2, 3), r"$H \rightarrow ZZ^* \rightarrow 4\ell$ — $\Delta R$ between all lepton pairs",
        np.arange(0, 6.2, .2), "Events / 0.2")

    signal = frame[frame.role == "signal"]
    correlation = signal[list(features)].corr().fillna(0.)
    size = max(8, min(18, len(features)*.52))
    fig, ax = plt.subplots(figsize=(size, size*.875))
    values = correlation.to_numpy()
    masked = np.ma.array(values, mask=np.triu(np.ones_like(values, dtype=bool)))
    image = ax.imshow(masked, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(features)), features, rotation=90, fontsize=7)
    ax.set_yticks(range(len(features)), features, fontsize=7)
    for row in range(len(features)):
        for column in range(row):
            value = values[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=5.5, color="white" if abs(value) > .5 else "#222222")
    ax.set_title("Feature correlation matrix (signal)", fontsize=14)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout(); fig.savefig(output/"signal_correlations.png", dpi=200); plt.close(fig)

def save_preselection_mass_plot(frame, path, include_data=False):
    """Reference-style m4l distribution before any ML cut."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.arange(80, 252.5, 2.5); centers = (bins[:-1]+bins[1:])/2
    background = frame[frame.role == "background"]
    groups = [(name, group) for name, group in background.groupby("sample", sort=False)]
    arrays = [group.mass.to_numpy() for _, group in groups]
    weights = [group.weight.to_numpy() for _, group in groups]
    colors = ["#6555d9", "#ff1717", "#bd63c5", "#e09b32"][:len(arrays)]
    labels = [name for name, _ in groups]
    fig, ax = plt.subplots(figsize=(13, 8))
    if arrays:
        ax.hist(arrays, bins=bins, weights=weights, stacked=True, color=colors,
                label=labels, histtype="stepfilled")
    sumw = np.histogram(background.mass, bins=bins, weights=background.weight)[0]
    sumw2 = np.histogram(background.mass, bins=bins,
                         weights=np.square(background.weight))[0]
    uncertainty = np.sqrt(sumw2)
    ax.bar(centers, 2*uncertainty, bottom=sumw-uncertainty, width=np.diff(bins),
           fill=False, hatch="////", edgecolor="black", linewidth=0, label="Stat. Unc.")
    signal = frame[frame.role == "signal"]
    ax.hist(signal.mass, bins=bins, weights=signal.weight, linewidth=2,
            color="#11bce5", label="Signal ($m_H$ = 125 GeV)", zorder=5)
    if include_data:
        observed = frame[frame.role == "data"]
        counts = np.histogram(observed.mass, bins=bins)[0]
        ax.errorbar(centers, counts, yerr=np.sqrt(counts), fmt="o", color="black",
                    markersize=4, label="Data", zorder=6)
    ax.set(xlim=(80, 250), ylim=(0, None),
           xlabel=r"4-lepton invariant mass $m_{4\ell}$ [GeV]",
           ylabel="Events / 2.5 GeV",
           title="")
    ax.text(.10, .96, "ATLAS Open Data", transform=ax.transAxes,
            fontsize=17, va="top")
    ax.text(.10, .905, "for education", transform=ax.transAxes,
            fontsize=12, va="top", style="italic")
    ax.text(.10, .845, r"$\sqrt{s}=13$ TeV, $\int L\,dt=36.6$ fb$^{-1}$",
            transform=ax.transAxes, fontsize=14, va="top")
    ax.text(.10, .785, r"$H \rightarrow ZZ^* \rightarrow 4\ell$",
            transform=ax.transAxes, fontsize=15, va="top")
    ax.minorticks_on(); ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

def save_classifier_metrics(comparison, output):
    """Reproduce the original accuracy and three-panel classifier summaries."""
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    labels = comparison.model.tolist()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.bar(x, comparison.weighted_accuracy, color="#6f9fc3", edgecolor="#263238")
    ax.set_xticks(x, [f"{name}\n(thr={threshold:.2f})" for name, threshold in
                      zip(labels, comparison.threshold)])
    ax.set_ylabel("Accuracy = (TP + TN) / (TP + TN + FP + FN)")
    ax.set_title("Classifier Accuracy on MC\n(weighted by lumi × MC weights)", weight="bold")
    ax.grid(axis="y", alpha=.25)
    for bar, value in zip(bars, comparison.weighted_accuracy):
        ax.text(bar.get_x()+bar.get_width()/2, value+.004, f"{value:.4f}",
                ha="center", va="bottom", weight="bold")
    lower = max(0, float(comparison.weighted_accuracy.min())-.02)
    ax.set_ylim(lower, min(1.02, float(comparison.weighted_accuracy.max())+.02))
    fig.tight_layout(); fig.savefig(output/"accuracy_comparison.png", dpi=200); plt.close(fig)

    metrics = [("weighted_accuracy", "Accuracy", "#6f9fc3"),
               ("signal_efficiency", "Signal Efficiency", "#4da64d"),
               ("background_rejection", "Background Rejection", "#ffa12e")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Classifier Performance Metrics on MC", fontsize=17, weight="bold")
    for ax, (column, title, color) in zip(axes, metrics):
        bars = ax.bar(x, comparison[column], color=color, edgecolor="#333333")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set(title=title, ylabel=title, ylim=(0, 1.05))
        ax.grid(axis="y", alpha=.25)
        for bar, value in zip(bars, comparison[column]):
            ax.text(bar.get_x()+bar.get_width()/2, value+.012, f"{value:.3f}",
                    ha="center", weight="bold", fontsize=9)
    fig.tight_layout(); fig.savefig(output/"classifier_metrics_3panel.png", dpi=200); plt.close(fig)

def save_model_significance(comparison, baseline, path):
    """Reference-style five-model significance forest plus no-ML baseline."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The reference significance comparison retains the five classifiers used
    # downstream and omits QDA/GaussianNB after the Table V study.
    excluded = {"QDA", "GaussianNB", "Gaussian NB"}
    retained = comparison[~comparison.model.isin(excluded)].copy()
    cmap = plt.get_cmap("tab10")
    # Lock the reference figure's colors rather than allowing small numerical
    # AUC changes to recolor models between runs.
    reference_color_index = {
        "XGBoost": 0, "Random Forest": 1, "LightGBM": 2,
        "MLP": 3, "Logistic Regression": 4, "LogReg": 4,
    }
    colors = {name: cmap(reference_color_index.get(name, 0))
              for name in retained.model}
    retained = retained.sort_values("Z_proxy", ascending=False)

    names = retained.model.tolist() + ["No ML"]
    means = retained.Z_proxy.to_list() + [baseline["Z_proxy"]]
    errors = retained.sigma_Z_proxy.to_list() + [baseline["sigma_Z_proxy"]]
    point_colors = [colors[name] for name in retained.model] + ["black"]

    fig, ax = plt.subplots(figsize=(11, 7))
    y_positions = np.arange(len(names))
    for y, z, error, color in zip(y_positions, means, errors, point_colors):
        ax.errorbar(z, y, xerr=error, fmt="o", color=color, capsize=5,
                    markersize=9, elinewidth=2)
    ax.set_yticks(y_positions, names, fontsize=15)
    ax.set_xlabel("Significance Z", fontsize=18)
    ax.set_title("Model Significance Comparison", fontsize=20, pad=15)
    # Match the requested reference image boundaries and half-unit ticks.
    ax.set_xlim(0, 4.14)
    ax.set_xticks(np.arange(0, 4.01, .5))
    ax.grid(axis="x", alpha=.3)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=14)
    x_range = max(means) - min(means)
    for y, z, error in zip(y_positions, means, errors):
        p_value = .5 * math.erfc(z / math.sqrt(2.))
        ax.text(z + error + .02*x_range, y,
                f"Z = {z:.2f} ± {error:.2f}\np = {p_value:.2e}",
                va="center", fontsize=12,
                bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)

def save_observed_model_significance(observed, path):
    """Reference Table-VII two-method significance comparison on real data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_order = ["XGBoost", "LightGBM", "Random Forest", "MLP",
                   "Logistic Regression"]
    colors = {"XGBoost": "#0072b2", "LightGBM": "#00a62b",
              "Random Forest": "#ff7800", "MLP": "#ed1024",
              "Logistic Regression": "#9664c8", "No ML": "black"}
    available = list(dict.fromkeys(observed.model.astype(str)))
    retained = ([name for name in model_order if name in available] +
                [name for name in available if name not in model_order])
    baseline_rows = observed[
        (observed.stage == "before_ml") &
        (observed.method == "mc_prediction")]
    if baseline_rows.empty:
        raise ValueError("Observed comparison is missing the No-ML baseline")
    baseline = baseline_rows.iloc[0]
    names = retained + ["No ML"]
    y = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True)
    panels = [("mc_prediction", "MC Prediction Method"),
              ("sideband", "Sideband Extrapolation Method")]
    for ax, (method, title) in zip(axes, panels):
        for index, name in enumerate(retained):
            rows = observed[(observed.model == name) &
                            (observed.stage == "after_ml") &
                            (observed.method == method)]
            if rows.empty: continue
            row = rows.iloc[0]
            ax.errorbar(row.Z, index, xerr=row.sigma_Z, fmt="o",
                        color=colors[name], markersize=11, capsize=6,
                        elinewidth=2)
            ax.text(row.Z + row.sigma_Z + .12, index,
                    f"{row.Z:.2f} ± {row.sigma_Z:.2f}",
                    va="center", fontsize=11)
        index = len(retained)
        ax.errorbar(baseline.Z, index, xerr=baseline.sigma_Z, fmt="o",
                    color=colors["No ML"], markersize=11, capsize=6,
                    elinewidth=2)
        ax.text(baseline.Z + baseline.sigma_Z + .12, index,
                f"{baseline.Z:.2f} ± {baseline.sigma_Z:.2f}",
                va="center", fontsize=11)
        ax.axvline(3., linestyle="--", color="orange", alpha=.7,
                   linewidth=2, label="Evidence (3σ)")
        ax.axvline(5., linestyle="--", color="red", alpha=.7,
                   linewidth=2, label="Discovery (5σ)")
        ax.set(xlim=(0, 9), xlabel="Significance Z (σ)", title=title)
        ax.set_xticks(np.arange(0, 10, 1))
        ax.title.set_fontsize(17); ax.title.set_weight("bold")
        ax.xaxis.label.set_fontsize(15)
        ax.grid(axis="x", alpha=.3)
        ax.tick_params(axis="x", labelsize=13)
        ax.legend(loc="lower right", fontsize=11)
    axes[0].set_yticks(y, names, fontsize=13)
    axes[0].invert_yaxis()
    fig.suptitle(r"H $\rightarrow$ ZZ* $\rightarrow$ 4$\ell$ Significance Across ML Models",
                 fontsize=19, weight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def save_observed_profile_significance(observed, path):
    """Plot one-bin profile-likelihood significance for both background methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_order = ["XGBoost", "LightGBM", "Random Forest", "MLP",
                   "Logistic Regression", "GaussianNB", "QDA"]
    colors = {"XGBoost": "#0072b2", "LightGBM": "#00a62b",
              "Random Forest": "#ff7800", "MLP": "#ed1024",
              "Logistic Regression": "#9664c8", "GaussianNB": "#8c564b",
              "QDA": "#e377c2", "No ML": "black"}
    retained = [name for name in model_order if name in set(observed.model)]
    names = retained + ["No ML"]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    for ax, (method, title) in zip(
            axes, [("mc_prediction", "MC Prediction Method"),
                   ("sideband", "Sideband Extrapolation Method")]):
        for index, name in enumerate(retained):
            rows = observed[(observed.model == name) &
                            (observed.stage == "after_ml") &
                            (observed.method == method)]
            if rows.empty or not np.isfinite(rows.iloc[0].profile_Z):
                continue
            row = rows.iloc[0]
            ax.plot(row.profile_Z, index, "o", color=colors.get(name, "#0072b2"),
                    markersize=10)
            ax.text(row.profile_Z+.10, index,
                    f"Z={row.profile_Z:.2f}, p={row.profile_p_value_one_sided:.2e}",
                    va="center", fontsize=10)
        baseline_rows = observed[(observed.stage == "before_ml") &
                                 (observed.method == method)]
        if not baseline_rows.empty:
            baseline = baseline_rows.iloc[0]
            index = len(retained)
            ax.plot(baseline.profile_Z, index, "o", color="black", markersize=10)
            ax.text(baseline.profile_Z+.10, index,
                    f"Z={baseline.profile_Z:.2f}, p={baseline.profile_p_value_one_sided:.2e}",
                    va="center", fontsize=10)
        ax.axvline(3, linestyle="--", color="orange", alpha=.7, label="Evidence (3σ)")
        ax.axvline(5, linestyle="--", color="red", alpha=.7, label="Discovery (5σ)")
        ax.set(xlim=(0, 9), xlabel="Local profile-likelihood significance Z (σ)",
               title=title)
        ax.grid(axis="x", alpha=.3); ax.legend(loc="lower right")
    axes[0].set_yticks(y, names); axes[0].invert_yaxis()
    fig.suptitle(r"$H \rightarrow ZZ^* \rightarrow 4\ell$ one-bin profile-likelihood significance",
                 fontsize=18, weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)

def save_detailed_mass_plot(predictions, path, threshold, model_name="Classifier"):
    """Stacked MC and observed-data m4l plot in the original analysis style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = predictions[predictions.score > threshold]
    bins = np.arange(80, 252.5, 2.5); centers = (bins[:-1]+bins[1:])/2
    backgrounds = selected[selected.role == "background"]
    signals = selected[selected.role == "signal"]
    observed = selected[selected.role == "data"]
    background_groups = [(name, group) for name, group in backgrounds.groupby("sample", sort=False)]
    arrays = [group.mass.to_numpy() for _, group in background_groups]
    weights = [group.weight.to_numpy() for _, group in background_groups]
    colors = ["#6555d9", "#ff1717", "#bd63c5", "#e09b32"][:len(arrays)]
    labels = [name for name, _ in background_groups]
    if len(signals):
        arrays.append(signals.mass.to_numpy()); weights.append(signals.weight.to_numpy())
        colors.append("#11bce5"); labels.append("Signal ($m_H$ = 125 GeV)")
    fig, ax = plt.subplots(figsize=(13, 8))
    if arrays:
        ax.hist(arrays, bins=bins, weights=weights, stacked=True, color=colors,
                label=labels, histtype="stepfilled")
    mc = selected[selected.role != "data"]
    sumw = np.histogram(mc.mass, bins=bins, weights=mc.weight)[0]
    sumw2 = np.histogram(mc.mass, bins=bins, weights=np.square(mc.weight))[0]
    uncertainty = np.sqrt(sumw2)
    ax.bar(centers, 2*uncertainty, bottom=sumw-uncertainty, width=np.diff(bins),
           fill=False, hatch="////", edgecolor="black", linewidth=0,
           label="Stat. Unc.")
    counts = np.histogram(observed.mass, bins=bins)[0]
    ax.errorbar(centers, counts, yerr=np.sqrt(counts), fmt="o", color="black",
                markersize=4, label=f"Real Data (score > {threshold:.2f})")
    ax.set(xlim=(80, 250), ylim=(0, None), xlabel=r"$m_{4\ell}$ [GeV]",
           ylabel="Events / 2.5 GeV",
           title=rf"$H \rightarrow ZZ^* \rightarrow 4\ell$ — Real data after {model_name} cut (score > {threshold:.2f})")
    ax.minorticks_on(); ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

def save_observed_method_comparison(results, path, threshold, mass_window):
    """Reproduce the original two-panel background/significance comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {(row.stage, row.method): row for row in results.itertuples()}
    rows = [lookup[("before_ml", "mc_prediction")],
            lookup[("after_ml", "mc_prediction")],
            lookup[("after_ml", "sideband")]]
    labels = ["No ML", "MC Pred", "Sideband"]
    colors = ["#fb8d8f", "#ffe04b", "#afd7c5"]
    backgrounds = np.array([row.background for row in rows], float)
    errors = np.array([row.sigma_background for row in rows], float)
    z_values = np.array([row.Z for row in rows], float)
    z_errors = np.array([row.sigma_Z for row in rows], float)
    observed_no_ml, observed_ml = rows[0].N_observed, rows[1].N_observed
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(rf"$H \rightarrow ZZ^* \rightarrow 4\ell$, $\sqrt{{s}}=13$ TeV, "
                 rf"$\int L\,dt=36.6$ fb$^{{-1}}$\nBDT Threshold = {threshold:.2f}", weight="bold")
    x = np.arange(3)
    bars = axes[0].bar(x, backgrounds, yerr=errors, capsize=5, color=colors, edgecolor="#444")
    axes[0].axhline(observed_ml, color="red", linestyle="--", label=f"Observed (with ML): {observed_ml}")
    axes[0].axhline(observed_no_ml, color="black", linestyle=":", label=f"Observed (no ML): {observed_no_ml}")
    axes[0].set(xticks=x, xticklabels=labels,
                ylabel=f"Events in {mass_window[0]:g}–{mass_window[1]:g} GeV window",
                title="Background Estimates vs Observed Data")
    axes[0].legend(); axes[0].grid(axis="y", alpha=.25)
    for bar, value in zip(bars, backgrounds):
        axes[0].text(bar.get_x()+bar.get_width()/2, value+max(errors)*.5, f"{value:.1f}", ha="center")
    bars = axes[1].bar(x, z_values, yerr=z_errors, capsize=5, color=colors, edgecolor="#444")
    axes[1].axhline(5, color="red", linestyle="--", label="Discovery (5σ)")
    axes[1].axhline(3, color="#f5a623", linestyle="--", label="Evidence (3σ)")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Significance Z (σ)",
                title="Significance from Different Methods")
    axes[1].legend(); axes[1].grid(axis="y", alpha=.25)
    for bar, value in zip(bars, z_values):
        if np.isfinite(value): axes[1].text(bar.get_x()+bar.get_width()/2, value+.15, f"{value:.2f}", ha="center", weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
