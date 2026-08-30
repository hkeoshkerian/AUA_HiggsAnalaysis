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

def save_feature_plots(frame, output, features):
    """Save weighted MC distributions and a signal correlation matrix."""
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for feature in features:
        values = frame[feature].to_numpy(float)
        low, high = np.nanquantile(values, [.01, .99])
        if low == high: low, high = low-.5, high+.5
        bins = np.linspace(low, high, 41)
        fig, ax = plt.subplots(figsize=(7, 5))
        for role, color in (("background", "tab:blue"), ("signal", "tab:red")):
            part = frame[frame.role == role]
            ax.hist(part[feature], bins=bins, weights=part.weight, histtype="step",
                    linewidth=1.7, label=role.title(), color=color)
        ax.set(xlabel=feature, ylabel="Weighted MC events", title=f"{feature} distribution")
        ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(output/f"{feature}.png", dpi=150); plt.close(fig)

    signal = frame[frame.role == "signal"]
    correlation = signal[list(features)].corr().fillna(0.)
    size = max(8, min(18, len(features)*.48))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(features)), features, rotation=90, fontsize=7)
    ax.set_yticks(range(len(features)), features, fontsize=7)
    ax.set_title("Signal feature correlations")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout(); fig.savefig(output/"signal_correlations.png", dpi=150); plt.close(fig)

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
