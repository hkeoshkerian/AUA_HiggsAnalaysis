"""Local Streamlit working surface for the Higgs Analysis Lab."""
from dataclasses import replace
from pathlib import Path

import pandas as pd


def main():
    import streamlit as st
    from higgs_lab.config import load_config

    st.set_page_config(page_title="Higgs Analysis Lab", page_icon="⚛", layout="wide")
    st.markdown("""
    <style>
    :root { --ink:#10212b; --cyan:#00a6c7; --paper:#f4f7f8; }
    .stApp { background:linear-gradient(135deg,#f8fbfc 0%,#eef4f5 100%); color:var(--ink); }
    [data-testid="stSidebar"] { background:#10212b; }
    [data-testid="stSidebar"] * { color:#eef8fa; }
    h1,h2,h3 { letter-spacing:-.025em; }
    div[data-testid="stMetric"] { background:white; border:1px solid #dce7e9; padding:14px; border-radius:8px; }
    .warning { border-left:4px solid #db8b18; background:#fff8e9; padding:10px 14px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Higgs Analysis Lab")
    st.caption("A local, reproducible workspace for four-lepton reconstruction, model studies, and guarded counting inference.")
    st.markdown('<div class="warning">Results remain scientifically provisional until validated against real ATLAS reference yields.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Workspace")
        config_path = st.text_input("Configuration", "configs/default.toml")
        prepared_path = st.text_input("Prepared dataset", "prepared/full")
        results_root = st.text_input("Results folder", "results")
        try:
            config = load_config(config_path)
            st.success("Configuration valid")
        except Exception as exc:
            st.error(str(exc)); st.stop()
        model = st.selectbox("Model", ["logistic_regression", "random_forest", "xgboost",
                                        "lightgbm", "mlp", "gaussian_nb", "qda"],
                             index=["logistic_regression", "random_forest", "xgboost",
                                    "lightgbm", "mlp", "gaussian_nb", "qda"].index(config.training.model))
        threshold = st.slider("Classifier threshold", .05, .95, float(config.training.threshold), .05)
        folds = st.number_input("Cross-validation folds", 2, 20, int(config.training.folds))
        seed = st.number_input("Random seed", 0, 2**31-1, int(config.training.seed))
        config = replace(config, training=replace(config.training, model=model,
                         threshold=threshold, folds=int(folds), seed=int(seed)))

    prepare_tab, train_tab, study_tab, inference_tab = st.tabs(
        ["Prepare", "Train", "Feature & threshold studies", "Observed summaries"])

    with prepare_tab:
        st.subheader("Prepare ROOT data")
        st.write("Resolve configured ATLAS samples, apply selections, reconstruct all 31 features, and create a verified cache.")
        destination = st.text_input("New prepared output", str(Path(prepared_path)), key="prepare_output")
        if st.button("Prepare dataset", type="primary"):
            from higgs_lab.pipeline import prepare
            with st.status("Preparing events…", expanded=True):
                try: st.success(f"Saved to {prepare(config, destination)}")
                except Exception as exc: st.exception(exc)

    with train_tab:
        st.subheader("Train and evaluate")
        run_output = st.text_input("New training output", str(Path(results_root)/model), key="run_output")
        st.write(f"Uses **{model}**, {int(folds)} folds, and {len(config.training.features)} configured features.")
        if st.button("Run training", type="primary"):
            from higgs_lab.pipeline import run
            with st.status("Training fold models…", expanded=True):
                try:
                    saved = run(config, prepared_path, run_output)
                    st.success(f"Saved to {saved}")
                except Exception as exc: st.exception(exc)
        summary_path = Path(run_output)/"summary.json"
        if summary_path.is_file():
            summary = pd.read_json(summary_path, typ="series")
            left, right = st.columns(2)
            left.metric("Weighted OOF AUC", f"{summary['weighted_oof_auc']:.4f}")
            right.metric("Threshold", f"{summary['threshold']:.2f}")
            for image_name in ("roc.png", "mass_before.png", "mass_after.png"):
                image_path = Path(run_output)/image_name
                if image_path.is_file(): st.image(str(image_path), caption=image_name.replace("_", " ").title())

    with study_tab:
        st.subheader("Diagnostics")
        prediction_path = st.text_input("Predictions CSV", str(Path(results_root)/model/"predictions.csv"), key="study_predictions")
        col1, col2 = st.columns(2)
        with col1:
            feature_output = st.text_input("Feature-study output", str(Path(results_root)/"feature-study"))
            if st.button("Analyze features"):
                from higgs_lab.feature_analysis import analyze_features
                try: st.success(f"Saved to {analyze_features(prepared_path, feature_output, int(seed), .7)}")
                except Exception as exc: st.exception(exc)
        with col2:
            threshold_output = st.text_input("Threshold-study output", str(Path(results_root)/f"{model}-thresholds"))
            if st.button("Scan thresholds"):
                from higgs_lab.thresholds import analyze_thresholds
                try:
                    values = [value/100 for value in range(10, 96, 5)]
                    saved = analyze_thresholds(prediction_path, threshold_output, values,
                        config.statistics.mass_window_gev,
                        config.statistics.background_fractional_systematic)
                    st.success(f"Saved to {saved}")
                except Exception as exc: st.exception(exc)

    with inference_tab:
        st.subheader("Observed counting summaries")
        st.warning("Run only on the full prepared dataset. These are approximate counting and sideband summaries, not a likelihood fit.")
        prediction_path = st.text_input("Predictions CSV", str(Path(results_root)/model/"predictions.csv"), key="inference_predictions")
        inference_output = st.text_input("New inference output", str(Path(results_root)/f"{model}-observed"))
        if st.button("Calculate observed summaries", type="primary"):
            from higgs_lab.inference import analyze_observed
            try:
                saved = analyze_observed(prediction_path, inference_output, threshold,
                    config.statistics.mass_window_gev, (90.,105.,140.,155.),
                    config.statistics.background_fractional_systematic, config.data.fraction)
                st.success(f"Saved to {saved}")
                st.dataframe(pd.read_csv(Path(saved)/"observed_methods.csv"), use_container_width=True)
            except Exception as exc: st.exception(exc)


if __name__ == "__main__":
    main()
