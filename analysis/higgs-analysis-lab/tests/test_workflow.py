"""Local tests use synthetic data, never represent ATLAS physics validation."""
import ast
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from higgs_lab.config import Config, load_config
from higgs_lab.features import validate_frame
from higgs_lab.provenance import preparation_settings, sha256, write_json
from higgs_lab.statistics import (weighted_yield, expected_profile_result,
                                  expected_profile_likelihood,
                                  profile_likelihood_discovery)
from higgs_lab.training import train_models, _outer_splits
from higgs_lab.pipeline import run
from higgs_lab.feature_analysis import separation_power, rank_features, prune_correlations
from higgs_lab.thresholds import scan_thresholds, optimal_threshold
from higgs_lab.stability import run_stability
from higgs_lab.plots import save_feature_plots, save_preselection_mass_plot
from higgs_lab.inference import observed_methods
from higgs_lab.comparison import (compare_prediction_frames,
                                  compare_selection_strategies,
                                  scan_signal_efficiencies)
from higgs_lab.features import FEATURE_KEYS
from higgs_lab.shape_likelihood import (MassShapeLikelihood, TemplateCategory,
                                        build_templates,
                                        compare_mass_shape_models)
from higgs_lab.mass_shape_validation import validate_mass_shapes
from higgs_lab.robust import (ROBUST_FEATURES, effective_events,
                              validate_robust_models)
from higgs_lab.primary_inference import (fixed_signal_efficiency_thresholds,
                                         run_primary_inference)
from higgs_lab.information_gain import _score_categories

ROOT = Path(__file__).resolve().parents[1]
HAS_ROOT = all(importlib.util.find_spec(x) is not None for x in ['awkward','vector','uproot'])

def fixture():
    rng = np.random.default_rng(6)
    y = np.r_[np.zeros(40,dtype=int),np.ones(40,dtype=int),np.full(8,-1)]
    frame = pd.DataFrame({'event_id':np.arange(len(y)), 'label':y,
        'sample':np.where(y==1,'signal',np.where(y==0,'background','data')),
        'role':np.where(y==1,'signal',np.where(y==0,'background','data')),
        'weight':np.where(y==-1,1.0,.1),
        'mass':rng.uniform(105,140,len(y)), 'mz1':rng.normal(85+5*y,3),
        'mz2':rng.normal(30+5*y,5)})
    config = Config(training=replace(Config().training,features=('mz1','mz2'),folds=4))
    return frame,config

def make_cache(directory,frame,config):
    directory.mkdir()
    frame.to_csv(directory/'features.csv',index=False)
    write_json(directory/'manifest.json',{'schema_version':1,'energy_unit':'GeV',
        'settings':preparation_settings(config),'features_sha256':sha256(directory/'features.csv'),
        'test_fixture':'SYNTHETIC, NOT ATLAS DATA'})

class CoreTests(unittest.TestCase):
    def test_threshold_scan_uses_oof_mc(self):
        frame, _ = fixture()
        frame["score"] = np.where(frame.label == 1, .8, np.where(frame.label == 0, .2, .99))
        frame.loc[frame.label == 0, "score"] = np.r_[.6, np.full(39, .2)]
        frame["score_kind"] = np.where(
            frame.role == "data", "fold_assigned_calibrated",
            "out_of_fold_calibrated")
        scan = scan_thresholds(frame, [.1, .5, .9], (105, 140), .3)
        self.assertEqual(len(scan), 3)
        self.assertAlmostEqual(scan.loc[scan.threshold == .5, "signal_efficiency"].iloc[0], 1.)
        self.assertAlmostEqual(scan.loc[scan.threshold == .5, "background_rejection"].iloc[0], 39/40)
        self.assertEqual(optimal_threshold(scan)["threshold"], .5)
        bad = frame.copy(); bad.loc[bad.role != "data", "score_kind"] = "in_sample"
        with self.assertRaisesRegex(ValueError, "out-of-fold"):
            scan_thresholds(bad, [.5])

    def test_feature_ranking_and_correlation_pruning(self):
        frame, _ = fixture()
        rng = np.random.default_rng(17)
        for feature in FEATURE_KEYS:
            if feature not in frame:
                frame[feature] = rng.normal(size=len(frame)) + .2 * frame.label.clip(lower=0)
        ranking, training = rank_features(frame, seed=9)
        self.assertEqual(set(ranking.feature), set(FEATURE_KEYS))
        self.assertTrue(np.isfinite(ranking[["separation_power", "single_feature_auc", "rf_importance"]]).all().all())
        scores = dict(zip(ranking.feature, ranking.separation_power))
        kept, dropped, decisions = prune_correlations(training, scores, threshold=.7)
        self.assertEqual(set(kept) | set(dropped), set(FEATURE_KEYS))
        self.assertTrue(set(decisions["drop"]).issubset(set(dropped)))
        self.assertEqual(separation_power([1, 1], [1, 1], [1, 1], [1, 1]), 0.)

    def test_default_configuration(self):
        self.assertEqual(load_config(ROOT/'configs/default.toml').data.luminosity_fb,36.6)

    def test_bad_configuration(self):
        with self.assertRaises(ValueError):
            replace(Config(),training=replace(Config().training,threshold=1.2)).validate()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.toml'; path.write_text('[data]\nluminositty_fb = 36.6\n')
            with self.assertRaises(ValueError): load_config(path)

    def test_atlas_selection_accepts_both_four_lepton_skims(self):
        for skim in ("4lep", "exactly4lep"):
            config = replace(
                Config(),
                data=replace(Config().data, skim=skim),
                selection=replace(Config().selection, mode="atlas_2017_fiducial"),
            )
            self.assertEqual(config.validate().data.skim, skim)

    def test_yield_window_boundaries_and_sumw2(self):
        self.assertEqual(weighted_yield([109,110,134.9,135],[9,3,4,9],(110,135)),(7.,5.))

    def test_expected_profile_result_includes_systematic_and_stat_error(self):
        result=expected_profile_result(12,30,2,3,.30)
        constraint=np.hypot(3,.30*30)
        self.assertAlmostEqual(result['background_constraint_sigma'],constraint)
        self.assertAlmostEqual(result['expected_profile_Z'],
                               expected_profile_likelihood(12,30,constraint))
        self.assertGreater(result['sigma_expected_profile_Z'],0)
        self.assertIsNone(expected_profile_result(12,0,2,0)['expected_profile_Z'])

    def test_one_bin_profile_likelihood(self):
        result = profile_likelihood_discovery(15, 10, 0)
        expected_q0 = 2*(15*np.log(1.5)-5)
        self.assertAlmostEqual(result["profile_q0"], expected_q0)
        self.assertAlmostEqual(result["profile_Z"], np.sqrt(expected_q0))
        self.assertEqual(profile_likelihood_discovery(
            8, 10, 2)["profile_Z"], 0.)
        self.assertGreater(expected_profile_likelihood(5, 10, 2), 0.)

    def test_mass_shape_profile_likelihood(self):
        category = TemplateCategory(
            "inclusive", np.array([11., 20., 10.]),
            np.array([0., 8., 0.]), np.array([10., 10., 10.]),
            np.array([1., 1., 1.]))
        likelihood = MassShapeLikelihood([category], {"test": .05})
        observed = likelihood.discovery([category.observed])
        expected = likelihood.expected()
        self.assertGreater(observed["Z"], 0.)
        self.assertGreater(expected["Z"], 0.)
        self.assertGreater(observed["mu_hat"], 0.)
        interval68 = likelihood.profile_mu_interval([category.observed], 1.)
        interval95 = likelihood.profile_mu_interval([category.observed], 3.841)
        self.assertLessEqual(interval68["low"], observed["mu_hat"])
        self.assertGreaterEqual(interval68["high"], observed["mu_hat"])
        self.assertLessEqual(interval95["low"], interval68["low"])
        self.assertGreaterEqual(interval95["high"], interval68["high"])
        background_only = likelihood.discovery([category.background])
        self.assertAlmostEqual(background_only["Z"], 0., places=5)
        statistical = likelihood.statistical_z_ensemble(20, seed=7)
        self.assertGreaterEqual(statistical["expected"]["sigma"], 0.)
        self.assertGreaterEqual(statistical["observed"]["sigma"], 0.)

    def test_mass_shape_templates_keep_pass_and_fail(self):
        frame, _ = fixture()
        frame["score"] = np.resize([.2, .8], len(frame))
        frame["score_kind"] = np.where(
            frame.role == "data", "fold_assigned_calibrated",
            "out_of_fold_calibrated")
        edges = np.linspace(105, 140, 8)
        categories = build_templates(frame, edges, threshold=.5)
        self.assertEqual({x.name for x in categories}, {"score_pass", "score_fail"})
        self.assertEqual(sum(x.observed.sum() for x in categories), 8)
        self.assertAlmostEqual(sum(x.signal.sum() for x in categories), 4.)

    def test_mass_shape_workflow_writes_separate_outputs(self):
        frame, _ = fixture()
        frame["score"] = np.resize([.2, .8], len(frame))
        frame["score_kind"] = np.where(
            frame.role == "data", "fold_assigned_calibrated",
            "out_of_fold_calibrated")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            predictions = base/"predictions.csv"
            frame.to_csv(predictions, index=False)
            output = compare_mass_shape_models(
                {"Test": predictions}, {"Test": .5}, base/"shape",
                fit_range=(105., 140.), bin_width=5., toys=0)
            summary = pd.read_csv(output/"mass_shape_significance.csv")
            self.assertEqual(set(summary.model), {"No ML", "Test"})
            self.assertTrue(np.isfinite(summary[[
                "observed_Z", "expected_Z", "observed_p_value",
                "expected_p_value"]]).all().all())
            self.assertGreater((output/"mass_shape_significance.png").stat().st_size, 0)
            self.assertGreater(
                (output/"expected_mc_significance_forest.png").stat().st_size, 0)
            self.assertGreater(
                (output/"observed_mc_background_significance_forest.png").stat().st_size, 0)
            self.assertGreater(
                (output/"observed_signal_strength_forest.png").stat().st_size, 0)

    def test_mass_shape_validation_writes_all_eight_diagnostics(self):
        frame, _ = fixture()
        frame["score"] = np.where(
            frame.label == 1, np.resize([.45, .8, .9], len(frame)),
            np.where(frame.label == 0,
                     np.resize([.1, .3, .7], len(frame)),
                     np.resize([.2, .6, .85], len(frame))))
        frame["score_kind"] = np.where(
            frame.role == "data", "fold_assigned_calibrated",
            "out_of_fold_calibrated")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            predictions = base/"predictions.csv"
            frame.to_csv(predictions, index=False)
            output = validate_mass_shapes(
                {"Test": predictions}, {"Test": .5}, base/"validation",
                fit_range=(105., 140.), signal_window=(118., 130.),
                sidebands=(105., 118., 130., 140.), bin_width=5.,
                gof_toys=20, seed=7)
            expected = {
                "test_prefit_residuals.csv", "test_prefit_residuals.png",
                "test_sideband_score_validation.png",
                "test_background_components.png",
                "goodness_of_fit_summary.csv",
                "sideband_pass_fail_closure.csv",
                "sideband_pass_fail_closure.png",
                "binning_stability.csv", "binning_stability.png",
                "likelihood_ablation.csv", "likelihood_ablation.png",
                "validation_method.json",
            }
            self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))
            method = json.loads((output/"validation_method.json").read_text())
            self.assertEqual(method["signal_window"], [118.0, 130.0])
            closure = pd.read_csv(output/"sideband_pass_fail_closure.csv")
            summary = closure[closure.source.eq("closure")]
            self.assertEqual(set(summary.sideband_region),
                             {"combined", "lower", "upper"})

    def test_robust_validation_is_blinded_and_may_refuse_working_point(self):
        frame, _ = fixture()
        for index, feature in enumerate(ROBUST_FEATURES):
            frame[feature] = np.sin(np.arange(len(frame))*.07+index)
        frame["score"] = np.where(
            frame.label == 1, np.resize([.4, .8, .95], len(frame)),
            np.where(frame.label == 0, np.resize([.1, .3, .7], len(frame)),
                     np.resize([.2, .5, .85], len(frame))))
        frame["raw_score"] = frame.score
        frame["fold"] = np.arange(len(frame)) % 4
        frame["score_kind"] = np.where(
            frame.role == "data", "fold_assigned_calibrated",
            "out_of_fold_calibrated")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); path = base/"predictions.csv"
            frame.to_csv(path, index=False)
            output = validate_robust_models(
                {"Test": path}, base/"robust", efficiencies=(.7, .8),
                minimum_background_neff=1e9)
            choices = pd.read_csv(output/"chosen_operating_points.csv")
            self.assertTrue(choices.status.str.startswith("no operating").all())
            method = json.loads((output/"robust_method.json").read_text())
            self.assertFalse(method["observed_signal_region_used"])
            self.assertEqual(method["features"], list(ROBUST_FEATURES))
            self.assertGreater((output/"robust_operating_point_scan.png").stat().st_size, 0)

    def test_robust_configs_share_mass_decorrelated_features(self):
        for model in ("xgboost", "lightgbm", "random_forest", "mlp"):
            config = load_config(ROOT/f"configs/robust_{model}.toml")
            self.assertEqual(config.training.model, model)
            self.assertEqual(tuple(config.training.features), ROBUST_FEATURES)
            self.assertEqual(config.data.skim, "4lep")
            self.assertEqual(config.data.luminosity_fb, 36.1)
            self.assertEqual(config.data.weight_mode, "legacy_absolute")
            self.assertEqual(config.selection.mode, "atlas_2017_fiducial")
            self.assertTrue(config.training.mass_window_only)
            self.assertEqual(tuple(config.statistics.mass_window_gev),
                             (115., 130.))

    def test_effective_events_detects_dominant_weight(self):
        self.assertAlmostEqual(effective_events([1., 1., 1.]), 3.)
        self.assertLess(effective_events([100., 1., 1.]), 1.1)

    def test_primary_inference_uses_pass_only_in_fixed_mass_window(self):
        frame, _ = fixture()
        frame["score"] = np.where(
            frame.label == 1, np.resize([.4, .8, .9], len(frame)),
            np.where(frame.label == 0, np.resize([.1, .3, .7], len(frame)),
                     np.resize([.2, .6, .85], len(frame))))
        frame["raw_score"] = frame.score
        frame["fold"] = np.arange(len(frame)) % 4
        frame["score_kind"] = np.where(
            frame.role == "data", "fold_assigned_calibrated",
            "out_of_fold_calibrated")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); predictions = base/"predictions.csv"
            frame.to_csv(predictions, index=False)
            output = run_primary_inference(
                {"Test": predictions}, {"Test": .5}, base/"primary",
                fit_range=(115., 130.), bin_width=5.,
                background_systematic=.30, statistical_repeats=20)
            summary = pd.read_csv(output/"primary_score_pass_inference.csv")
            selected = summary[summary.model.eq("Test")].iloc[0]
            expected_pass = frame[(frame.role == "data") & (frame.score > .5) &
                                  (frame.mass >= 115) & (frame.mass < 130)]
            self.assertEqual(selected.observed_events, len(expected_pass))
            method = json.loads((output/"primary_inference_method.json").read_text())
            self.assertEqual(method["primary_category"], "score_pass only")
            self.assertEqual(method["score_fail_category"],
                             "excluded from primary inference")
            self.assertEqual(method["fit_range_gev"], [115., 130.])
            self.assertEqual(method["normalization_uncertainties"],
                             {"background_systematic": .30})
            self.assertTrue((output/"primary_operating_points.csv").is_file())

    def test_model_specific_signal_efficiency_thresholds_use_oof_mc(self):
        frame, _ = fixture()
        frame["score"] = np.linspace(0., 1., len(frame))
        frame["score_kind"] = np.where(
            frame.role.eq("data"), "fold_assigned_calibrated",
            "out_of_fold_calibrated")
        frame["mass"] = 124.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"predictions.csv"
            frame.to_csv(path, index=False)
            threshold = fixed_signal_efficiency_thresholds(
                {"Test": path}, .8)["Test"]
            signal = frame[frame.role.eq("signal")]
            achieved = signal.loc[signal.score > threshold, "weight"].sum()/signal.weight.sum()
            self.assertAlmostEqual(achieved, .8, delta=.03)

    def test_reconstruction_body_preserved(self):
        source=ast.parse((ROOT/'reference/higgs_analysis_original.py').read_text())
        target=ast.parse((ROOT/'src/higgs_lab/reconstruction.py').read_text())
        for name in ['reconstruct_z1_z2_fast','calc_mass','opening_dphi','delta_r']:
            old=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
            new=[n for n in target.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
            self.assertEqual(ast.dump(old),ast.dump(new))

    def test_data_cannot_be_labelled_background(self):
        frame,config=fixture(); frame.loc[frame.role=='data','label']=0
        with self.assertRaises(ValueError): validate_frame(frame,config.training.features)

    def test_forbidden_training_columns(self):
        frame,config=fixture()
        for feature in ['mass','label','weight','sample','event_id']:
            with self.assertRaises(ValueError): validate_frame(frame,[feature])

    def test_oof_coverage_and_no_observed_data_leakage(self):
        frame,config=fixture()
        first=train_models(frame,config)
        changed=frame.copy(); changed.loc[changed.role=='data',['mz1','mz2']]=1e6
        second=train_models(changed,config)
        mc=first.predictions[first.predictions.role!='data']
        np.testing.assert_allclose(mc.score,second.predictions.query("role != 'data'").score)
        self.assertEqual(set(mc.fold),set(range(config.training.folds)))
        self.assertTrue(mc.score.notna().all())
        observed = first.predictions.query("role == 'data'")
        self.assertTrue(observed.fold.between(0, config.training.folds-1).all())
        self.assertEqual(set(observed.score_kind), {"fold_assigned_calibrated"})
        self.assertEqual(set(mc.score_kind), {"out_of_fold_calibrated"})

    def test_negative_weights_rejected(self):
        frame,config=fixture(); frame.loc[0,'weight']=-.1
        with self.assertRaisesRegex(ValueError,'Negative MC weights'): train_models(frame,config)

    def test_group_aware_outer_cv_has_no_group_leakage(self):
        # Six independent sources per class, with multiple events per source.
        y = np.repeat(np.r_[np.zeros(6, dtype=int), np.ones(6, dtype=int)], 4)
        groups = np.repeat([f"b{i}" for i in range(6)] +
                           [f"s{i}" for i in range(6)], 4)
        X = np.arange(len(y))[:, None]
        splits = _outer_splits(X, y, groups, 3, 17, True)
        self.assertEqual(len(splits), 3)
        for train_idx, test_idx in splits:
            self.assertFalse(set(groups[train_idx]) & set(groups[test_idx]))
            self.assertEqual(set(y[test_idx]), {0, 1})

    def test_group_splits_handle_highly_unequal_source_sizes(self):
        y, groups = [], []
        for label, prefix, sizes in ((0, "b", [6000, 2800, 800, 400, 100]),
                                     (1, "s", [178000, 41000, 2800, 2200, 1])):
            for index, size in enumerate(sizes):
                y.extend([label] * size)
                groups.extend([f"{prefix}{index}"] * size)
        y, groups = np.asarray(y), np.asarray(groups)
        splits = _outer_splits(np.zeros((len(y), 1)), y, groups, 5, 42, True)
        for train_idx, test_idx in splits:
            self.assertEqual(set(y[test_idx]), {0, 1})
            self.assertFalse(set(groups[train_idx]) & set(groups[test_idx]))

    def test_group_aware_training_requires_prepared_group_column(self):
        frame, config = fixture()
        config = replace(config, training=replace(
            config.training, group_aware=True, group_column="source_id"))
        with self.assertRaisesRegex(ValueError, "not the nominal analysis"):
            config.validate()

    def test_random_forest_adapter(self):
        frame,config=fixture(); config=replace(config,training=replace(config.training,model='random_forest',folds=2,nested_tuning=True))
        result = train_models(frame,config)
        self.assertTrue(np.isfinite(result.oof_auc))
        self.assertEqual(len(result.tuning), 2)
        for fold, tuning in zip(result.fold_metrics, result.tuning):
            self.assertEqual(fold["max_depth"], tuning["best_max_depth"])
            self.assertEqual(fold["min_samples_leaf"],
                             tuning["best_min_samples_leaf"])

    def test_nested_logistic_regression_uses_fold_selected_c(self):
        frame, config = fixture()
        config = replace(config, training=replace(
            config.training, model="logistic_regression", folds=2,
            nested_tuning=True))
        result = train_models(frame, config)
        self.assertEqual(len(result.tuning), 2)
        for fold, tuning in zip(result.fold_metrics, result.tuning):
            self.assertEqual(fold["C"], tuning["best_C"])
            self.assertEqual(tuning["inner_folds"], 3)

    def test_gaussian_nb_and_qda_adapters(self):
        frame, config = fixture()
        for name in ("gaussian_nb", "qda"):
            chosen = replace(config, training=replace(config.training, model=name, folds=2))
            self.assertTrue(np.isfinite(train_models(frame, chosen).oof_auc))

    def test_nested_qda_and_gaussian_nb_use_selected_parameters(self):
        frame, config = fixture()
        keys = {"qda": ("reg_param", "best_reg_param"),
                "gaussian_nb": ("var_smoothing", "best_var_smoothing")}
        for name, (fold_key, tuning_key) in keys.items():
            chosen = replace(config, training=replace(
                config.training, model=name, folds=2, nested_tuning=True))
            result = train_models(frame, chosen)
            self.assertEqual(len(result.tuning), 2)
            for fold, tuning in zip(result.fold_metrics, result.tuning):
                self.assertEqual(fold[fold_key], tuning[tuning_key])

    def test_stability_study(self):
        frame, config = fixture()
        results = run_stability(frame, config, seeds=[3, 4], folds=[2])
        self.assertEqual(len(results), 2)
        self.assertEqual(set(results.seed), {3, 4})
        self.assertTrue(np.isfinite(results.weighted_oof_auc).all())

    def test_feature_diagnostic_plots(self):
        frame, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            save_feature_plots(frame, directory, ["mz1", "mz2"])
            save_preselection_mass_plot(frame, Path(directory)/"pre_ml_m4l.png", include_data=True)
            self.assertGreater((Path(directory)/"mz1.png").stat().st_size, 0)
            self.assertGreater((Path(directory)/"signal_correlations.png").stat().st_size, 0)
            self.assertGreater((Path(directory)/"pre_ml_m4l.png").stat().st_size, 0)

    def test_observed_mc_and_sideband_methods(self):
        frame, _ = fixture()
        frame["score"] = np.where(frame.label == 1, .8, np.where(frame.label == 0, .2, .7))
        frame["score_kind"] = np.where(
            frame.role == "data", "fold_assigned_calibrated",
            "out_of_fold_calibrated")
        # Put observed events into both the signal region and sidebands.
        data_indices = frame.index[frame.role == "data"]
        frame.loc[data_indices, "mass"] = [112, 120, 130, 100, 102, 145, 150, 160]
        results = observed_methods(frame, .5, (110, 135), (90, 105, 140, 155), .3)
        self.assertEqual(set(results.method), {"mc_prediction", "sideband"})
        self.assertEqual(set(results.stage), {"before_ml", "after_ml"})
        self.assertEqual(results.query("stage == 'after_ml' and method == 'sideband'").N_observed.iloc[0], 3)
        self.assertIn("profile_Z", results)
        self.assertTrue(results.query("stage == 'before_ml'").profile_Z.notna().all())

    def test_model_comparison(self):
        frame, _ = fixture()
        frame["score"] = np.where(frame.label == 1, .8, np.where(frame.label == 0, .2, .7))
        frame["score_kind"] = np.where(
            frame.role == "data", "fold_assigned_calibrated",
            "out_of_fold_calibrated")
        frame["fold"] = np.resize(np.arange(4), len(frame))
        results, observed = compare_prediction_frames(
            {"a": frame, "b": frame.copy()}, {"a": .5, "b": .5},
            mass_window=(105, 140), bootstrap_repeats=20, include_observed=False)
        self.assertEqual(set(results.model), {"a", "b"})
        self.assertTrue((results.weighted_oof_auc == 1).all())
        self.assertTrue(observed.empty)

    def test_raw_cut_vs_fixed_efficiency_table(self):
        frame, config = fixture()
        frame["raw_score"] = np.where(frame.label == 1, .8, .3)
        frame["score"] = np.where(frame.label == 1, .8, .1)
        frame["score_kind"] = np.where(
            frame.role == "data", "fold_assigned_calibrated",
            "out_of_fold_calibrated")
        table = compare_selection_strategies(
            {"test": frame}, config.training.threshold,
            mass_window=(105, 140), systematic=.3)
        self.assertEqual(len(table), 2)
        self.assertEqual(set(table.selection_strategy), {
            "global raw score > 0.65",
            "fold-local 80% signal efficiency"})
        self.assertTrue((table.oof_signal_efficiency == 1).all())
        self.assertTrue((table.oof_background_rejection == 1).all())

    def test_signal_efficiency_scan_is_oof_mc_only(self):
        frame, _ = fixture()
        frame["score"] = .99
        frame.loc[frame.label == 1, "score"] = np.linspace(.45, .95, 40)
        frame.loc[frame.label == 0, "score"] = np.linspace(.05, .55, 40)
        frame["score_kind"] = np.where(
            frame.role == "data", "fold_assigned_calibrated",
            "out_of_fold_calibrated")
        frame["fold"] = np.resize(np.arange(4), len(frame))
        aggregate, folds, summary, chosen = scan_signal_efficiencies(
            {"test": frame}, [.6, .8], mass_window=(105, 140), systematic=.3)
        self.assertEqual(len(aggregate), 2)
        self.assertEqual(set(folds.fold), {0, 1, 2, 3})
        self.assertEqual(set(summary.target_signal_efficiency), {.6, .8})
        self.assertIn(chosen["target_signal_efficiency"], {.6, .8})
        self.assertTrue(aggregate.background_rejection.between(0, 1).all())
        self.assertTrue(aggregate.expected_profile_Z.notna().all())

    def test_pipeline_and_cache_guards(self):
        frame,config=fixture()
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); cache=base/'prepared'; make_cache(cache,frame,config)
            output=run(config,cache,base/'run')
            for name in ['predictions.csv','summary.json','config.json','provenance.json','mass_before.png','mass_after.png','roc.png']:
                self.assertGreater((output/name).stat().st_size,0)
            summary=json.loads((output/'summary.json').read_text())
            self.assertIn('baseline',summary)
            with self.assertRaises(FileExistsError): run(config,cache,output)
            other=replace(config,data=replace(config.data,luminosity_fb=1))
            with self.assertRaisesRegex(ValueError,'Preparation settings differ'): run(other,cache,base/'other')
            with (cache/'features.csv').open('a') as f: f.write('\n')
            with self.assertRaisesRegex(ValueError,'checksum'): run(config,cache,base/'tampered')

    def test_partial_sample_suppresses_significance(self):
        from higgs_lab.statistics import summarize_mc
        frame,_=fixture()
        self.assertIsNone(summarize_mc(frame,(110,135),.3,fraction=.1)['expected_profile_Z'])

@unittest.skipUnless(HAS_ROOT,'Optional ROOT dependencies unavailable in test environment')
class RootTests(unittest.TestCase):
    def events(self):
        import awkward as ak
        return ak.Array({'lep_n':[4,4], 'lep_pt':[[45.,40.,20.,10.]]*2,
            'lep_eta':[[.1,-.1,.2,-.2]]*2,'lep_phi':[[0.,3.,1.,-2.]]*2,
            'lep_e':[[46.,41.,21.,11.]]*2,'lep_charge':[[1,-1,1,-1]]*2,
            'lep_type':[[11,11,13,13]]*2,'trigE':[True,False],'trigM':[False,False],
            'lep_isTrigMatched':[[True,False,False,False]]*2,
            'lep_isLooseID':[[True]*4]*2,'lep_isMediumID':[[True]*4]*2,
            'lep_isLooseIso':[[True]*4]*2,'jet_n':[0,1], 'met':[10.,15.],'met_phi':[.2,.4]})

    def test_selection_and_reconstruction(self):
        from higgs_lab.selection import select_events
        from higgs_lab.reconstruction import reconstruct_z1_z2_fast
        selected,cutflow=select_events(self.events(),Config(),'data')
        self.assertEqual(len(selected),1)
        rec=reconstruct_z1_z2_fast(selected)
        self.assertEqual(len(rec['mass']),1)
        self.assertEqual(rec['w'][0],1.)
        self.assertTrue(np.isfinite(rec['mz1']).all())

    def test_unsorted_leptons_are_sorted_with_fields_aligned(self):
        import awkward as ak
        from higgs_lab.selection import select_events
        events = self.events()
        permutation = ak.Array([[3, 0, 2, 1], [3, 0, 2, 1]])
        for name in [field for field in events.fields if field.startswith("lep_") and field != "lep_n"]:
            events = ak.with_field(events, events[name][permutation], name)
        sorted_config = replace(Config(), selection=replace(
            Config().selection, sort_leptons_by_pt=True))
        selected, _ = select_events(events, sorted_config, "data")
        self.assertEqual(len(selected), 1)
        np.testing.assert_allclose(ak.to_numpy(selected.lep_pt[0]), [45., 40., 20., 10.])
        np.testing.assert_array_equal(ak.to_numpy(selected.lep_charge[0]), [1, -1, 1, -1])

    def test_weight_modes(self):
        import awkward as ak
        from higgs_lab.selection import calculate_weights, WEIGHT_BRANCHES
        fields={key:[1.,1.] for key in WEIGHT_BRANCHES}
        fields['mcWeight']=[-1.,2.]; fields['sum_of_weights']=[1000.,1000.]
        events=ak.Array(fields)
        np.testing.assert_allclose(calculate_weights(events,1),[1,2])
        np.testing.assert_allclose(calculate_weights(events,1,'signed'),[-1,2])

    def test_normalization_file_audit_detects_absolute_weight_inflation(self):
        import awkward as ak
        import uproot
        from higgs_lab.normalization import _audit_file
        from higgs_lab.selection import WEIGHT_BRANCHES
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fixture_mc_123456.test.root"
            fields = {key: np.ones(3) for key in WEIGHT_BRANCHES}
            fields["mcWeight"] = np.array([1., -1., 2.])
            fields["xsec"] = np.full(3, 2.)
            fields["filteff"] = np.full(3, .5)
            fields["kfac"] = np.full(3, 1.2)
            fields["sum_of_weights"] = np.full(3, 10.)
            with uproot.recreate(path) as root_file:
                root_file["analysis"] = ak.Array(fields)
            metadata = {
                "cross_section_pb": 2., "genFiltEff": .5,
                "kFactor": 1.2, "sumOfWeights": 10.,
            }
            row = _audit_file(path, "test", "background", 1., metadata)
            self.assertTrue(row["catalogue_match"])
            self.assertAlmostEqual(row["negative_mcweight_fraction"], 1/3)
            self.assertAlmostEqual(row["absolute_to_signed_yield_ratio"], 2.)

    def test_score_categories_partition_all_events(self):
        rows = []
        for role in ("signal", "background", "data"):
            for score in (.1, .3, .55, .7, .9):
                for mass in (107.5, 112.5, 117.5, 122.5, 127.5, 132.5, 137.5):
                    rows.append({"role": role, "score": score, "mass": mass,
                                 "weight": 1.})
        frame = pd.DataFrame(rows)
        categories = _score_categories(
            frame, np.arange(105., 145., 5.), [0., .2, .5, .8, 1.], "score")
        self.assertEqual(len(categories), 4)
        self.assertEqual(sum(item.observed.sum() for item in categories), 35.)
        self.assertEqual(sum(item.signal.sum() for item in categories), 35.)
        self.assertEqual(sum(item.background.sum() for item in categories), 35.)
        self.assertTrue(all(np.all(item.background > 0) for item in categories))

    def test_prepare_to_run_with_local_root_samples(self):
        import awkward as ak
        import uproot
        from unittest.mock import patch
        from higgs_lab.pipeline import prepare
        from higgs_lab.selection import WEIGHT_BRANCHES
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            resolved=[]
            for role in ['signal','background','data']:
                events=ak.concatenate([self.events()]*20)
                if role != 'data':
                    for key in WEIGHT_BRANCHES:
                        events=ak.with_field(events,np.ones(len(events)),key)
                    events=ak.with_field(events,np.full(len(events),100000.),'sum_of_weights')
                path=base/f'{role}.root'
                with uproot.recreate(path) as f:
                    f['analysis']={key:events[key] for key in events.fields}
                resolved.append(({'name':role,'role':role},[str(path)]))
            config=Config()
            with patch('higgs_lab.datasets.resolve_samples',return_value=resolved):
                cache=prepare(config,base/'prepared')
            frame=pd.read_csv(cache/'features.csv')
            self.assertEqual(len(frame),60)
            self.assertTrue(frame.query("role == 'data'").weight.eq(1).all())
            output=run(config,cache,base/'results')
            self.assertTrue((output/'summary.json').is_file())

    def test_local_root_reader(self):
        import awkward as ak
        import uproot
        from higgs_lab.datasets import iter_batches
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'fixture.root'
            with uproot.recreate(path) as f: f['analysis']={k:self.events()[k] for k in self.events().fields}
            batches=list(iter_batches(str(path),Config(),'data'))
            self.assertEqual(sum(len(b) for b in batches),2)

if __name__=='__main__': unittest.main()
