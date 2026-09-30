"""Small student-facing command line entry point."""
import argparse
import importlib
import importlib.metadata
import sys
import numpy as np
from .config import load_config

def doctor(include_root=False):
    print(f"Python {sys.version.split()[0]}")
    failed = not ((3,11) <= sys.version_info[:2] < (3,13))
    if failed: print("FAIL: this package supports Python 3.11 or 3.12")
    checks = {"numpy":"numpy","pandas":"pandas","matplotlib":"matplotlib","scikit-learn":"sklearn"}
    if include_root:
        checks.update({"uproot":"uproot","awkward":"awkward","vector":"vector",
                       "atlasopenmagic":"atlasopenmagic","requests":"requests",
                       "aiohttp":"aiohttp"})
    for package,module in checks.items():
        try:
            importlib.import_module(module)
            print(f"OK   {package}: {importlib.metadata.version(package)}")
        except Exception as exc:
            failed = True
            print(f"FAIL {package}: {exc}")
    print("Import checks only; this does not verify remote data access or scientific results.")
    return 1 if failed else 0

def main(argv=None):
    parser = argparse.ArgumentParser(description="Higgs Analysis Lab — modular starter, not a paper reproduction")
    sub = parser.add_subparsers(dest="command",required=True)
    check = sub.add_parser("doctor",help="Check installed dependencies")
    check.add_argument("--root",action="store_true",help="Also check raw ROOT data dependencies")
    validate = sub.add_parser("check-config",help="Validate analysis settings without running")
    validate.add_argument("--config",default="configs/default.toml")
    normalization = sub.add_parser(
        "audit-normalization",
        help="Check luminosity, cross sections and ROOT normalization metadata")
    normalization.add_argument("--config", default="configs/default.toml")
    normalization.add_argument(
        "--output", required=True,
        help="New directory; existing directories are refused")
    weight_check = sub.add_parser(
        "compare-weight-conventions",
        help="Compare signed and legacy-absolute yields after event selection")
    weight_check.add_argument("--config", default="configs/default.toml")
    weight_check.add_argument("--output", required=True,
                              help="New directory; existing directories are refused")
    weight_check.add_argument("--fit-range", default="105,140")
    weight_check.add_argument("--signal-window", default="118,130")
    for name in ["prepare","run"]:
        command = sub.add_parser(name)
        command.add_argument("--config",default="configs/default.toml")
        command.add_argument("--output",required=True,help="New directory; existing directories are refused")
        if name == "run": command.add_argument("--prepared",required=True)
    features = sub.add_parser("analyze-features", help="Rank features in a prepared dataset")
    features.add_argument("--prepared", required=True)
    features.add_argument("--output", required=True, help="New directory; existing directories are refused")
    features.add_argument("--seed", type=int, default=42)
    features.add_argument("--correlation-threshold", type=float, default=.70)
    thresholds = sub.add_parser("scan-thresholds", help="Scan cuts on MC out-of-fold scores")
    thresholds.add_argument("--config", default="configs/default.toml")
    thresholds.add_argument("--predictions", required=True)
    thresholds.add_argument("--output", required=True, help="New directory; existing directories are refused")
    thresholds.add_argument("--start", type=float, default=.10)
    thresholds.add_argument("--stop", type=float, default=.95)
    thresholds.add_argument("--step", type=float, default=.05)
    stability = sub.add_parser("study-stability", help="Repeat training across seeds and fold counts")
    stability.add_argument("--config", default="configs/default.toml")
    stability.add_argument("--prepared", required=True)
    stability.add_argument("--output", required=True, help="New directory; existing directories are refused")
    stability.add_argument("--seeds", default="1,7,21,42,99")
    stability.add_argument("--folds", default="3,5,7,10")
    diagnostics = sub.add_parser("plot-features", help="Plot all reconstructed features")
    diagnostics.add_argument("--prepared", required=True)
    diagnostics.add_argument("--output", required=True, help="New directory; existing directories are refused")
    preselection = sub.add_parser("plot-preselection", help="Plot Data and stacked MC before machine learning")
    preselection.add_argument("--prepared", required=True)
    preselection.add_argument("--output", required=True, help="New directory; existing directories are refused")
    observed = sub.add_parser("infer-observed", help="Run guarded observed-count and sideband summaries")
    observed.add_argument("--config", default="configs/default.toml")
    observed.add_argument("--predictions", required=True)
    observed.add_argument("--output", required=True, help="New directory; existing directories are refused")
    observed.add_argument("--threshold", type=float)
    observed.add_argument("--sidebands", default="90,105,140,155")
    comparison = sub.add_parser("compare-models", help="Compare saved model prediction tables")
    comparison.add_argument("--config", default="configs/default.toml")
    comparison.add_argument("--prediction", action="append", required=True,
                            help="Repeat as MODEL=path/to/predictions.csv")
    comparison.add_argument("--threshold", action="append", default=[],
                            help="Optional repeat as MODEL=value; config threshold is the default")
    comparison.add_argument("--output", required=True, help="New directory; existing directories are refused")
    comparison.add_argument("--bootstrap-repeats", type=int, default=1000)
    comparison.add_argument(
        "--efficiencies", default="0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        help="Comma-separated common target signal efficiencies for the MC-only operating-point study")
    shape = sub.add_parser(
        "compare-mass-shapes",
        help="ATLAS-inspired binned m4l profile likelihood using MC background")
    shape.add_argument("--config", default="configs/default.toml")
    shape.add_argument("--prediction", action="append", required=True,
                       help="Repeat as MODEL=path/to/predictions.csv")
    shape.add_argument("--threshold", action="append", default=[],
                       help="Optional repeat as MODEL=value; config threshold is the default")
    shape.add_argument("--output", required=True,
                       help="New directory; existing directories are refused")
    shape.add_argument("--fit-range", default="105,160",
                       help="Lower,upper m4l fit range in GeV")
    shape.add_argument("--bin-width", type=float, default=2.5)
    shape.add_argument("--toys", type=int, default=0,
                       help="Optional conditional background-only toys (can be slow)")
    shape.add_argument("--stat-repeats", type=int, default=200,
                       help="Statistical template/data ensembles for Z error bars")
    shape.add_argument("--seed", type=int, default=42)
    validation = sub.add_parser(
        "validate-mass-shapes",
        help="Run residual, closure, GOF, binning and likelihood-ablation diagnostics")
    validation.add_argument("--config", default="configs/default.toml")
    validation.add_argument("--prediction", action="append", required=True,
                            help="Repeat as MODEL=path/to/predictions.csv")
    validation.add_argument("--threshold", action="append", default=[],
                            help="Repeat as MODEL=value; default is 0.20")
    validation.add_argument("--output", required=True,
                            help="New directory; existing directories are refused")
    validation.add_argument("--fit-range", default="105,140")
    validation.add_argument("--signal-window", default="118,130")
    validation.add_argument("--sidebands", default="105,118,130,140")
    validation.add_argument("--bin-width", type=float, default=2.5)
    validation.add_argument("--gof-toys", type=int, default=300)
    validation.add_argument("--seed", type=int, default=42)
    robust = sub.add_parser(
        "validate-robust-models",
        help="Validate separately retrained mass-decorrelated OOF models")
    robust.add_argument("--prediction", action="append", required=True,
                        help="Repeat as MODEL=path/to/predictions.csv")
    robust.add_argument("--output", required=True,
                        help="New directory; existing directories are refused")
    robust.add_argument("--efficiencies",
                        default="0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    robust.add_argument("--mass-window", default="118,130")
    robust.add_argument("--sidebands", default="105,118,130,140")
    robust.add_argument("--background-systematic", type=float, default=.30)
    robust.add_argument("--minimum-background-neff", type=float, default=10.)
    robust.add_argument("--maximum-fold-cv", type=float, default=.25)
    primary = sub.add_parser(
        "primary-mass-shapes",
        help="Score-pass binned profile likelihood in the Higgs mass window")
    primary.add_argument("--prediction", action="append", required=True,
                         help="Repeat as MODEL=path/to/predictions.csv")
    primary.add_argument("--threshold", action="append", default=[],
                         help="Repeat as MODEL=value; mutually exclusive with target efficiency")
    primary.add_argument("--target-signal-efficiency", type=float,
                         help="Derive model-specific cuts from OOF signal MC in the signal window")
    primary.add_argument("--output", required=True,
                         help="New directory; existing directories are refused")
    primary.add_argument("--fit-range", default="115,130")
    primary.add_argument("--bin-width", type=float, default=5.)
    primary.add_argument("--background-systematic", type=float, default=.30)
    primary.add_argument("--stat-repeats", type=int, default=1000)
    primary.add_argument("--seed", type=int, default=42)
    information = sub.add_parser(
        "compare-information",
        help="Compare mass-only, ML-cut, categorized and 2D mass-score fits")
    information.add_argument("--prediction", action="append", required=True,
                             help="Repeat as MODEL=path/to/predictions.csv")
    information.add_argument("--output", required=True,
                             help="New directory; existing directories are refused")
    information.add_argument("--fit-range", default="105,140")
    information.add_argument("--mass-bin-width", type=float, default=5.)
    information.add_argument("--target-signal-efficiency", type=float, default=.80)
    information.add_argument("--stat-repeats", type=int, default=1000)
    information.add_argument("--seed", type=int, default=42)
    sub.add_parser("ui", help="Open the local browser interface")
    args = parser.parse_args(argv)
    if args.command == "doctor": return doctor(args.root)
    if args.command == "ui":
        import subprocess
        from pathlib import Path
        app = Path(__file__).with_name("app.py")
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])
    try:
        if args.command == "compare-information":
            def information_tuple(value, count, option):
                try:
                    parsed = tuple(float(item.strip()) for item in value.split(","))
                except ValueError as exc:
                    raise ValueError(
                        f"{option} must contain {count} comma-separated numbers") from exc
                if len(parsed) != count or not np.isfinite(parsed).all():
                    raise ValueError(
                        f"{option} must contain {count} finite comma-separated numbers")
                return parsed
            paths = {}
            for value in args.prediction:
                if "=" not in value:
                    raise ValueError("Expected MODEL=path/to/predictions.csv")
                name, path = value.split("=", 1)
                if not name or name in paths:
                    raise ValueError("Model names must be nonempty and unique")
                paths[name] = path
            if not 0 < args.target_signal_efficiency < 1:
                raise ValueError("Target signal efficiency must be in (0, 1)")
            if args.mass_bin_width <= 0 or args.stat_repeats < 0:
                raise ValueError("Mass-bin width must be positive and repeats nonnegative")
            from .information_gain import compare_information_strategies
            result = compare_information_strategies(
                paths, args.output,
                fit_range=information_tuple(args.fit_range, 2, "--fit-range"),
                mass_bin_width=args.mass_bin_width,
                target_efficiency=args.target_signal_efficiency,
                statistical_repeats=args.stat_repeats, seed=args.seed)
            print(f"Saved {result}")
            return 0
        if args.command == "primary-mass-shapes":
            def primary_assignments(values, cast):
                parsed = {}
                for value in values:
                    if "=" not in value:
                        raise ValueError("Expected MODEL=value")
                    name, item = value.split("=", 1)
                    if not name or name in parsed:
                        raise ValueError("Model names must be nonempty and unique")
                    parsed[name] = cast(item)
                return parsed
            def primary_tuple(value, count, option):
                try:
                    parsed = tuple(float(item.strip()) for item in value.split(","))
                except ValueError as exc:
                    raise ValueError(
                        f"{option} must contain {count} comma-separated numbers") from exc
                if len(parsed) != count or not np.isfinite(parsed).all():
                    raise ValueError(
                        f"{option} must contain {count} finite comma-separated numbers")
                return parsed
            paths = primary_assignments(args.prediction, str)
            thresholds = primary_assignments(args.threshold, float)
            if "Logistic Regression" in paths:
                raise ValueError("Logistic Regression is excluded from primary inference")
            if args.target_signal_efficiency is not None and thresholds:
                raise ValueError(
                    "Use either --threshold or --target-signal-efficiency, not both")
            if args.target_signal_efficiency is not None:
                if not 0 < args.target_signal_efficiency < 1:
                    raise ValueError("Target signal efficiency must be in (0, 1)")
                from .primary_inference import fixed_signal_efficiency_thresholds
                thresholds = fixed_signal_efficiency_thresholds(
                    paths, args.target_signal_efficiency,
                    primary_tuple(args.fit_range, 2, "--fit-range"))
            elif set(paths) != set(thresholds):
                raise ValueError(
                    "Every primary model needs a threshold, or set target signal efficiency")
            if any(not 0 < value < 1 for value in thresholds.values()):
                raise ValueError("Thresholds must be in (0, 1)")
            if args.stat_repeats < 20:
                raise ValueError("--stat-repeats must be at least 20")
            if args.background_systematic < 0:
                raise ValueError("--background-systematic must be nonnegative")
            from .primary_inference import run_primary_inference
            result = run_primary_inference(
                paths, thresholds, args.output,
                fit_range=primary_tuple(args.fit_range, 2, "--fit-range"),
                bin_width=args.bin_width,
                background_systematic=args.background_systematic,
                statistical_repeats=args.stat_repeats,
                seed=args.seed)
            print(f"Saved {result}")
            return 0
        if args.command == "validate-robust-models":
            def robust_tuple(value, count, option):
                try:
                    parsed = tuple(float(item.strip()) for item in value.split(","))
                except ValueError as exc:
                    raise ValueError(
                        f"{option} must contain {count} comma-separated numbers") from exc
                if len(parsed) != count or not np.isfinite(parsed).all():
                    raise ValueError(
                        f"{option} must contain {count} finite comma-separated numbers")
                return parsed
            paths = {}
            for value in args.prediction:
                if "=" not in value:
                    raise ValueError("Expected MODEL=path/to/predictions.csv")
                name, path = value.split("=", 1)
                if not name or name in paths:
                    raise ValueError("Model names must be nonempty and unique")
                paths[name] = path
            efficiencies = robust_tuple(args.efficiencies,
                                        len(args.efficiencies.split(",")),
                                        "--efficiencies")
            if not efficiencies or any(not 0 < item < 1 for item in efficiencies):
                raise ValueError("--efficiencies values must be in (0, 1)")
            mass_window = robust_tuple(args.mass_window, 2, "--mass-window")
            sidebands = robust_tuple(args.sidebands, 4, "--sidebands")
            if args.background_systematic < 0 or args.minimum_background_neff <= 0:
                raise ValueError("Systematic must be nonnegative and minimum Neff positive")
            if args.maximum_fold_cv <= 0:
                raise ValueError("--maximum-fold-cv must be positive")
            from .robust import validate_robust_models
            result = validate_robust_models(
                paths, args.output, efficiencies=efficiencies,
                mass_window=mass_window, sidebands=sidebands,
                systematic=args.background_systematic,
                minimum_background_neff=args.minimum_background_neff,
                maximum_fold_cv=args.maximum_fold_cv)
            print(f"Saved {result}")
            return 0
        if args.command == "analyze-features":
            from .feature_analysis import analyze_features
            result = analyze_features(args.prepared, args.output, args.seed, args.correlation_threshold)
            print(f"Saved {result}")
            return 0
        if args.command == "scan-thresholds":
            if args.step <= 0 or args.stop <= args.start:
                raise ValueError("Threshold range requires stop > start and step > 0")
            config = load_config(args.config)
            from .thresholds import analyze_thresholds
            values = np.arange(args.start, args.stop + args.step/2, args.step)
            result = analyze_thresholds(
                args.predictions, args.output, values,
                config.statistics.mass_window_gev,
                config.statistics.background_fractional_systematic,
                config.training.threshold)
            print(f"Saved {result}")
            return 0
        if args.command == "study-stability":
            config = load_config(args.config)
            try:
                seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
                folds = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
            except ValueError as exc:
                raise ValueError("--seeds and --folds must be comma-separated integers") from exc
            from .stability import analyze_stability
            result = analyze_stability(args.prepared, args.output, config, seeds, folds)
            print(f"Saved {result}")
            return 0
        if args.command == "plot-features":
            from .diagnostics import create_diagnostics
            result = create_diagnostics(args.prepared, args.output)
            print(f"Saved {result}")
            return 0
        if args.command == "plot-preselection":
            from .preselection import plot_preselection
            result = plot_preselection(args.prepared, args.output)
            print(f"Saved {result}")
            return 0
        if args.command == "infer-observed":
            config = load_config(args.config)
            try:
                sidebands = tuple(float(value.strip()) for value in args.sidebands.split(","))
            except ValueError as exc:
                raise ValueError("--sidebands must contain four comma-separated numbers") from exc
            if len(sidebands) != 4:
                raise ValueError("--sidebands must contain four comma-separated numbers")
            from .inference import analyze_observed
            result = analyze_observed(
                args.predictions, args.output,
                config.training.threshold if args.threshold is None else args.threshold,
                config.statistics.mass_window_gev, sidebands,
                config.statistics.background_fractional_systematic,
                config.data.fraction)
            print(f"Saved {result}")
            return 0
        if args.command in {"compare-models", "compare-mass-shapes",
                            "validate-mass-shapes"}:
            config = load_config(args.config)
            def assignments(values, cast):
                result = {}
                for value in values:
                    if "=" not in value: raise ValueError("Expected MODEL=value")
                    name, item = value.split("=", 1)
                    if not name or name in result: raise ValueError("Model names must be nonempty and unique")
                    result[name] = cast(item)
                return result
            paths = assignments(args.prediction, str)
            chosen = {name: config.training.threshold for name in paths}
            for name, value in assignments(args.threshold, float).items():
                if name not in paths: raise ValueError(f"Threshold supplied for unknown model: {name}")
                chosen[name] = value
            if args.command == "compare-mass-shapes":
                try:
                    fit_range = tuple(float(value.strip()) for value in
                                      args.fit_range.split(","))
                except ValueError as exc:
                    raise ValueError("--fit-range must contain two comma-separated numbers") from exc
                if len(fit_range) != 2:
                    raise ValueError("--fit-range must contain two comma-separated numbers")
                if args.toys < 0:
                    raise ValueError("--toys must be nonnegative")
                if args.stat_repeats < 20:
                    raise ValueError("--stat-repeats must be at least 20")
                from .shape_likelihood import compare_mass_shape_models
                result = compare_mass_shape_models(
                    paths, chosen, args.output, fit_range=fit_range,
                    bin_width=args.bin_width, toys=args.toys, seed=args.seed,
                    stat_repeats=args.stat_repeats)
                print(f"Saved {result}")
                return 0
            if args.command == "validate-mass-shapes":
                def numeric_tuple(value, count, option):
                    try:
                        result = tuple(float(item.strip()) for item in value.split(","))
                    except ValueError as exc:
                        raise ValueError(f"{option} must contain {count} comma-separated numbers") from exc
                    if len(result) != count:
                        raise ValueError(f"{option} must contain {count} comma-separated numbers")
                    return result
                fit_range = numeric_tuple(args.fit_range, 2, "--fit-range")
                signal_window = numeric_tuple(args.signal_window, 2, "--signal-window")
                sidebands = numeric_tuple(args.sidebands, 4, "--sidebands")
                if args.gof_toys < 20:
                    raise ValueError("--gof-toys must be at least 20")
                from .mass_shape_validation import validate_mass_shapes
                result = validate_mass_shapes(
                    paths, chosen, args.output, fit_range=fit_range,
                    signal_window=signal_window, sidebands=sidebands,
                    bin_width=args.bin_width, gof_toys=args.gof_toys,
                    seed=args.seed)
                print(f"Saved {result}")
                return 0
            if args.bootstrap_repeats < 10: raise ValueError("--bootstrap-repeats must be >= 10")
            try:
                efficiencies = [float(value.strip()) for value in
                                args.efficiencies.split(",") if value.strip()]
            except ValueError as exc:
                raise ValueError("--efficiencies must be comma-separated numbers") from exc
            from .comparison import compare_models
            result = compare_models(paths, chosen, args.output, config,
                                    bootstrap_repeats=args.bootstrap_repeats,
                                    efficiencies=efficiencies)
            print(f"Saved {result}")
            return 0
        config = load_config(args.config)
        if args.command == "check-config":
            print("Configuration valid")
            return 0
        if args.command == "audit-normalization":
            from .normalization import audit_normalization
            result = audit_normalization(config, args.output)
            print(f"Saved {result}")
            return 0
        if args.command == "compare-weight-conventions":
            def pair(value, option):
                try:
                    parsed = tuple(float(item.strip()) for item in value.split(","))
                except ValueError as exc:
                    raise ValueError(f"{option} must contain two numbers") from exc
                if len(parsed) != 2 or not np.isfinite(parsed).all():
                    raise ValueError(f"{option} must contain two finite numbers")
                return parsed
            from .normalization import compare_weight_conventions
            result = compare_weight_conventions(
                config, args.output,
                fit_range=pair(args.fit_range, "--fit-range"),
                signal_window=pair(args.signal_window, "--signal-window"))
            print(f"Saved {result}")
            return 0
        from .pipeline import prepare, run
        result = prepare(config,args.output) if args.command == "prepare" else run(config,args.prepared,args.output)
        print(f"Saved {result}")
        return 0
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f"Error: {exc}",file=sys.stderr)
        if isinstance(exc,ImportError): print('For ROOT preparation, install: python -m pip install ".[root]"',file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
