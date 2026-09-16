"""Small student-facing command line entry point."""
import argparse
import importlib
import importlib.metadata
import sys
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
    sub.add_parser("ui", help="Open the local browser interface")
    args = parser.parse_args(argv)
    if args.command == "doctor": return doctor(args.root)
    if args.command == "ui":
        import subprocess
        from pathlib import Path
        app = Path(__file__).with_name("app.py")
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])
    try:
        if args.command == "analyze-features":
            from .feature_analysis import analyze_features
            result = analyze_features(args.prepared, args.output, args.seed, args.correlation_threshold)
            print(f"Saved {result}")
            return 0
        if args.command == "scan-thresholds":
            import numpy as np
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
        if args.command == "compare-models":
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
