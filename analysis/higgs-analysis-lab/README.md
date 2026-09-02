# Higgs Analysis Lab — first modular version

A student-oriented Python package derived from `higgs_analysis(1).py`.
**This is a first migration, not a validated reproduction of the paper or a browser app.**
The complete original script remains unchanged in `reference/`.

## 1. Install once

Use Python **3.11 or 3.12**. Open Terminal in the extracted `higgs-analysis-lab` folder.
Do not run these commands from inside the Python prompt.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
higgs-lab doctor
```

On Windows use `py -3.12 -m venv .venv`, then `.venv\Scripts\Activate.ps1`
in PowerShell instead of the first two commands above.

`pyproject.toml` describes installation; you do not execute it directly.
The default install supports prepared CSV data, logistic regression and random forest.
No installer runs during analysis and no global Python environment is modified.

For raw ROOT data preparation, additionally install:

```bash
python -m pip install ".[root]"
higgs-lab doctor --root
```

This optional extra includes uproot, awkward, vector and atlasopenmagic.
The package uses HTTPS and does not directly require xrootd or TensorFlow.
Optional packages may have their own dependencies. Remote access and the selected
ATLAS release must be checked on the target machine; see `docs/TESTING.md`.

For the boosted-tree models, install:

```bash
python -m pip install ".[boosters]"
```

For the TensorFlow MLP, install `".[neural]"`. To install every model backend,
use `".[all-models]"`.

For the local browser interface, install and launch:

```bash
python -m pip install ".[ui]"
higgs-lab ui
```

## 2. Check the analysis settings

```bash
higgs-lab check-config --config configs/default.toml
```

Edit **`configs/default.toml`** to change features, model, seed, cuts or threshold.
This is different from `pyproject.toml`, which controls package installation.

Default values follow the source for luminosity (36.6 fb^-1), first three lepton
pT thresholds (20, 15, 10 GeV), mass window [110, 135) GeV, threshold 0.65 and
background systematic assumption 30%. The starter uses a fixed seven-feature
subset and logistic regression; it does **not** rerun the original feature
ranking, model tuning or full model-comparison study.

## 3. Prepare data once

```bash
higgs-lab prepare --config configs/default.toml --output prepared/full
```

This accesses the configured ATLAS data and MC samples, applies selections,
reconstructs the final Z variables, and saves:

- `features.csv`: all 31 available features, mass, original weights and explicit sample roles.
- `cutflow.csv`: unweighted event counts at each selection stage, by sample.
- `manifest.json`: data settings, source file list, checksum and environment.

The prepared CSV is a deliberate simple interchange format. No data is bundled
with this project. ROOT files are read in chunks, though the ML stage loads the
prepared feature table into memory. Full processing can take substantial time
and requires network access. There is no reliable runtime estimate yet.

For a quick plumbing check, reduce `fraction`, e.g. to `0.01`. It selects the
first fraction of each file, not a random representative sample; some classes
may then be too small to train. Partial-sample weights remain unrescaled and
significance output is disabled. Do not compare these counts to the paper.

Preparation also writes `preselection_mass.csv` before Z-pair reconstruction.
To reproduce the reference pre-ML mass figure (and a separate Data-overlay
extension), run:

```bash
higgs-lab plot-preselection --prepared prepared/full --output results/preselection
```

## 4. Train and save results

Before training, you can reproduce the three feature-ranking diagnostics on a
verified prepared dataset:

```bash
higgs-lab analyze-features --prepared prepared/full --output results/feature-study
```

This writes `feature_ranking.csv`, `correlation_pruning.csv`,
`feature_analysis.json`, and one plot for each ranking method. Ranking and
correlation pruning use only a stratified MC training split. Review the result
before copying a selected feature list into the configuration; the command does
not silently modify an experiment configuration.

Additional reproducible studies operate on saved prepared data or predictions:

```bash
higgs-lab plot-features --prepared prepared/full --output results/feature-plots
higgs-lab scan-thresholds --predictions results/baseline/predictions.csv --output results/thresholds
higgs-lab study-stability --prepared prepared/full --output results/stability
higgs-lab infer-observed --predictions results/baseline/predictions.csv --output results/observed
```

Compare several completed model runs with repeated `MODEL=PATH` arguments:

```bash
higgs-lab compare-models \
  --prediction logistic=results/logistic/predictions.csv \
  --prediction forest=results/forest/predictions.csv \
  --output results/model-comparison
```

The comparison output includes `roc_all_models.png`, reproducing the reference
weighted out-of-fold ROC overlay with bootstrap 95% AUC intervals in the legend.

```bash
higgs-lab run --config configs/default.toml --prepared prepared/full --output results/baseline
```

Outputs: `predictions.csv`, `summary.json`, `config.json`, `provenance.json`,
`mass_before.png`, `mass_after.png`, `roc.png`.

MC predictions are out-of-fold. Observed data are scored by the average of the
fold models and never used for fitting, feature scaling or AUC evaluation.
The current summary reports **expected MC count proxies**, not observed discovery
significances. See the scientific limitations below.

For a second experiment, copy the config, set `model = "random_forest"`, and use
a new result directory. You can reuse the prepared directory if data and selection
settings are unchanged. Changing cuts or luminosity requires preparing a new cache.
Existing output directories are refused to prevent overwriting earlier runs.

If a run fails, its output directory may be incomplete. Retain it for diagnosis
and choose a new directory on retry. This version does not resume failed runs.

## Files and responsibilities

| File | Responsibility |
|---|---|
| `pyproject.toml` | Installation metadata, dependencies and terminal command |
| `configs/default.toml` | Analysis settings |
| `src/higgs_lab/config.py` | Configuration loading and validation |
| `src/higgs_lab/samples.py` | Original sample IDs and explicit physics roles |
| `src/higgs_lab/datasets.py` | Dataset resolution and chunked ROOT reading |
| `src/higgs_lab/selection.py` | Cuts and event weights |
| `src/higgs_lab/reconstruction.py` | Original final Z reconstruction and angles |
| `src/higgs_lab/features.py` | Feature table and data/MC separation checks |
| `src/higgs_lab/feature_analysis.py` | Training-only ranking and correlation pruning |
| `src/higgs_lab/training.py` | Common OOF cross-validation for seven model adapters |
| `src/higgs_lab/thresholds.py` | MC-only threshold scans and diagnostics |
| `src/higgs_lab/stability.py` | Seed and fold-count studies |
| `src/higgs_lab/inference.py` | Guarded observed-count and sideband summaries |
| `src/higgs_lab/comparison.py` | Multi-model tables, bootstrap intervals and forest plots |
| `src/higgs_lab/diagnostics.py` | All-feature distribution and correlation plots |
| `src/higgs_lab/app.py` | Local browser working surface |
| `src/higgs_lab/statistics.py` | Weighted yields and legacy expected-count proxy |
| `src/higgs_lab/plots.py` | Reusable mass and ROC plots |
| `src/higgs_lab/provenance.py` | Checksums, environment records and output safety |
| `src/higgs_lab/pipeline.py` | Prepare/run orchestration |
| `src/higgs_lab/cli.py` | Terminal commands and environment checks |
| `tests/test_workflow.py` | Synthetic-data regression and integration tests |
| `reference/higgs_analysis_original.py` | Unchanged original research script; not the entry point |
| `docs/MIGRATION.md` | Preserved behavior, intentional changes and pending work |
| `docs/TESTING.md` | Verification performed and its limits |

## Scientific limitations

- `legacy_absolute` preserves the source's `abs()` multiplication of MC weight
  factors. It is explicitly labelled, not endorsed as the correct signed-MC
  treatment. `signed` preserves signs; training refuses negative weights until
  a reviewed ML weighting policy is supplied.
- Reconstruction is the final function in the original source, preserved
  verbatim. Angular conventions, degenerate vectors and units still need a
  physics review and comparison against trusted events.
- The source's approximate expected-count formula is preserved. P-values are
  intentionally not exported, because that approximation is not a calibrated
  significance procedure. Observed inference, sidebands and a likelihood fit
  are pending. Do not quote this tool's output as a discovery significance.
- Hyperparameters, features and threshold are fixed by configuration. Selecting
  them after inspecting OOF results creates selection bias; use independent
  evaluation or nested validation for such optimization.
- Fold-averaged data scores and single-fold MC scores do not have identical
  construction. Validate their responses before using them for observed inference.
- Original sample metadata and normalization assumptions have not yet been
  validated against the remote release. Exact paper numbers are not promised.

## Tests

```bash
python -m unittest discover -s tests -v
```

ROOT tests skip when optional dependencies are unavailable. Synthetic tests prove
software behavior, not physical validity or reproduction of the paper.

## Next migration steps

1. Validate raw processing and reference yields on real ATLAS input.
2. Review weights, units, selection and angular conventions with the supervisor.
3. Validate all migrated model adapters and diagnostics against the original
   standalone experiments on fixed real input.
4. Replace approximate counting/sideband summaries with a reviewed likelihood
   model before making scientific claims.
5. Add safe model persistence and a resumable job layer for long-running UI work.

Dependency ranges in `pyproject.toml` are not a lock file. The tested core-version
constraints are supplied separately; a complete platform-specific lock including
ROOT dependencies should be generated only after clean-install validation.
