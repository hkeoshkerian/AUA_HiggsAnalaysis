# Higgs Analysis Lab

This repository's primary workflow trains four classifiers and performs a
binned profile-likelihood analysis of the four-lepton invariant mass in
`115 <= m4l < 130 GeV`.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[root,boosters,neural]"
higgs-lab doctor --root
```

## Authoritative preparation

The primary configuration is `configs/atlas_2017_legacy_absolute.toml`. It uses:

- the inclusive ATLAS Open Data `4lep` skim;
- 36.1 fb^-1;
- the ATLAS-2017-like four-lepton fiducial selection;
- the legacy absolute event-weight convention required for this reproduction;
- the full retained mass range for diagnostic plots and a separately reported
  `115–130 GeV` yield.

Prepare once:

```bash
higgs-lab prepare \
  --config configs/atlas_2017_legacy_absolute.toml \
  --output prepared/atlas-2017-36p1-legacy-absolute
```

The current prepared sample has, in `115 <= m4l < 130 GeV`, 63 data events,
38.069 weighted signal events, and 32.599 weighted background events (21.112
irreducible and 11.486 reducible/rare).

## Primary training and inference

Run the complete analysis:

```bash
./run_primary_analysis.sh
```

An alternative prepared-data or output location may be supplied:

```bash
./run_primary_analysis.sh PREPARED_DIRECTORY OUTPUT_DIRECTORY
```

The analysis is fixed as follows:

- models: XGBoost, LightGBM, Random Forest, and MLP;
- features: the same nine mass-decorrelated variables in every model;
- training population: signal and background MC within `115–130 GeV` only;
- validation: the same deterministic five-fold, class-and-process-stratified
  out-of-fold assignment for every model;
- calibration: fitted within each training fold, never on its held-out fold;
- operating point: a model-specific threshold giving the closest attainable
  value to 80% weighted OOF signal efficiency, derived using MC only (ties at
  the weighted quantile are included);
- inference category: score-pass only; observed data never enter training or
  threshold selection;
- likelihood observable: `m4l` in three 5 GeV bins over `[115,130)`;
- nuisance model: per-bin MC statistical uncertainties plus one correlated 30%
  background-normalization uncertainty;
- reported results: expected and observed local discovery significance,
  one-sided p-value, fitted signal strength, and 68%/95% profile intervals;
- quoted statistical error on significance: 1,000 statistical repetitions.

The main outputs are under `results/primary-analysis/profile-likelihood/`:

- `primary_score_pass_inference.csv`;
- `primary_operating_points.csv`;
- `primary_expected_significance_forest.png`;
- `primary_observed_significance_forest.png`;
- `primary_signal_strength_forest.png`;
- one mass-fit and likelihood-scan plot per model;
- `primary_inference_method.json`, the machine-readable method record.

The No-ML row is the inclusive mass-only reference. Logistic Regression,
sideband extrapolation, score-fail inference, transfer-factor models, global
raw-score cuts, and alternative fit windows are not part of this primary path.

## Verification

```bash
MPLCONFIGDIR=/tmp/higgs-mpl \
  .venv/bin/python -m unittest discover -s tests -v
```

The configuration and tests enforce the common skim, luminosity, selection,
features, mass window, fold design, and profile-likelihood settings.

## Scope

This is an educational Open Data reproduction, not the full ATLAS statistical
model. The 30% background uncertainty is an explicit analysis assumption, not
a substitute for the complete experimental and theoretical nuisance model.
