# Migration notes

## Scope of version 0.1.0

This is a working modular starter, not a mechanical split of all 7,000+ lines.
Repeated standalone experiments remain in the unchanged reference script.
No GUI, notebook, prepared ATLAS data or trained model binaries are included.

## Preserved

- Exact dataset IDs and release/skim strings from the source.
- Luminosity 36.6 fb^-1, trigger and matching logic, ID/isolation,
  first three strict pT cuts, flavor-sum filter and zero total charge.
- The source's final `reconstruct_z1_z2_fast`, `calc_mass`, `opening_dphi`
  and `delta_r` function bodies are copied verbatim. Tests compare their ASTs.
- All 31 named reconstructed ML features; explicit [110,135) GeV mass window.
- Separation power, single-feature AUC, random-forest importance and correlation
  pruning are available as a training-only feature-analysis stage.
- Seven model adapters cover XGBoost, LightGBM, random forest, MLP, logistic
  regression, Gaussian naive Bayes and QDA in a common OOF framework.
- MC-only threshold scans, seed/fold stability studies, all-feature plots,
  bootstrap model comparisons and guarded observed/sideband summaries are modular.
- A local browser interface calls the same package functions as the CLI.
- Default 30% background systematic and 0.65 threshold.
- Legacy absolute weighting and expected MC count formula, explicitly labelled.

## Intentional changes

- No import-time installation, dataset access, plotting or training.
- Data/MC roles use explicit metadata. Data requests do not include MC-only branches.
- Exactly-four-lepton array lengths are checked before indexing. Reference mode
  preserves stored lepton order before positional pT cuts, matching the source.
  Optional `sort_leptons_by_pt = true` sorts all per-lepton fields together for
  studies that prefer an explicit ordering contract.
- No fourth-lepton threshold is added by default. The uploaded code, unlike some
  descriptions of the analysis, only applies the first three pT thresholds.
- Fractional entry limits are converted to integers for ROOT iteration.
- File reading is chunked. Selected feature rows are written incrementally.
- The final reconstruction implementation is unique, avoiding function redefinition.
- A fixed seven-feature preset replaces learned feature ranking in this starter.
- The five migrated model adapters use explicit baseline hyperparameters rather
  than reproducing every standalone experiment in the source. Class balancing,
  scaling, validation splits and early stopping are handled per model. Fold count
  and seeds are configurable.
- Tuning, early stopping, calibrated threshold scans and nested model selection
  are performed inside training data only; outer-fold events remain held out.
- Data is deterministically fold-assigned and evaluated with the corresponding
  fold model and fold-local calibration.
- All significance outputs use explicit mass boundaries and a one-bin Poisson
  profile likelihood. The Gaussian background constraint combines weighted-count
  statistical uncertainty and the configured 30% systematic in quadrature.
- Expected MC Z uses the Asimov data set. Observed Z uses the signal-window data
  count. Displayed Z uncertainties propagate finite-count statistics numerically.
- Partial-input significance is suppressed instead of reporting a misleading
  full-luminosity result. Weights are not silently rescaled.
- Predictions, raw preparation settings, configuration and version metadata are
  saved. Models are returned in memory but not serialized to unsafe pickle files.

## Not yet migrated

Exact reproduction of every standalone cell and its historical hyperparameters;
the original specialized composite physics figures; a full mass-shape likelihood;
safe fitted-model persistence; resumable background jobs; and real-ATLAS numerical
validation. Migrated feature selection and threshold choices use leakage-resistant
OOF/training boundaries and therefore are not intended to reproduce numbers from
source cells that inspect the complete dataset.
All original code for these remains available in `reference/`.

## Reference-validation procedure to perform next

Use an isolated environment and fixed source files. Compare per-sample event counts,
weights, sum of squared weights, mass/feature arrays and selected yields. Separate
configuration differences from refactoring differences. Review the negative-weight
policy before approving a scientific baseline. Only then migrate tuning experiments
and quote any performance/significance comparison.
