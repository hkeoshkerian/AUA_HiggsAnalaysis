# Verification — 28 August 2026

## Passed

- Package built into a wheel and installed into an isolated virtual environment
  using locally available dependencies (offline, no build isolation).
- Installed `higgs-lab doctor` and `check-config` commands passed.
- **23 unittest tests passed, zero skipped**, using synthetic fixtures.
- Tests cover configuration errors, window boundaries and sum-of-squared-weights
  uncertainty, equality of the source count formula, unchanged reconstruction
  function bodies, forbidden feature columns, observed-data separation,
  out-of-fold coverage/scaling, negative-weight refusal, both model adapters,
  cache checksums and settings, overwrite protection, partial-data suppression,
  local ROOT reading, reconstruction, and weight modes.
- Tests additionally cover all seven model adapters, feature ranking and pruning,
  threshold scans, stability studies, diagnostic plots, observed/sideband methods,
  and multi-model comparison calculations.
- The full prepare -> CSV -> train -> report/plots path passed on locally written
  synthetic ROOT samples. Dataset resolution was mocked to local files; ROOT I/O,
  cuts, reconstruction and model fitting ran normally.
- The original supplied script is byte-for-byte preserved in `reference/`.

## Environment used

Python 3.12.13, Linux. Selected installed versions:

| Package | Version |
|---|---|
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| Matplotlib | 3.10.8 |
| scikit-learn | 1.8.0 |
| SciPy | 1.17.0 |
| uproot | 5.7.6 |
| awkward | 2.13.0 |
| vector | 1.8.1 |

`constraints-tested.txt` records these selected versions. It is **not a complete
transitive dependency lock** and does not claim compatibility on every platform.
For a fresh local install, it can be used with:

```bash
python -m pip install -c constraints-tested.txt .
```

## Not verified

- `atlasopenmagic` was not installed; live dataset resolution and downloads were
  not exercised. No successful full-network installation is claimed.
- The `.[root]` extra as a whole was not clean-installed and validated.
- No real ATLAS input, paper-yield reproduction, model-performance reproduction,
  sideband result or observed-significance calculation was validated.
- No clean Python 3.11 installation, full transitive lock, live browser interaction,
  or full optional boosted/neural dependency matrix was tested.

Synthetic tests confirm software behavior only. The next scientific milestone
is a small real-data/MC comparison against the original script before publishing
or teaching from any quantitative physics results.
