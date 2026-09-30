#!/usr/bin/env bash
set -euo pipefail

PREPARED="${1:-prepared/atlas-2017-36p1-legacy-absolute}"
ROOT="${2:-results/primary-analysis}"

run_model() {
  local name="$1"
  local config="$2"
  local output="$ROOT/$name"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing output: $output" >&2
    exit 1
  fi
  echo "Running $name"
  higgs-lab run --config "$config" --prepared "$PREPARED" --output "$output"
}

mkdir -p "$ROOT"
run_model xgboost configs/robust_xgboost.toml
run_model lightgbm configs/robust_lightgbm.toml
run_model random-forest configs/robust_random_forest.toml
run_model mlp configs/robust_mlp.toml

higgs-lab primary-mass-shapes \
  --prediction "XGBoost=$ROOT/xgboost/predictions.csv" \
  --prediction "LightGBM=$ROOT/lightgbm/predictions.csv" \
  --prediction "Random Forest=$ROOT/random-forest/predictions.csv" \
  --prediction "MLP=$ROOT/mlp/predictions.csv" \
  --target-signal-efficiency 0.80 \
  --fit-range 115,130 \
  --bin-width 5 \
  --background-systematic 0.30 \
  --stat-repeats 1000 \
  --output "$ROOT/profile-likelihood"

echo "Primary analysis saved under $ROOT"
