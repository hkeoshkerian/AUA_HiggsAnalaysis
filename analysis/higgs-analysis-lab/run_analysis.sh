#!/usr/bin/env bash
set -euo pipefail

# Run this script from anywhere. Change this path if the project is moved.
PROJECT_DIR="/Users/hourykeoshkerian/AUA_HiggsAnalaysis/analysis/higgs-analysis-lab"

cd "$PROJECT_DIR"

# ----------------------------------------------------------------------
# 1. Create or activate the virtual environment
# ----------------------------------------------------------------------

if [[ ! -d ".venv" ]]; then
    python3.12 -m venv .venv
fi

source .venv/bin/activate
rehash

# ----------------------------------------------------------------------
# 2. Install the project and all required dependencies
# ----------------------------------------------------------------------

python -m pip install --upgrade pip
python -m pip install -e ".[root,all-models]"

higgs-lab doctor --root
higgs-lab check-config --config configs/default.toml

# ----------------------------------------------------------------------
# 3. Prepare the full ATLAS Open Data sample
#
# This step is skipped when the prepared cache already exists.
# Remove or rename prepared/reference-full if you intentionally need to
# prepare the data again after changing data or selection settings.
# ----------------------------------------------------------------------

if [[ ! -f "prepared/reference-full/manifest.json" ]]; then
    higgs-lab prepare \
        --config configs/default.toml \
        --output prepared/reference-full
else
    echo "Using existing prepared data: prepared/reference-full"
fi

# ----------------------------------------------------------------------
# 4. Produce the preselection Data-versus-MC m4l plots
# ----------------------------------------------------------------------
if [[ ! -d "results" ]]; then
    mkdir results
else
    echo "Skipping existing results folder creation"
fi

if [[ ! -d "results/reference-preselection" ]]; then
    higgs-lab plot-preselection \
       --prepared prepared/reference-full \
        --output results/reference-preselection
else
    echo "Skipping existing output: results/reference-preselection"
fi

# ----------------------------------------------------------------------
# 5. Produce MC-only kinematic plots and the signal correlation matrix
# ----------------------------------------------------------------------

if [[ ! -d "results/reference-kinematics" ]]; then
    higgs-lab plot-features \
        --prepared prepared/reference-full \
        --output results/reference-kinematics
else
    echo "Skipping existing output: results/reference-kinematics"
fi


# ----------------------------------------------------------------------
# 6. Calculate feature separation power and feature rankings
# ----------------------------------------------------------------------

if [[ ! -d "results/reference-feature-ranking" ]]; then
    higgs-lab analyze-features \
        --prepared prepared/reference-full \
        --output results/reference-feature-ranking \
        --seed 42 \
        --correlation-threshold 0.70
else
    echo "Skipping existing output: results/reference-feature-ranking"
fi

# ----------------------------------------------------------------------
# 7. Train the five retained classifiers
#
# Every model uses the same frozen process-stratified folds. MC is scored OOF;
# data is fold-assigned; both raw and fold-calibrated scores are saved.
# ----------------------------------------------------------------------

run_model() {
    local config_path="$1"
    local output_path="$2"

    echo "Running model: $config_path"
    higgs-lab run \
        --config "$config_path" \
        --prepared prepared/reference-full \
        --output "$output_path"
}

run_model configs/xgboost.toml            results/xgboost
run_model configs/lightgbm.toml            results/lightgbm
run_model configs/random_forest.toml       results/random-forest
run_model configs/mlp.toml                 results/mlp
run_model configs/logistic_regression.toml results/logistic-regression

# ----------------------------------------------------------------------
# 8. Compare all models and produce the combined ROC plot
# ----------------------------------------------------------------------

higgs-lab compare-models \
  --config configs/default.toml \
  --prediction "XGBoost=results/xgboost/predictions.csv" \
  --prediction "LightGBM=results/lightgbm/predictions.csv" \
  --prediction "Random Forest=results/random-forest/predictions.csv" \
  --prediction "MLP=results/mlp/predictions.csv" \
  --prediction "Logistic Regression=results/logistic-regression/predictions.csv" \
  --output results/all-model-comparison \
  --bootstrap-repeats 1000

echo
echo "Analysis completed successfully."
echo "Selection comparison: results/all-model-comparison/selection_strategy_comparison.md"
