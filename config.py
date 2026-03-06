"""
config.py — Central configuration for the satellite collision LightGBM pipeline.
Edit this file to change paths, targets, and hyperparameter search bounds.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/satellite_conjunctions.csv"   # ← change to your CSV path
OUTPUT_DIR   = "outputs"
MODEL_DIR    = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR     = os.path.join(OUTPUT_DIR, "plots")
LOG_DIR      = os.path.join(OUTPUT_DIR, "logs")

# ─────────────────────────────────────────────────────────────────────────────
# TARGET COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
REGRESSION_TARGET    = "risk"           # continuous log-scale collision probability
CLASSIFICATION_TARGET = "high_risk"     # derived binary alert label (see feature_engineering)
HIGH_RISK_PERCENTILE  = 0.80            # top 20% by risk = high-risk

# ─────────────────────────────────────────────────────────────────────────────
# COLUMNS TO DROP BEFORE MODELLING
# ─────────────────────────────────────────────────────────────────────────────
DROP_COLS = [
    "event_id",                  # identifier — no predictive signal
    "max_risk_estimate",         # target-adjacent leakage risk in regression
    "max_risk_scaling",          # derived from target
]

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORICAL COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["c_object_type", "mission_id"]

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VALIDATION / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
TEST_SIZE       = 0.15    # 15% held-out test
VAL_SIZE        = 0.15    # 15% validation (from remaining train)
RANDOM_STATE    = 42

# ─────────────────────────────────────────────────────────────────────────────
# LIGHTGBM — REGRESSION (risk score prediction)
# ─────────────────────────────────────────────────────────────────────────────
LGBM_REG_PARAMS = {
    "objective":        "regression",
    "metric":           ["rmse", "mae"],
    "boosting_type":    "gbdt",
    "n_estimators":     3000,
    "learning_rate":    0.03,
    "num_leaves":       127,
    "max_depth":        -1,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       0.2,
    "n_jobs":           -1,
    "random_state":     RANDOM_STATE,
    "verbose":          -1,
}

# ─────────────────────────────────────────────────────────────────────────────
# LIGHTGBM — CLASSIFICATION (high-risk alert)
# ─────────────────────────────────────────────────────────────────────────────
LGBM_CLF_PARAMS = {
    "objective":        "binary",
    "metric":           ["binary_logloss", "auc"],
    "boosting_type":    "gbdt",
    "n_estimators":     3000,
    "learning_rate":    0.03,
    "num_leaves":       127,
    "max_depth":        -1,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       0.2,
    "is_unbalance":     True,       # handles class imbalance automatically
    "n_jobs":           -1,
    "random_state":     RANDOM_STATE,
    "verbose":          -1,
}

# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
OPTUNA_N_TRIALS     = 50       # increase to 100+ for production
OPTUNA_TIMEOUT_SEC  = 1800     # 30 min max tuning time
OPTUNA_CV_FOLDS     = 5

# Optuna search bounds
OPTUNA_SEARCH_SPACE = {
    "num_leaves":         (31, 255),
    "min_child_samples":  (10, 100),
    "learning_rate":      (0.01, 0.1),
    "feature_fraction":   (0.6, 1.0),
    "bagging_fraction":   (0.6, 1.0),
    "bagging_freq":       (1, 10),
    "reg_alpha":          (0.0, 1.0),
    "reg_lambda":         (0.0, 1.0),
    "n_estimators":       (500, 5000),
}

# ─────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────
EARLY_STOPPING_ROUNDS = 100

# ─────────────────────────────────────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
SHAP_SAMPLE_SIZE    = 2000     # rows to use for SHAP (full set is slow)
SHAP_TOP_N_FEATURES = 20       # show top N features in plots
