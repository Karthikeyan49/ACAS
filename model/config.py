"""
model/config.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    Single source of truth for every constant in the LightGBM pipeline.
    Edit this file to change paths, targets, or hyperparameter search bounds.
    Nothing else needs to change.

CALLED FROM
    data/data_pipeline.py    DATA_PATH, DROP_COLS, CATEGORICAL_COLS, splits
    model/lgbm_model.py      LGBM_REG_PARAMS, LGBM_CLF_PARAMS, OPTUNA_*
    model/train.py           MODEL_DIR, PLOT_DIR, LOG_DIR, all above
    model/inference.py       MODEL_DIR
    model/evaluate.py        PLOT_DIR, SHAP_SAMPLE_SIZE, SHAP_TOP_N_FEATURES
    pipeline/model_bridge.py MODEL_DIR

CALLS INTO
    os only.

WHAT IT PROVIDES (all module-level constants)
    PATHS
        BASE_DIR     project root (resolved from this file's location)
        DATA_PATH    path to CDM training CSV
        MODEL_DIR    trained_models/lgbm/  (regressor.pkl lives here)
        PLOT_DIR     outputs/plots/
        LOG_DIR      outputs/logs/

    TARGETS
        REGRESSION_TARGET      "risk"       log10 collision probability
        CLASSIFICATION_TARGET  "high_risk"  binary — top 20% by risk
        HIGH_RISK_PERCENTILE   0.80

    FEATURE CONTROL
        DROP_COLS         removed before modelling (leakage risk)
        CATEGORICAL_COLS  ["c_object_type", "mission_id"]

    SPLITS
        TEST_SIZE 0.15 | VAL_SIZE 0.15 | RANDOM_STATE 42

    LIGHTGBM PARAMS
        LGBM_REG_PARAMS   regression booster config dict
        LGBM_CLF_PARAMS   classification booster config (is_unbalance=True)

    OPTUNA
        OPTUNA_N_TRIALS 50 | OPTUNA_TIMEOUT_SEC 1800 | OPTUNA_CV_FOLDS 5
        OPTUNA_SEARCH_SPACE  bounds for num_leaves, learning_rate, etc.

    TRAINING
        EARLY_STOPPING_ROUNDS 100

    EXPLAINABILITY
        SHAP_SAMPLE_SIZE 2000 | SHAP_TOP_N_FEATURES 20

PATH CHANGE FROM ORIGINAL (satellite_lgbm/config.py)
    MODEL_DIR  outputs/models  →  trained_models/lgbm/
══════════════════════════════════════════════════════════════════════════════
"""
import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/satellite_conjunctions.csv"   # ← change to your CSV path
OUTPUT_DIR   = "outputs"
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "trained_models", "lgbm")
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
