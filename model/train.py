"""
model/train.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    Master training orchestrator. The single entry point for training,
    tuning, evaluating, and running inference on both LightGBM models.

CALLED FROM
    Terminal:  python model/train.py [--tune] [--cv]
               python model/train.py --infer --input new.csv
    pyproject.toml entry point: "acas-train"

CALLS INTO
    model/config.py         all constants
    data/data_pipeline.py   prepare_data()
    model/lgbm_model.py     SatelliteRiskRegressor, SatelliteRiskClassifier,
                             tune_hyperparameters
    model/evaluate.py       evaluate_regression, evaluate_classification,
                             plot_feature_importance, compute_shap,
                             print_final_summary
    model/inference.py      SatelliteCollisionPredictor (--infer mode)

PIPELINE STEPS (in order)
    1. data_pipeline.prepare_data()       load → impute → clip → engineer → split
    2. [--tune]  tune_hyperparameters()   Optuna 50 trials, 30 min timeout
    3. Train SatelliteRiskRegressor       → trained_models/lgbm/regressor.pkl
    4. Train SatelliteRiskClassifier      → trained_models/lgbm/classifier.pkl
    5. [--cv]  5-fold cross-validation    print R² per fold
    6. evaluate_regression()              + evaluate_classification()
    7. plot_feature_importance()          → outputs/plots/
    8. compute_shap()                     → outputs/plots/
    9. Save encoders.pkl

CLI FLAGS
    --tune    Optuna hyperparameter search before training
    --cv      5-fold cross-validation
    --infer   batch inference (requires --input)
    --input   path to input CSV
    --output  path for predictions CSV
    --data    path to training CSV (overrides config.DATA_PATH)

RENAMED FROM
    satellite_lgbm/main.py  →  model/train.py

IMPORT CHANGES
    import config              →  from model import config
    from data_pipeline import  →  from data.data_pipeline import
    from model import …        →  from model.lgbm_model import …
    from evaluate import …     →  from model.evaluate import …
══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

from model import config

os.makedirs(config.LOG_DIR,   exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.PLOT_DIR,  exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOG_DIR, "pipeline.log"), mode="a"),
    ],
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS (after logging is configured)
# ─────────────────────────────────────────────────────────────────────────────

from data.data_pipeline import prepare_data
from model.lgbm_model import (SatelliteRiskRegressor, SatelliteRiskClassifier,
                               tune_hyperparameters)
from model.evaluate import (evaluate_regression, evaluate_classification,
                             plot_feature_importance, compute_shap, print_final_summary)


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Satellite Collision LightGBM Pipeline")
    parser.add_argument("--tune",   action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--cv",     action="store_true", help="Run cross-validation")
    parser.add_argument("--infer",  action="store_true", help="Run batch inference")
    parser.add_argument("--input",  type=str, default=None, help="Input CSV for inference")
    parser.add_argument("--output", type=str, default="outputs/predictions.csv",
                        help="Output path for inference results")
    parser.add_argument("--data",   type=str, default=config.DATA_PATH,
                        help="Path to training data CSV")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args):
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("  SATELLITE COLLISION RISK — LightGBM PIPELINE")
    logger.info("=" * 60)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1 — Data preparation
    # ──────────────────────────────────────────────────────────────────────
    logger.info("\n[STEP 1/5] Data preparation...")
    data = prepare_data(path=args.data)

    encoders = data["encoders"]
    df       = data["df"]

    # Regression splits
    X_train_r, X_val_r, X_test_r, y_train_r, y_val_r, y_test_r, feat_r = data["regression"]
    # Classification splits
    X_train_c, X_val_c, X_test_c, y_train_c, y_val_c, y_test_c, feat_c = data["classification"]

    cat_cols = [c for c in config.CATEGORICAL_COLS if c in X_train_r.columns]

    logger.info(f"  Regression  features : {len(feat_r)}")
    logger.info(f"  Classification features: {len(feat_c)}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2 — Hyperparameter tuning (optional)
    # ──────────────────────────────────────────────────────────────────────
    reg_params = config.LGBM_REG_PARAMS.copy()
    clf_params = config.LGBM_CLF_PARAMS.copy()

    if args.tune:
        logger.info("\n[STEP 2/5] Hyperparameter tuning with Optuna...")

        logger.info("  Tuning regression model...")
        best_reg, reg_study = tune_hyperparameters(
            "regression", X_train_r, y_train_r, X_val_r, y_val_r,
            cat_cols=cat_cols,
        )
        reg_params.update(best_reg)

        logger.info("  Tuning classification model...")
        best_clf, clf_study = tune_hyperparameters(
            "classification", X_train_c, y_train_c, X_val_c, y_val_c,
            cat_cols=cat_cols,
        )
        clf_params.update(best_clf)

        # Save best params
        import json
        params_path = os.path.join(config.OUTPUT_DIR, "best_params.json")
        with open(params_path, "w") as f:
            json.dump({"regression": reg_params, "classification": clf_params}, f, indent=2)
        logger.info(f"  Best params saved → {params_path}")
    else:
        logger.info("\n[STEP 2/5] Using default hyperparameters (pass --tune to optimise)")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3 — Model training
    # ──────────────────────────────────────────────────────────────────────
    logger.info("\n[STEP 3/5] Training models...")

    # Regression
    t1 = time.time()
    regressor = SatelliteRiskRegressor(params=reg_params)
    regressor.train(
        X_train_r, y_train_r,
        X_val_r,   y_val_r,
        feature_names=feat_r,
        cat_cols=cat_cols,
    )
    logger.info(f"  Regressor trained in {time.time()-t1:.1f}s")

    # Classification
    t1 = time.time()
    classifier = SatelliteRiskClassifier(params=clf_params)
    classifier.train(
        X_train_c, y_train_c,
        X_val_c,   y_val_c,
        feature_names=feat_c,
        cat_cols=cat_cols,
    )
    logger.info(f"  Classifier trained in {time.time()-t1:.1f}s")

    # Save models
    regressor.save()
    classifier.save()

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4 — Evaluation
    # ──────────────────────────────────────────────────────────────────────
    logger.info("\n[STEP 4/5] Evaluating on held-out test set...")

    reg_metrics = evaluate_regression(regressor,  X_test_r, y_test_r)
    clf_metrics = evaluate_classification(classifier, X_test_c, y_test_c)

    plot_feature_importance(
        regressor,  feat_r,
        title="Feature Importance — Risk Score Regression",
        filename="feature_importance_regression.png",
    )
    plot_feature_importance(
        classifier, feat_c,
        title="Feature Importance — High-Risk Classification",
        filename="feature_importance_classification.png",
    )

    # Cross-validation (optional)
    reg_cv, clf_cv = None, None
    if args.cv:
        logger.info("  Running cross-validation (this may take several minutes)...")
        X_full_r = pd.concat([X_train_r, X_val_r])
        y_full_r = pd.concat([y_train_r, y_val_r])
        reg_cv   = regressor.cross_validate(X_full_r, y_full_r, cat_cols=cat_cols)

        X_full_c = pd.concat([X_train_c, X_val_c])
        y_full_c = pd.concat([y_train_c, y_val_c])
        clf_cv   = classifier.cross_validate(X_full_c, y_full_c, cat_cols=cat_cols)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5 — SHAP Explainability
    # ──────────────────────────────────────────────────────────────────────
    logger.info("\n[STEP 5/5] Computing SHAP values...")

    # Sample for SHAP (full test set can be slow)
    n_shap = min(config.SHAP_SAMPLE_SIZE, len(X_test_r))
    X_shap_r = X_test_r.sample(n_shap, random_state=config.RANDOM_STATE)
    X_shap_c = X_test_c.iloc[:len(X_shap_r)] if len(X_test_c) >= n_shap else X_test_c

    compute_shap(regressor,  X_shap_r, task="regression")
    compute_shap(classifier, X_shap_c, task="classification")

    # ──────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────────────────────────────
    print_final_summary(reg_metrics, clf_metrics, reg_cv, clf_cv)
    logger.info(f"\nTotal pipeline time: {(time.time()-t0)/60:.1f} minutes")

    return regressor, classifier, encoders


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(args):
    """Load saved models and run batch inference on new data."""
    from inference import SatelliteCollisionPredictor
    import pickle

    # Load encoders
    enc_path = os.path.join(config.MODEL_DIR, "encoders.pkl")
    encoders = None
    if os.path.exists(enc_path):
        with open(enc_path, "rb") as f:
            encoders = pickle.load(f)

    predictor = SatelliteCollisionPredictor(encoders=encoders)
    df_preds  = predictor.predict_batch(args.input, output_path=args.output)
    predictor.generate_alert_report(df_preds)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE — runs with synthetic data if no CSV provided
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    """
    Demo mode: generates synthetic data matching the real dataset schema,
    runs the full pipeline so you can verify everything works before
    loading the real 162k-row CSV.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    logger.info("=" * 60)
    logger.info("  DEMO MODE — Generating synthetic data")
    logger.info("=" * 60)

    # ── Generate synthetic data matching real schema ──────────────────────
    rng = np.random.default_rng(42)
    n   = 5000

    raw_p = np.array([0.15,0.12,0.10,0.08,0.07,0.06,0.05,0.05,0.04,0.04,
                      0.03,0.03,0.03,0.02,0.02,0.02,0.02,0.02,0.01,0.01,
                      0.01,0.01,0.01,0.01])
    raw_p /= raw_p.sum()

    miss_dist   = rng.exponential(8000, n).clip(17, 65900)
    rel_speed   = rng.exponential(3000, n).clip(58, 17000)
    tca         = rng.uniform(2, 6.99, n)
    sigma_r_t   = rng.exponential(10, n).clip(0.1, 500)
    sigma_r_c   = rng.exponential(100, n).clip(1, 5000)

    base_risk = (
        -15
        - 0.9 * np.log1p(miss_dist / 1000)
        + 0.4 * np.log1p(rel_speed / 1000)
        - 0.2 * tca
        - 0.3 * np.log1p(sigma_r_t)
        + rng.normal(0, 1.5, n)
    )
    risk = base_risk.clip(-30, -1.53)

    df = pd.DataFrame({
        "event_id":               range(n),
        "time_to_tca":            tca,
        "mission_id":             rng.choice(range(1,25), n, p=raw_p),
        "risk":                   risk,
        "max_risk_estimate":      (risk * 0.35 + rng.normal(0, 0.5, n)).clip(-10, -1.26),
        "max_risk_scaling":       rng.exponential(0.5, n).clip(0, 4.16),
        "miss_distance":          miss_dist,
        "relative_speed":         rel_speed,
        "relative_position_r":    rng.normal(0, 500, n),
        "relative_position_t":    rng.normal(0, 15000, n),
        "relative_position_n":    rng.normal(0, 10000, n),
        "relative_velocity_r":    rng.normal(0, 50, n),
        "relative_velocity_t":    -rng.exponential(3000, n),
        "relative_velocity_n":    rng.normal(0, 4000, n),
        "t_actual_od_span":       rng.uniform(5, 10, n),
        "t_recommended_od_span":  rng.uniform(6, 10, n),
        "t_obs_available":        rng.integers(100, 300, n).astype(float),
        "t_obs_used":             rng.integers(80, 250, n).astype(float),
        "t_weighted_rms":         rng.uniform(0.8, 2.0, n),
        "t_residuals_accepted":   rng.uniform(95, 100, n),
        "t_rcs_estimate":         rng.exponential(0.5, n),
        "c_rcs_estimate":         np.where(rng.random(n) > 0.3, rng.exponential(0.3, n), np.nan),
        "t_cd_area_over_mass":    rng.uniform(0.01, 0.03, n),
        "c_cd_area_over_mass":    rng.uniform(0.01, 0.5, n),
        "t_cr_area_over_mass":    rng.uniform(0.01, 0.03, n),
        "c_cr_area_over_mass":    rng.uniform(0.01, 0.5, n),
        "t_sedr":                 rng.exponential(1e-5, n),
        "c_sedr":                 rng.exponential(1e-4, n),
        "t_j2k_sma":              rng.uniform(6900, 7200, n),
        "c_j2k_sma":              rng.uniform(6900, 7200, n),
        "t_j2k_ecc":              rng.exponential(0.003, n).clip(0, 0.05),
        "c_j2k_ecc":              rng.exponential(0.003, n).clip(0, 0.05),
        "t_j2k_inc":              rng.uniform(50, 100, n),
        "c_j2k_inc":              rng.uniform(50, 100, n),
        "t_h_apo":                rng.uniform(650, 800, n),
        "t_h_per":                rng.uniform(630, 780, n),
        "c_h_apo":                rng.uniform(650, 800, n),
        "c_h_per":                rng.uniform(630, 780, n),
        "t_span":                 rng.uniform(1, 3, n),
        "c_span":                 rng.uniform(1, 3, n),
        "t_sigma_r":              sigma_r_t,
        "c_sigma_r":              sigma_r_c,
        "t_sigma_t":              rng.exponential(200, n).clip(1, 50000),
        "c_sigma_t":              rng.exponential(5000, n).clip(10, 100000),
        "t_sigma_n":              rng.exponential(30, n).clip(0.5, 2000),
        "c_sigma_n":              rng.exponential(500, n).clip(5, 20000),
        "t_sigma_rdot":           rng.exponential(0.5, n),
        "c_sigma_rdot":           rng.exponential(2, n),
        "t_sigma_tdot":           rng.exponential(0.01, n),
        "c_sigma_tdot":           rng.exponential(0.05, n),
        "t_sigma_ndot":           rng.exponential(0.02, n),
        "c_sigma_ndot":           rng.exponential(0.1, n),
        "t_position_covariance_det": rng.exponential(1e8, n),
        "c_position_covariance_det": rng.exponential(1e16, n),
        "mahalanobis_distance":   rng.exponential(30, n).clip(1, 500),
        "geocentric_latitude":    rng.uniform(-90, 90, n),
        "azimuth":                rng.uniform(0, 360, n),
        "elevation":              rng.uniform(0, 90, n),
        "F10":                    rng.uniform(65, 200, n),
        "F3M":                    rng.uniform(65, 200, n),
        "SSN":                    rng.uniform(0, 200, n),
        "AP":                     rng.integers(0, 50, n).astype(float),
        "c_object_type":          rng.choice(["UNKNOWN", "DEBRIS", "PAYLOAD", "ROCKET BODY"], n,
                                             p=[0.4, 0.35, 0.2, 0.05]),
        # Covariance terms (sample)
        **{col: rng.normal(0, 0.1, n) for col in [
            "t_ct_r","t_cn_r","t_cn_t",
            "t_crdot_r","t_crdot_t","t_crdot_n",
            "t_ctdot_r","t_ctdot_t","t_ctdot_n","t_ctdot_rdot",
            "t_cndot_r","t_cndot_t","t_cndot_n","t_cndot_rdot","t_cndot_tdot",
            "c_ct_r","c_cn_r","c_cn_t",
            "c_crdot_r","c_crdot_t","c_crdot_n",
            "c_ctdot_r","c_ctdot_t","c_ctdot_n","c_ctdot_rdot",
            "c_cndot_r","c_cndot_t","c_cndot_n","c_cndot_rdot","c_cndot_tdot",
        ]},
        # Observation timing
        "t_time_lastob_start":    rng.uniform(0.5, 2, n),
        "t_time_lastob_end":      rng.uniform(0, 0.5, n),
        "c_time_lastob_start":    rng.uniform(100, 200, n),
        "c_time_lastob_end":      rng.uniform(0, 5, n),
        "c_actual_od_span":       rng.uniform(20, 35, n),
        "c_recommended_od_span":  rng.uniform(25, 35, n),
        "c_obs_available":        rng.integers(10, 25, n).astype(float),
        "c_obs_used":             rng.integers(8, 20, n).astype(float),
        "c_weighted_rms":         rng.uniform(2, 6, n),
        "c_residuals_accepted":   rng.uniform(75, 95, n),
    })

    os.makedirs("data", exist_ok=True)
    demo_path = "data/demo_satellite_data.csv"
    df.to_csv(demo_path, index=False)
    logger.info(f"Demo dataset saved → {demo_path}  ({df.shape})")
    return demo_path


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.infer:
        if not args.input:
            logger.error("--infer requires --input <csv_path>")
            sys.exit(1)
        run_inference(args)

    else:
        # Check if real data exists; otherwise run demo
        data_path = args.data
        if not os.path.exists(data_path):
            logger.warning(f"Data not found at '{data_path}' — running DEMO mode with synthetic data")
            logger.warning("To use your real data, set DATA_PATH in config.py or pass --data <path>")
            data_path = run_demo()
            args.data = data_path

        regressor, classifier, encoders = run_pipeline(args)

        # Save encoders for inference
        import pickle
        enc_path = os.path.join(config.MODEL_DIR, "encoders.pkl")
        with open(enc_path, "wb") as f:
            pickle.dump(encoders, f)
        logger.info(f"Encoders saved → {enc_path}")
