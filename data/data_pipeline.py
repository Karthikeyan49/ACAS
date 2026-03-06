"""
data/data_pipeline.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    All data preparation for the LightGBM model. Loads the raw CDM CSV,
    handles missing values, engineers 35+ new features from orbital
    mechanics, encodes categoricals, and produces train/val/test splits.

CALLED FROM
    model/lgbm_engine.py    impute_missing, clip_outliers,
                             engineer_features, encode_categoricals
    model/inference.py      same four functions
    model/train.py          prepare_data()

CALLS INTO
    model/config.py         DATA_PATH, DROP_COLS, CATEGORICAL_COLS,
                             TEST_SIZE, VAL_SIZE, RANDOM_STATE
    sklearn                 train_test_split, LabelEncoder
    pandas, numpy

WHAT IT PROVIDES
    load_data(path) → pd.DataFrame
    missing_value_report(df) → pd.DataFrame
    impute_missing(df) → pd.DataFrame
        covariance cols → 0 (uncorrelated)
        sigma cols      → median
        orbital cols    → median
        space weather   → forward-fill then median
    clip_outliers(df, lower=0.001, upper=0.999) → pd.DataFrame
    engineer_features(df) → pd.DataFrame
        position_magnitude, velocity_magnitude
        normalised_miss_distance  (miss / combined sigma_r)
        mahalanobis_distance
        combined_covariance_trace
        log covariance determinants
        orbital element differences (SMA, inc, ecc)
        OD span ratio (actual / recommended)
        space weather interactions (F10 × SEDR, AP × SEDR)
        kinetic energy proxy
        TCA time bins (0-2h, 2-6h, 6-24h, >24h)
    encode_categoricals(df, encoders=None, fit=True) → (df, encoders)
    split_data(df) → (X_train, X_val, X_test, y_reg_*, y_clf_*)
    prepare_data(path) → tuple
        Full pipeline in one call: load→impute→clip→engineer→encode→split.

IMPORT CHANGE FROM ORIGINAL (satellite_lgbm/data_pipeline.py)
    import config  →  from model import config
══════════════════════════════════════════════════════════════════════════════
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from model import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP DEFINITIONS
# Used for targeted imputation strategies
# ─────────────────────────────────────────────────────────────────────────────

# Covariance matrix elements — physically bounded, impute with 0 (uncorrelated)
COVARIANCE_COLS = [
    "t_ct_r", "t_cn_r", "t_cn_t",
    "t_crdot_r", "t_crdot_t", "t_crdot_n",
    "t_ctdot_r", "t_ctdot_t", "t_ctdot_n", "t_ctdot_rdot",
    "t_cndot_r", "t_cndot_t", "t_cndot_n", "t_cndot_rdot", "t_cndot_tdot",
    "c_ct_r", "c_cn_r", "c_cn_t",
    "c_crdot_r", "c_crdot_t", "c_crdot_n",
    "c_ctdot_r", "c_ctdot_t", "c_ctdot_n", "c_ctdot_rdot",
    "c_cndot_r", "c_cndot_t", "c_cndot_n", "c_cndot_rdot", "c_cndot_tdot",
]

# Sigma (uncertainty) columns — impute with median
SIGMA_COLS = [
    "t_sigma_r", "c_sigma_r", "t_sigma_t", "c_sigma_t",
    "t_sigma_n", "c_sigma_n", "t_sigma_rdot", "c_sigma_rdot",
    "t_sigma_tdot", "c_sigma_tdot", "t_sigma_ndot", "c_sigma_ndot",
]

# Orbital mechanics — impute with median
ORBITAL_COLS = [
    "t_j2k_sma", "t_j2k_ecc", "t_j2k_inc",
    "c_j2k_sma", "c_j2k_ecc", "c_j2k_inc",
    "t_h_apo", "t_h_per", "c_h_apo", "c_h_per",
    "t_span", "c_span",
]

# Observation quality — impute with median
OBSERVATION_COLS = [
    "t_time_lastob_start", "t_time_lastob_end",
    "t_recommended_od_span", "t_actual_od_span",
    "t_obs_available", "t_obs_used",
    "t_residuals_accepted", "t_weighted_rms",
    "c_time_lastob_start", "c_time_lastob_end",
    "c_recommended_od_span", "c_actual_od_span",
    "c_obs_available", "c_obs_used",
    "c_residuals_accepted", "c_weighted_rms",
]

# Physical properties — RCS can be NaN for unknown objects, impute with median
PHYSICAL_COLS = [
    "t_rcs_estimate", "c_rcs_estimate",
    "t_cd_area_over_mass", "c_cd_area_over_mass",
    "t_cr_area_over_mass", "c_cr_area_over_mass",
    "t_sedr", "c_sedr",
]

# Space weather — global indices, impute with forward fill then median
SPACE_WEATHER_COLS = ["F10", "F3M", "SSN", "AP"]


# ─────────────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str = config.DATA_PATH) -> pd.DataFrame:
    """Load CSV with efficient dtypes."""
    logger.info(f"Loading dataset from: {path}")

    df = pd.read_csv(
        path,
        low_memory=False,
        na_values=["", "NA", "N/A", "nan", "NaN", "null", "NULL", "None"],
    )

    logger.info(f"Loaded shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Downcast numeric columns to save memory
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    logger.info(f"Memory after downcast: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MISSING VALUE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return sorted DataFrame of missing value statistics."""
    total   = df.isnull().sum()
    percent = (total / len(df) * 100).round(2)
    report  = pd.DataFrame({"missing_count": total, "missing_pct": percent})
    report  = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)
    if not report.empty:
        logger.info(f"\nMissing value report ({len(report)} columns affected):\n{report.to_string()}")
    else:
        logger.info("No missing values found.")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# IMPUTATION — Group-aware strategy
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply domain-appropriate imputation per feature group."""
    df = df.copy()

    # Covariance matrix elements → 0 (physically: no correlation known)
    for col in COVARIANCE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # All other numeric groups → median
    median_groups = [SIGMA_COLS, ORBITAL_COLS, OBSERVATION_COLS, PHYSICAL_COLS]
    for group in median_groups:
        for col in group:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    # Space weather → forward fill (temporally correlated), then median
    for col in SPACE_WEATHER_COLS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
            df[col] = df[col].fillna(df[col].median())

    # Remaining numeric → median fallback
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Remaining categorical → "UNKNOWN"
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna("UNKNOWN")

    remaining = df.isnull().sum().sum()
    logger.info(f"Remaining nulls after imputation: {remaining}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# OUTLIER CLIPPING — Winsorise extreme values per group
# ─────────────────────────────────────────────────────────────────────────────

def clip_outliers(df: pd.DataFrame, lower: float = 0.001, upper: float = 0.999) -> pd.DataFrame:
    """Winsorise outliers in physical measurement columns only."""
    df = df.copy()
    clip_cols = [
        "miss_distance", "relative_speed", "mahalanobis_distance",
        "t_sigma_r", "c_sigma_r", "t_sigma_t", "c_sigma_t",
        "t_position_covariance_det", "c_position_covariance_det",
    ]
    for col in clip_cols:
        if col in df.columns:
            lo = df[col].quantile(lower)
            hi = df[col].quantile(upper)
            df[col] = df[col].clip(lo, hi)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — Domain-informed derived features
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create physics-informed derived features that encode domain knowledge
    the model would otherwise have to learn from scratch.
    """
    df = df.copy()

    # ── Relative state ──────────────────────────────────────────────────────
    # 3D miss distance magnitude (confirm/cross-check scalar miss_distance)
    if all(c in df for c in ["relative_position_r", "relative_position_t", "relative_position_n"]):
        df["position_3d_magnitude"] = np.sqrt(
            df["relative_position_r"]**2 +
            df["relative_position_t"]**2 +
            df["relative_position_n"]**2
        )

    # Velocity magnitude
    if all(c in df for c in ["relative_velocity_r", "relative_velocity_t", "relative_velocity_n"]):
        df["velocity_3d_magnitude"] = np.sqrt(
            df["relative_velocity_r"]**2 +
            df["relative_velocity_t"]**2 +
            df["relative_velocity_n"]**2
        )

    # Time-to-impact proxy: miss_distance / relative_speed
    if "miss_distance" in df and "relative_speed" in df:
        df["time_to_impact_proxy"] = df["miss_distance"] / (df["relative_speed"].replace(0, np.nan) + 1e-9)

    # Kinetic energy proxy (½v²) — proportional to impact energy
    if "relative_speed" in df:
        df["kinetic_energy_proxy"] = 0.5 * df["relative_speed"]**2

    # ── Uncertainty quantification ────────────────────────────────────────
    # Combined positional uncertainty (RSS of sigma_r, sigma_t, sigma_n)
    if all(c in df for c in ["t_sigma_r", "t_sigma_t", "t_sigma_n"]):
        df["t_position_uncertainty"] = np.sqrt(
            df["t_sigma_r"]**2 + df["t_sigma_t"]**2 + df["t_sigma_n"]**2
        )
    if all(c in df for c in ["c_sigma_r", "c_sigma_t", "c_sigma_n"]):
        df["c_position_uncertainty"] = np.sqrt(
            df["c_sigma_r"]**2 + df["c_sigma_t"]**2 + df["c_sigma_n"]**2
        )

    # Combined uncertainty of both objects
    if "t_position_uncertainty" in df and "c_position_uncertainty" in df:
        df["combined_position_uncertainty"] = np.sqrt(
            df["t_position_uncertainty"]**2 + df["c_position_uncertainty"]**2
        )

    # Uncertainty-normalised miss distance — key risk factor
    if "miss_distance" in df and "combined_position_uncertainty" in df:
        df["normalised_miss_distance"] = (
            df["miss_distance"] / (df["combined_position_uncertainty"] + 1e-9)
        )

    # Ratio of uncertainties between target and chaser
    if "t_position_uncertainty" in df and "c_position_uncertainty" in df:
        df["uncertainty_ratio"] = (
            df["t_position_uncertainty"] /
            (df["c_position_uncertainty"] + 1e-9)
        )

    # ── Covariance trace — total uncertainty volume ───────────────────────
    if all(c in df for c in ["t_sigma_r", "t_sigma_t", "t_sigma_n"]):
        df["t_covariance_trace"] = df["t_sigma_r"]**2 + df["t_sigma_t"]**2 + df["t_sigma_n"]**2
    if all(c in df for c in ["c_sigma_r", "c_sigma_t", "c_sigma_n"]):
        df["c_covariance_trace"] = df["c_sigma_r"]**2 + df["c_sigma_t"]**2 + df["c_sigma_n"]**2

    # Log covariance determinants (avoid extreme scale)
    if "t_position_covariance_det" in df:
        df["log_t_cov_det"] = np.log1p(np.abs(df["t_position_covariance_det"]))
    if "c_position_covariance_det" in df:
        df["log_c_cov_det"] = np.log1p(np.abs(df["c_position_covariance_det"]))

    # ── Orbital mechanics ──────────────────────────────────────────────────
    # Semi-major axis difference — orbits close together = higher risk
    if "t_j2k_sma" in df and "c_j2k_sma" in df:
        df["sma_difference"]  = np.abs(df["t_j2k_sma"] - df["c_j2k_sma"])

    # Orbital altitude difference
    if "t_h_per" in df and "c_h_per" in df:
        df["perigee_diff"]    = np.abs(df["t_h_per"] - df["c_h_per"])
    if "t_h_apo" in df and "c_h_apo" in df:
        df["apogee_diff"]     = np.abs(df["t_h_apo"] - df["c_h_apo"])

    # Inclination difference — crossing orbits are higher risk
    if "t_j2k_inc" in df and "c_j2k_inc" in df:
        df["inclination_diff"] = np.abs(df["t_j2k_inc"] - df["c_j2k_inc"])

    # Eccentricity difference
    if "t_j2k_ecc" in df and "c_j2k_ecc" in df:
        df["eccentricity_diff"] = np.abs(df["t_j2k_ecc"] - df["c_j2k_ecc"])

    # ── Observation quality ───────────────────────────────────────────────
    # Observation utilisation ratio
    if "t_obs_used" in df and "t_obs_available" in df:
        df["t_obs_utilisation"] = df["t_obs_used"] / (df["t_obs_available"] + 1e-9)
    if "c_obs_used" in df and "c_obs_available" in df:
        df["c_obs_utilisation"] = df["c_obs_used"] / (df["c_obs_available"] + 1e-9)

    # OD span quality (actual vs recommended)
    if "t_actual_od_span" in df and "t_recommended_od_span" in df:
        df["t_od_quality"] = df["t_actual_od_span"] / (df["t_recommended_od_span"] + 1e-9)
    if "c_actual_od_span" in df and "c_recommended_od_span" in df:
        df["c_od_quality"] = df["c_actual_od_span"] / (df["c_recommended_od_span"] + 1e-9)

    # ── Physical properties ───────────────────────────────────────────────
    # Log RCS (radar cross section — proxy for object size)
    for col in ["t_rcs_estimate", "c_rcs_estimate"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(np.abs(df[col]))

    # Area-to-mass ratio difference (affects drag differently)
    if "t_cd_area_over_mass" in df and "c_cd_area_over_mass" in df:
        df["cd_area_mass_diff"] = np.abs(df["t_cd_area_over_mass"] - df["c_cd_area_over_mass"])

    # ── Space weather interaction ─────────────────────────────────────────
    # High solar activity → atmosphere expands → higher drag → altered orbits
    if "F10" in df and "t_sedr" in df:
        df["solar_drag_interaction"] = df["F10"] * df["t_sedr"]
    if "AP" in df and "t_sedr" in df:
        df["geomag_drag_interaction"] = df["AP"] * df["t_sedr"]

    # ── Time features ──────────────────────────────────────────────────────
    # Time to TCA binned (risk changes non-linearly with time)
    if "time_to_tca" in df:
        df["tca_bin"] = pd.cut(
            df["time_to_tca"],
            bins=[0, 1, 2, 3, 5, 7],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

    # ── TARGET LABEL ──────────────────────────────────────────────────────
    # Binary high-risk label: top N% by risk score
    if "risk" in df:
        threshold = df["risk"].quantile(config.HIGH_RISK_PERCENTILE)
        df["high_risk"] = (df["risk"] > threshold).astype(np.int8)
        logger.info(f"High-risk threshold: {threshold:.4f} | "
                    f"High-risk events: {df['high_risk'].sum()} "
                    f"({df['high_risk'].mean()*100:.1f}%)")

    logger.info(f"Features after engineering: {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORICAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns.
    LightGBM accepts label-encoded integers natively with categorical_feature param.

    Parameters
    ----------
    df       : DataFrame
    encoders : fitted encoders (pass during inference)
    fit      : True during training, False during inference
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in config.CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna("UNKNOWN")

        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen categories gracefully
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "UNKNOWN")
            df[col] = le.transform(df[col])

    return df, encoders


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series,
           List[str]]:
    """
    Split into train / val / test sets.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    """
    drop = config.DROP_COLS + [
        config.REGRESSION_TARGET,
        config.CLASSIFICATION_TARGET,
    ]
    # Only drop columns that exist
    drop = [c for c in drop if c in df.columns]

    # Remove target from drop if it IS the target we want to keep
    if target in drop:
        drop.remove(target)

    feature_cols = [c for c in df.columns if c not in drop and c != target]
    X = df[feature_cols]
    y = df[target]

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y if target == config.CLASSIFICATION_TARGET else None,
    )

    # Second split: train vs val
    val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=config.RANDOM_STATE,
        stratify=y_trainval if target == config.CLASSIFICATION_TARGET else None,
    )

    logger.info(f"Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Feature count: {len(feature_cols)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(path: str = config.DATA_PATH):
    """
    Full data preparation pipeline.

    Returns
    -------
    dict with all splits, encoders, and feature lists for both tasks
    """
    df = load_data(path)
    missing_value_report(df)
    df = impute_missing(df)
    df = clip_outliers(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df, fit=True)

    # Regression splits
    reg_splits = split_data(df, config.REGRESSION_TARGET)

    # Classification splits
    clf_splits = split_data(df, config.CLASSIFICATION_TARGET)

    return {
        "df":            df,
        "encoders":      encoders,
        "regression":    reg_splits,
        "classification": clf_splits,
    }
