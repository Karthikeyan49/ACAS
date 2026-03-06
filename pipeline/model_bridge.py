"""
pipeline/model_bridge.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    The single import api/endpoint.py needs. Chains tle_processor →
    LightGBM → structured response dict in one function call.

CALLED FROM
    api/endpoint.py   CollisionRiskPredictor(), predict_collision_risk()

CALLS INTO
    pipeline/tle_processor.py    tle_pair_to_cdm_features()
    model/config.py               MODEL_DIR
    data/data_pipeline.py         impute_missing, clip_outliers,
                                   engineer_features, encode_categoricals
    model/lgbm_model.py           SatelliteRiskRegressor.load(),
                                   SatelliteRiskClassifier.load()
    pickle, numpy, pandas

WHAT IT PROVIDES
    CollisionRiskPredictor()
        predict(satellite_tle_str, object_tle_str, **kwargs) → dict
        health_check() → dict

    predict_collision_risk(satellite_tle, object_tle, **kwargs) → dict
        Convenience function. Instantiates predictor and calls predict().

RESPONSE DICT
    status, timestamp,
    risk:        {score, probability, probability_formatted,
                  alert_probability, is_high_risk, threshold_used}
    alert:       {level, colour, action, model_flag}
    conjunction: {tca_datetime, tca_minutes_from_now, miss_distance_km,
                  relative_speed_ms, position_ric_m}
    satellite:   {name, norad_id, altitude_km, inclination_deg, tle_age_days}
    object:      {name, norad_id, type, altitude_km, tle_age_days}
    manoeuvre:   {required, delta_v_ms, direction}

IMPORT CHANGES FROM ORIGINAL (satellite_lgbm/model_bridge.py)
    from tle_processor import →  from pipeline.tle_processor import
══════════════════════════════════════════════════════════════════════════════
"""
"""
model_bridge.py
───────────────
DROP-IN REPLACEMENT for your old synthetic-data model.

This is the ONLY file your dashboard backend needs to import.
It exposes one function:

    predict_collision_risk(satellite_tle, object_tle, **kwargs)

Your old backend call:
    result = old_model.predict(synthetic_features)

New backend call (replace with this):
    from model_bridge import CollisionRiskPredictor
    predictor = CollisionRiskPredictor()
    result = predictor.predict(satellite_tle_str, object_tle_str)

Returns a structured dict the dashboard can render directly.
"""

import os
import sys
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
logger = logging.getLogger("model_bridge")

# ── Locate pipeline modules ───────────────────────────────────────────────────
from model import config
from model.lgbm_model import SatelliteRiskRegressor, SatelliteRiskClassifier
from data.data_pipeline import (impute_missing, clip_outliers,
                                  engineer_features, encode_categoricals)
from pipeline.tle_processor import tle_pair_to_cdm_features


# ═════════════════════════════════════════════════════════════════════════════
# ALERT LEVELS — maps model output to dashboard severity levels
# ═════════════════════════════════════════════════════════════════════════════

def _risk_to_alert_level(risk_score: float, collision_prob: float) -> Dict:
    """
    Map risk score → operational alert level for dashboard display.

    Follows ESA/NASA Space Debris Mitigation Guidelines thresholds.
    """
    if risk_score > -4.0:       # P > 1e-4
        return {
            "level":       "CRITICAL",
            "colour":      "#B71C1C",     # deep red
            "action":      "EMERGENCY MANOEUVRE REQUIRED",
            "description": "Collision probability exceeds 1:10,000. "
                           "Immediate avoidance manoeuvre mandatory.",
            "priority":    1,
        }
    elif risk_score > -5.0:     # P > 1e-5
        return {
            "level":       "HIGH",
            "colour":      "#FF6F00",     # amber
            "action":      "MANOEUVRE RECOMMENDED",
            "description": "Collision probability exceeds ESA alert threshold (1:100,000). "
                           "Evaluate avoidance manoeuvre.",
            "priority":    2,
        }
    elif risk_score > -6.0:     # P > 1e-6
        return {
            "level":       "ELEVATED",
            "colour":      "#F9A825",     # yellow
            "action":      "ENHANCED MONITORING",
            "description": "Elevated risk. Monitor closely. "
                           "Prepare contingency manoeuvre plan.",
            "priority":    3,
        }
    elif risk_score > -8.0:     # P > 1e-8
        return {
            "level":       "LOW",
            "colour":      "#1565C0",     # blue
            "action":      "LOG AND WATCH",
            "description": "Low risk. Log event and continue monitoring.",
            "priority":    4,
        }
    else:
        return {
            "level":       "NOMINAL",
            "colour":      "#2E7D32",     # green
            "action":      "NO ACTION REQUIRED",
            "description": "Collision probability below monitoring threshold.",
            "priority":    5,
        }


def _manoeuvre_recommendation(
    risk_score: float,
    time_to_tca_days: float,
    relative_speed_ms: float,
    miss_distance_m: float,
) -> Dict:
    """
    Generate manoeuvre recommendation parameters.
    Simplified delta-V estimation for prototype dashboard.
    """
    if risk_score <= -6.0:
        return {"required": False, "delta_v_ms": 0.0, "burn_time_s": 0.0,
                "direction": "NONE", "window_hours": 0.0}

    # Simple delta-V estimate (Clohessy-Wiltshire approximation)
    # Target: increase miss distance to at least 2km
    target_miss_m = max(2000.0, miss_distance_m * 3)
    delta_miss_m  = target_miss_m - miss_distance_m
    delta_v_ms    = delta_miss_m / max(1.0, time_to_tca_days * 86400)

    # Manoeuvre window: should execute at least 24h before TCA
    window_hours = max(0.0, time_to_tca_days * 24 - 24)

    return {
        "required":       True,
        "delta_v_ms":     round(delta_v_ms, 4),
        "burn_time_s":    round(delta_v_ms / 0.1, 1),   # assuming 0.1 m/s² thrust
        "direction":      "RADIAL" if abs(risk_score) < 6 else "IN-TRACK",
        "window_hours":   round(window_hours, 1),
        "execute_before": f"{window_hours:.1f} hours from now",
    }


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTOR CLASS — singleton, loaded once at startup
# ═════════════════════════════════════════════════════════════════════════════

class CollisionRiskPredictor:
    """
    Drop-in replacement for old synthetic model.

    Usage in your backend:
        predictor = CollisionRiskPredictor()   # load once at startup
        result    = predictor.predict(sat_tle, obj_tle)
    """

    _instance = None   # singleton

    def __new__(cls, model_dir: str = None):
        """Singleton — only load models once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self, model_dir: str = None):
        if self._loaded:
            return

        self.model_dir = model_dir or (
            config.MODEL_DIR
        )
        self._load_models()
        self._loaded = True

    def _load_models(self):
        """Load LightGBM regressor, classifier, and encoders."""

        if not self.model_dir:
            logger.warning("Running in DEMO mode — models not loaded")
            self.regressor  = None
            self.classifier = None
            self.encoders   = None
            self._demo_mode = True
            return

        reg_path = os.path.join(self.model_dir, "regressor.pkl")
        clf_path = os.path.join(self.model_dir, "classifier.pkl")
        enc_path = os.path.join(self.model_dir, "encoders.pkl")

        if not os.path.exists(reg_path):
            raise FileNotFoundError(
                f"Trained model not found at {reg_path}\n"
                f"Run: python main.py --data train_data.csv --tune --cv"
            )

        self.regressor  = SatelliteRiskRegressor.load(reg_path)
        self.classifier = SatelliteRiskClassifier.load(clf_path)
        self.encoders   = None

        if os.path.exists(enc_path):
            with open(enc_path, "rb") as f:
                self.encoders = pickle.load(f)

        self._demo_mode = False
        logger.info(
            f"LightGBM models loaded | "
            f"Reg iter={self.regressor.best_iteration} | "
            f"Clf iter={self.classifier.best_iteration} | "
            f"Threshold={self.classifier.optimal_threshold:.3f}"
        )

    def _preprocess_features(self, raw_features: Dict) -> pd.DataFrame:
        """Apply full training preprocessing pipeline to raw CDM features."""

        # Strip _meta key before feeding to model
        features = {k: v for k, v in raw_features.items() if k != "_meta"}

        df = pd.DataFrame([features])

        df = impute_missing(df)
        df = clip_outliers(df)
        df = engineer_features(df)

        if self.encoders:
            df, _ = encode_categoricals(df, encoders=self.encoders, fit=False)
        else:
            df, _ = encode_categoricals(df, fit=True)

        return df

    def _align(self, df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
        """Fill missing columns with 0 and reorder to match training."""
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0
        return df[feature_names]

    def _demo_predict(self, features: Dict) -> Dict:
        """Fallback demo prediction when models are not loaded."""
        miss_dist = features.get("miss_distance", 10000)
        speed     = features.get("relative_speed", 5000)

        # Approximate risk using physics
        risk = -6 - np.log10(max(1, miss_dist / 100)) + np.log10(max(1, speed / 1000))
        risk = float(np.clip(risk, -30, -1.5))

        return risk, 10 ** risk, 0.5 if risk > -6 else 0.1

    def predict(
        self,
        satellite_tle_str:  str,
        object_tle_str:     str,
        object_type:        str   = "UNKNOWN",
        satellite_rcs_m2:   float = None,
        object_rcs_m2:      float = None,
        space_weather:      Dict  = None,
        mission_id:         int   = 1,
        event_id:           int   = 0,
    ) -> Dict:
        """
        ╔══════════════════════════════════════════════════════════╗
        ║  MAIN PREDICTION METHOD                                  ║
        ║                                                          ║
        ║  Input : Two raw TLE strings from dashboard              ║
        ║  Output: Full structured response for dashboard          ║
        ╚══════════════════════════════════════════════════════════╝

        Parameters
        ----------
        satellite_tle_str : TLE of your satellite (3-line string)
        object_tle_str    : TLE of threat object (3-line string)
        object_type       : "PAYLOAD"|"ROCKET BODY"|"DEBRIS"|"UNKNOWN"
        satellite_rcs_m2  : RCS of satellite in m² (optional)
        object_rcs_m2     : RCS of object in m² (optional)
        space_weather     : {"F10":..,"F3M":..,"SSN":..,"AP":..} (optional)
        mission_id        : integer mission ID
        event_id          : integer event counter

        Returns
        -------
        dict — full dashboard-ready response (see structure below)
        """

        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # ── STEP 1: TLE → CDM features ────────────────────────────────
            logger.info("Processing TLE pair...")
            raw_features = tle_pair_to_cdm_features(
                satellite_tle_str  = satellite_tle_str,
                object_tle_str     = object_tle_str,
                object_type        = object_type,
                satellite_rcs_m2   = satellite_rcs_m2,
                object_rcs_m2      = object_rcs_m2,
                space_weather      = space_weather,
                mission_id         = mission_id,
                event_id           = event_id,
            )
            meta = raw_features.get("_meta", {})

            # ── STEP 2: Model prediction ──────────────────────────────────
            if self._demo_mode:
                risk_score, coll_prob, alert_prob = self._demo_predict(raw_features)
                threshold = 0.65
            else:
                df = self._preprocess_features(raw_features)

                X_reg = self._align(df.copy(), self.regressor.feature_names)
                X_clf = self._align(df.copy(), self.classifier.feature_names)

                risk_score  = float(self.regressor.predict(X_reg)[0])
                alert_prob  = float(self.classifier.predict_proba(X_clf)[0])
                coll_prob   = 10 ** risk_score
                threshold   = self.classifier.optimal_threshold

            # ── STEP 3: Classify alert level ──────────────────────────────
            alert_level = _risk_to_alert_level(risk_score, coll_prob)
            is_high_risk = alert_prob >= threshold

            # ── STEP 4: Manoeuvre recommendation ──────────────────────────
            manoeuvre = _manoeuvre_recommendation(
                risk_score        = risk_score,
                time_to_tca_days  = raw_features["time_to_tca"],
                relative_speed_ms = raw_features["relative_speed"],
                miss_distance_m   = raw_features["miss_distance"],
            )

            # ── STEP 5: Build full dashboard response ─────────────────────
            return {
                "status": "success",
                "timestamp": timestamp,

                # ── PRIMARY OUTPUTS (what dashboard shows prominently) ──────
                "risk": {
                    "score":       round(risk_score, 6),
                    "probability": float(f"{coll_prob:.4e}"),
                    "probability_formatted": f"{coll_prob:.2e}",
                    "alert_probability": round(alert_prob, 4),
                    "is_high_risk": bool(is_high_risk),
                    "threshold_used": round(threshold, 3),
                },

                # ── ALERT LEVEL (colour + action for dashboard UI) ──────────
                "alert": {
                    **alert_level,
                    "model_flag": "HIGH_RISK" if is_high_risk else "NORMAL",
                },

                # ── CONJUNCTION GEOMETRY (for orbit visualisation) ──────────
                "conjunction": {
                    "tca_datetime":       meta.get("tca_datetime"),
                    "tca_minutes_from_now": raw_features.get("time_to_tca", 0) * 1440,
                    "time_to_tca_days":   round(raw_features["time_to_tca"], 4),
                    "miss_distance_m":    round(raw_features["miss_distance"], 1),
                    "miss_distance_km":   round(raw_features["miss_distance"] / 1000, 4),
                    "relative_speed_ms":  round(raw_features["relative_speed"], 1),
                    "relative_speed_kms": round(raw_features["relative_speed"] / 1000, 3),
                    "position_ric_m": {
                        "radial":    round(raw_features["relative_position_r"], 2),
                        "intrack":   round(raw_features["relative_position_t"], 2),
                        "crosstrack":round(raw_features["relative_position_n"], 2),
                    },
                    "velocity_ric_ms": {
                        "radial":    round(raw_features["relative_velocity_r"], 4),
                        "intrack":   round(raw_features["relative_velocity_t"], 4),
                        "crosstrack":round(raw_features["relative_velocity_n"], 4),
                    },
                },

                # ── SATELLITE INFO ───────────────────────────────────────────
                "satellite": {
                    "name":        meta.get("satellite_name"),
                    "norad_id":    meta.get("satellite_norad"),
                    "altitude_km": round(
                        (raw_features["t_h_apo"] + raw_features["t_h_per"]) / 2, 1
                    ),
                    "inclination_deg": round(raw_features["t_j2k_inc"], 3),
                    "eccentricity":    round(raw_features["t_j2k_ecc"], 6),
                    "tle_age_days":    round(meta.get("sat_tle_age_days", 0), 2),
                },

                # ── OBJECT INFO ──────────────────────────────────────────────
                "object": {
                    "name":        meta.get("object_name"),
                    "norad_id":    meta.get("object_norad"),
                    "type":        meta.get("object_type", object_type),
                    "altitude_km": round(
                        (raw_features["c_h_apo"] + raw_features["c_h_per"]) / 2, 1
                    ),
                    "inclination_deg": round(raw_features["c_j2k_inc"], 3),
                    "tle_age_days":    round(meta.get("obj_tle_age_days", 0), 2),
                },

                # ── MANOEUVRE RECOMMENDATION ──────────────────────────────────
                "manoeuvre": manoeuvre,

                # ── UNCERTAINTY INFO (for confidence display) ─────────────────
                "uncertainty": {
                    "satellite_sigma_r_m":  round(raw_features["t_sigma_r"], 2),
                    "object_sigma_r_m":     round(raw_features["c_sigma_r"], 2),
                    "combined_sigma_r_m":   round(
                        np.sqrt(raw_features["t_sigma_r"]**2 +
                                raw_features["c_sigma_r"]**2), 2
                    ),
                    "mahalanobis_distance": round(raw_features["mahalanobis_distance"], 2),
                },

                # ── OPERATIONAL THRESHOLDS (for dashboard legend) ─────────────
                "thresholds": {
                    "emergency_manoeuvre": {"risk_score": -4.0, "probability": "1e-4"},
                    "manoeuvre_recommended": {"risk_score": -5.0, "probability": "1e-5"},
                    "enhanced_monitoring":  {"risk_score": -6.0, "probability": "1e-6"},
                    "log_and_watch":        {"risk_score": -8.0, "probability": "1e-8"},
                },
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return {
                "status":    "error",
                "timestamp": timestamp,
                "error":     str(e),
                "message":   "Prediction failed — check TLE format and model files",
            }

    def health_check(self) -> Dict:
        """Returns model status for dashboard health endpoint."""
        return {
            "status":      "healthy" if not self._demo_mode else "demo_mode",
            "models_loaded": not self._demo_mode,
            "regressor_iterations":  getattr(getattr(self, "regressor", None),
                                              "best_iteration", None),
            "classifier_iterations": getattr(getattr(self, "classifier", None),
                                              "best_iteration", None),
            "classifier_threshold":  getattr(getattr(self, "classifier", None),
                                              "optimal_threshold", None),
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }


# ═════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION — for simple one-line usage in backend
# ═════════════════════════════════════════════════════════════════════════════

def predict_collision_risk(
    satellite_tle: str,
    object_tle:    str,
    object_type:   str  = "UNKNOWN",
    **kwargs
) -> Dict:
    """
    One-line function for backend usage.
    Equivalent to CollisionRiskPredictor().predict(...)

    Example in your FastAPI backend:
        from model_bridge import predict_collision_risk

        @app.post("/api/predict")
        def predict(data: TLEInput):
            return predict_collision_risk(
                satellite_tle = data.satellite_tle,
                object_tle    = data.object_tle,
                object_type   = data.object_type,
            )
    """
    predictor = CollisionRiskPredictor()
    return predictor.predict(satellite_tle, object_tle, object_type, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST — run directly to verify the bridge works
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 65)
    print("  MODEL BRIDGE — Standalone Test")
    print("=" * 65)

    # Sample TLE pair (ISS + debris)
    SAT_TLE = """ISS (ZARYA)
1 25544U 98067A   26063.51953704  .00002182  00000-0  40768-4 0  9995
2 25544  51.6435 275.1234 0004567  89.1234 271.0234 15.49567890123456"""

    OBJ_TLE = """COSMOS 2251 DEB
1 33442U 93036ABH  26063.12345678  .00000100  00000-0  12345-4 0  9991
2 33442  74.0234 123.4567 0012345 234.5678 125.4321 14.34567890123456"""

    predictor = CollisionRiskPredictor()

    print("\nHealth check:")
    print(json.dumps(predictor.health_check(), indent=2))

    print("\nRunning prediction...")
    result = predictor.predict(
        satellite_tle_str = SAT_TLE,
        object_tle_str    = OBJ_TLE,
        object_type       = "DEBRIS",
        mission_id        = 1,
        event_id          = 0,
    )

    print("\nFull API Response:")
    # Remove non-serialisable items for display
    display = {k: v for k, v in result.items() if k != "_meta"}
    print(json.dumps(display, indent=2, default=str))

    if result["status"] == "success":
        print("\n" + "=" * 65)
        print(f"  ALERT LEVEL     : {result['alert']['level']}")
        print(f"  RISK SCORE      : {result['risk']['score']}")
        print(f"  COLLISION PROB  : {result['risk']['probability_formatted']}")
        print(f"  MISS DISTANCE   : {result['conjunction']['miss_distance_km']} km")
        print(f"  TIME TO TCA     : {result['conjunction']['time_to_tca_days']} days")
        print(f"  ACTION          : {result['alert']['action']}")
        if result["manoeuvre"]["required"]:
            print(f"  DELTA-V         : {result['manoeuvre']['delta_v_ms']} m/s")
        print("=" * 65)