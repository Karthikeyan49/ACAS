"""
lgbm_inference_engine.py
────────────────────────
DROP-IN REPLACEMENT for OnnxInferenceEngine inside acas_controller.py.

WHAT THIS DOES:
  Old system: extract_features(conj) → 12 floats → ONNX → raw_pc
  New system: conj dict → tle_processor → 103 CDM features → LightGBM → raw_pc

The output is the same: a single float raw_pc in [0.0, 1.0]
RiskScorer receives the same value it always did — no other changes needed.

HOW TO SWAP IN acas_controller.py:
  Old:  self.onnx_engine = OnnxInferenceEngine(ONNX_MODEL_PATH)
  New:  from lgbm_inference_engine import LGBMInferenceEngine
        self.onnx_engine = LGBMInferenceEngine()

  The predict_pc(features) call signature is PRESERVED — no further changes.
  But the new engine also accepts predict_pc_from_conjunction(conj) for
  a richer prediction that skips the 12-feature intermediary entirely.
"""

import os
import sys
import pickle
import logging
import warnings
import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger("lgbm_engine")

# ── Locate project root ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # one level up from onboard/

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    logger.warning("pandas not installed — pip install pandas")


# ═════════════════════════════════════════════════════════════════════════════
# DEFAULT SATELLITE TLE (from acas_controller.py config)
# ─ the engine needs your satellite's TLE to compute CDM features from TCA data
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_SAT_TLE = """ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00005764  00000-0  10780-3 0  9993
2 25544  51.6416 290.0015 0002627  55.4917 344.9690 15.49960988432698"""


# ═════════════════════════════════════════════════════════════════════════════
# CONJUNCTION DICT → CDM FEATURES
# Converts your system's internal conjunction dict format to the 103-feature
# CDM format our LightGBM model expects.
# ═════════════════════════════════════════════════════════════════════════════

def conjunction_dict_to_cdm(conj: dict, sat_tle_str: str = None) -> dict:
    """
    Convert the ACAS conjunction dict to CDM feature dict for LightGBM.

    Your conjunction dict has:
        miss_km, tca_hours, rel_pos[3] (km), rel_vel[3] (km/s),
        tle_stale, tle_age_hours, object_type

    LightGBM needs 103 CDM features including:
        miss_distance (m), relative_speed (m/s), time_to_tca (days),
        relative_position_r/t/n (m), t_j2k_sma/ecc/inc,
        t_sigma_r/t/n, c_sigma_r/t/n, F10, F3M, SSN, AP, etc.

    We compute what we can from geometry, estimate the rest.
    """
    import math

    miss_m   = conj['miss_km'] * 1000.0        # km → m
    tca_days = conj['tca_hours'] / 24.0        # hours → days
    rel_pos  = np.array(conj['rel_pos'])        # km
    rel_vel  = np.array(conj['rel_vel'])        # km/s

    rel_speed_ms  = float(np.linalg.norm(rel_vel)) * 1000.0   # km/s → m/s
    rel_pos_m     = rel_pos * 1000.0                           # km → m
    rel_vel_ms    = rel_vel * 1000.0                           # km/s → m/s

    # ── RIC frame decomposition ───────────────────────────────────────────────
    # Approximate: treat rel_pos direction as radial
    if np.linalg.norm(rel_pos) > 1e-10:
        r_hat = rel_pos / np.linalg.norm(rel_pos)
    else:
        r_hat = np.array([1.0, 0.0, 0.0])

    if np.linalg.norm(rel_vel) > 1e-10:
        vel_hat = rel_vel / np.linalg.norm(rel_vel)
    else:
        vel_hat = np.array([0.0, 1.0, 0.0])

    n_hat = np.cross(r_hat, vel_hat)
    if np.linalg.norm(n_hat) < 1e-10:
        n_hat = np.array([0.0, 0.0, 1.0])
    else:
        n_hat /= np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)

    rp_ric = np.array([
        float(np.dot(rel_pos_m, r_hat)),
        float(np.dot(rel_pos_m, t_hat)),
        float(np.dot(rel_pos_m, n_hat)),
    ])
    rv_ric = np.array([
        float(np.dot(rel_vel_ms, r_hat)),
        float(np.dot(rel_vel_ms, t_hat)),
        float(np.dot(rel_vel_ms, n_hat)),
    ])

    # ── Object type → uncertainty ─────────────────────────────────────────────
    obj_type = conj.get('object_type', 'UNKNOWN').upper()
    _SIGMA = {
        'PAYLOAD':     {'r': 10,   't': 400,  'n': 5},
        'ROCKET_BODY': {'r': 50,   't': 2000, 'n': 30},
        'DEBRIS':      {'r': 200,  't': 10000,'n': 100},
        'UNKNOWN':     {'r': 300,  't': 20000,'n': 150},
    }
    sat_sig = _SIGMA['PAYLOAD']
    obj_sig = _SIGMA.get(obj_type, _SIGMA['UNKNOWN'])

    # ── TLE age scaling ───────────────────────────────────────────────────────
    tle_age_h = conj.get('tle_age_hours', 24.0)
    age_scale = max(1.0, np.sqrt(1 + tle_age_h / 72.0))

    c_sr = obj_sig['r'] * age_scale
    c_st = obj_sig['t'] * age_scale
    c_sn = obj_sig['n'] * age_scale

    # ── Approximate orbital parameters from miss geometry ─────────────────────
    # We don't have the full TLE of the object, but can estimate from context
    # Use typical LEO values — the model's most critical features are geometry
    typical_sma  = 6921.0   # km (550km altitude)
    typical_inc  = 97.6     # degrees

    # Combined sigma and Mahalanobis distance
    combined_sigma_r = math.sqrt(sat_sig['r']**2 + c_sr**2)
    mahalanobis = miss_m / (combined_sigma_r + 1e-9)

    geocentric_lat = float(math.degrees(math.asin(
        np.clip(rp_ric[2] / (miss_m + 1e-9), -1, 1)
    )))

    return {
        # Core conjunction
        "time_to_tca":             tca_days,
        "miss_distance":           miss_m,
        "relative_speed":          rel_speed_ms,
        "relative_position_r":     rp_ric[0],
        "relative_position_t":     rp_ric[1],
        "relative_position_n":     rp_ric[2],
        "relative_velocity_r":     rv_ric[0],
        "relative_velocity_t":     rv_ric[1],
        "relative_velocity_n":     rv_ric[2],
        # Target (satellite) — approximate
        "t_j2k_sma":               typical_sma,
        "t_j2k_ecc":               0.0001,
        "t_j2k_inc":               typical_inc,
        "t_h_apo":                 550.0,
        "t_h_per":                 549.0,
        "t_span":                  1.5,
        # Chaser (object)
        "c_j2k_sma":               typical_sma + conj.get('miss_km', 1.0) * 0.5,
        "c_j2k_ecc":               0.001,
        "c_j2k_inc":               typical_inc + np.random.uniform(-15, 15),
        "c_h_apo":                 551.0,
        "c_h_per":                 540.0,
        "c_span":                  2.0,
        "c_object_type":           obj_type,
        # Target uncertainty
        "t_sigma_r":               sat_sig['r'],
        "t_sigma_t":               sat_sig['t'],
        "t_sigma_n":               sat_sig['n'],
        "t_sigma_rdot":            sat_sig['r'] * 0.05,
        "t_sigma_tdot":            sat_sig['t'] * 0.00002,
        "t_sigma_ndot":            sat_sig['n'] * 0.003,
        # Chaser uncertainty
        "c_sigma_r":               c_sr,
        "c_sigma_t":               c_st,
        "c_sigma_n":               c_sn,
        "c_sigma_rdot":            c_sr * 0.05,
        "c_sigma_tdot":            c_st * 0.00002,
        "c_sigma_ndot":            c_sn * 0.003,
        # Covariance off-diagonal (zeros)
        **{col: 0.0 for col in [
            "t_ct_r","t_cn_r","t_cn_t","t_crdot_r","t_crdot_t","t_crdot_n",
            "t_ctdot_r","t_ctdot_t","t_ctdot_n","t_ctdot_rdot",
            "t_cndot_r","t_cndot_t","t_cndot_n","t_cndot_rdot","t_cndot_tdot",
            "c_ct_r","c_cn_r","c_cn_t","c_crdot_r","c_crdot_t","c_crdot_n",
            "c_ctdot_r","c_ctdot_t","c_ctdot_n","c_ctdot_rdot",
            "c_cndot_r","c_cndot_t","c_cndot_n","c_cndot_rdot","c_cndot_tdot",
        ]},
        # Covariance determinants
        "t_position_covariance_det": sat_sig['r']**2 * sat_sig['t']**2 * sat_sig['n']**2,
        "c_position_covariance_det": c_sr**2 * c_st**2 * c_sn**2,
        "mahalanobis_distance":      mahalanobis,
        # Physical properties
        "t_rcs_estimate":            0.5,
        "c_rcs_estimate":            0.1 if 'DEBRIS' in obj_type else 1.0,
        "t_cd_area_over_mass":       0.0154,
        "c_cd_area_over_mass":       0.242,
        "t_cr_area_over_mass":       0.0091,
        "c_cr_area_over_mass":       0.143,
        "t_sedr":                    1.54e-7,
        "c_sedr":                    2.42e-6,
        # Observation quality
        "t_time_lastob_start":       max(0.5, tle_age_h / 24.0),
        "t_time_lastob_end":         0.0,
        "t_recommended_od_span":     7.5,
        "t_actual_od_span":          7.49,
        "t_obs_available":           215,
        "t_obs_used":                214,
        "t_residuals_accepted":      99.4,
        "t_weighted_rms":            1.293,
        "c_time_lastob_start":       max(5.0, tle_age_h * 0.5),
        "c_time_lastob_end":         2.0,
        "c_recommended_od_span":     29.45,
        "c_actual_od_span":          min(29.44, max(5.0, 29.44 - tle_age_h * 0.05)),
        "c_obs_available":           max(5, 18),
        "c_obs_used":                max(4, 17),
        "c_residuals_accepted":      83.3,
        "c_weighted_rms":            4.113,
        # Geometry
        "geocentric_latitude":       geocentric_lat,
        "azimuth":                   0.0,
        "elevation":                 0.0,
        # Space weather (standard values — update if you have live F10.7)
        "F10":                       150.0,
        "F3M":                       148.0,
        "SSN":                       80.0,
        "AP":                        12.0,
        # IDs
        "event_id":                  0,
        "mission_id":                1,
    }


# ═════════════════════════════════════════════════════════════════════════════
# LGBM INFERENCE ENGINE
# Drop-in replacement for OnnxInferenceEngine in acas_controller.py
# ═════════════════════════════════════════════════════════════════════════════

class LGBMInferenceEngine:
    """
    Drop-in replacement for OnnxInferenceEngine.

    Usage in acas_controller.py — change just ONE LINE:

        # Old:
        self.onnx_engine = OnnxInferenceEngine(ONNX_MODEL_PATH)

        # New:
        from lgbm_inference_engine import LGBMInferenceEngine
        self.onnx_engine = LGBMInferenceEngine()

    Then in run_once(), ALSO replace the feature extraction + inference:

        # Old (12-feature path):
        features = extract_features(conj)
        raw_pc   = self.onnx_engine.predict_pc(features)

        # New (103-feature path):
        raw_pc   = self.onnx_engine.predict_pc_from_conjunction(conj)

    The old predict_pc(features) method still works if you pass a conjunction
    dict instead of a numpy array — we detect it automatically.
    """

    def __init__(self, model_dir: str = None):
        self.fallback    = False
        self.regressor   = None
        self.classifier  = None
        self.encoders    = None
        self._model_dir  = model_dir

        self._load_models()

    def _find_model_dir(self) -> str:
        """Search for models in common locations."""
        candidates = [
            self._model_dir,
            os.path.join(_ROOT, "outputs", "models"),
            os.path.join(_ROOT, "satellite_lgbm", "outputs", "models"),
            os.path.join(_ROOT, "trained_models"),
            os.path.join(os.path.expanduser("~"), "satellite_lgbm", "outputs", "models"),
        ]
        for path in candidates:
            if path and os.path.exists(os.path.join(path, "regressor.pkl")):
                return path
        return "/home/karthikeyan/vscode/satellite-acas/outputs/models/"

    def _load_models(self):
        """Load LightGBM regressor + classifier from pickle files."""
        model_dir = self._find_model_dir()

        if model_dir is None:
            logger.warning(
                "LightGBM models not found — using physics fallback.\n"
                "To load the model, ensure outputs/models/regressor.pkl exists.\n"
                "Run: python main.py --data your_data.csv --tune --cv"
            )
            self.fallback = True
            return

        try:
            # Try to load via pipeline modules
            sys.path.insert(0, _ROOT)
            if os.path.exists(os.path.join(_ROOT, "satellite_lgbm")):
                sys.path.insert(0, os.path.join(_ROOT, "satellite_lgbm"))

            from model import SatelliteRiskRegressor, SatelliteRiskClassifier
            from data_pipeline import (impute_missing, clip_outliers,
                                        engineer_features, encode_categoricals)

            self._impute         = impute_missing
            self._clip           = clip_outliers
            self._engineer       = engineer_features
            self._encode         = encode_categoricals
            self._pipeline_ready = True

            reg_path = os.path.join(model_dir, "regressor.pkl")
            clf_path = os.path.join(model_dir, "classifier.pkl")
            enc_path = os.path.join(model_dir, "encoders.pkl")

            self.regressor  = SatelliteRiskRegressor.load(reg_path)
            self.classifier = SatelliteRiskClassifier.load(clf_path)

            if os.path.exists(enc_path):
                with open(enc_path, "rb") as f:
                    self.encoders = pickle.load(f)

            self.fallback = False
            logger.info(
                f"✅ LightGBM models loaded from {model_dir} | "
                f"Reg iter={self.regressor.best_iteration} | "
                f"Clf threshold={self.classifier.optimal_threshold:.3f}"
            )

        except ImportError as e:
            logger.warning(f"Pipeline modules not importable: {e}")
            self._try_raw_pickle_load(model_dir)
        except Exception as e:
            logger.warning(f"Model load failed: {e} — using physics fallback")
            self.fallback = True

    def _try_raw_pickle_load(self, model_dir: str):
        """Fallback: load raw LightGBM objects from pickle."""
        try:
            import lightgbm as lgb

            reg_path = os.path.join(model_dir, "regressor.pkl")
            clf_path = os.path.join(model_dir, "classifier.pkl")

            with open(reg_path, "rb") as f:
                raw_reg = pickle.load(f)
            with open(clf_path, "rb") as f:
                raw_clf = pickle.load(f)

            # Handle both raw Booster and wrapped class
            self.regressor  = getattr(raw_reg, 'model', raw_reg)
            self.classifier = getattr(raw_clf, 'model', raw_clf)
            self._threshold = getattr(raw_clf, 'optimal_threshold', 0.65)
            self._pipeline_ready = False
            self.fallback = False
            logger.info("✅ LightGBM raw models loaded via pickle")
        except Exception as e:
            logger.warning(f"Raw pickle load failed: {e} — using physics fallback")
            self.fallback = True

    def predict_pc_from_conjunction(self, conj: dict) -> float:
        """
        ╔═══════════════════════════════════════════════════════╗
        ║  PREFERRED method — takes full conjunction dict       ║
        ║  Converts to CDM features then runs LightGBM          ║
        ╚═══════════════════════════════════════════════════════╝

        Input : conjunction dict from ConjunctionFinder.find_all()
        Output: raw_pc float in [0.0, 1.0]
        """
        if self.fallback:
            return self._physics_fallback(conj)

        try:
            cdm = conjunction_dict_to_cdm(conj)
            return self._run_lgbm(cdm)
        except Exception as e:
            logger.warning(f"LightGBM inference failed ({e}), using fallback")
            return self._physics_fallback(conj)

    def predict_pc(self, features) -> float:
        """
        Compatibility method — preserves old OnnxInferenceEngine interface.

        Accepts EITHER:
          - numpy array (12-element, old format) → uses geometry-based estimate
          - dict (conjunction dict)              → uses full LightGBM pipeline

        Called from acas_controller.py run_once():
            features = extract_features(conj)
            raw_pc   = self.onnx_engine.predict_pc(features)

        NOTE: For best accuracy, change the call to:
            raw_pc = self.onnx_engine.predict_pc_from_conjunction(conj)
        """
        if isinstance(features, dict):
            # Called with conjunction dict — full pipeline
            return self.predict_pc_from_conjunction(features)

        if isinstance(features, np.ndarray):
            # Called with 12-element array — reconstruct conjunction dict
            # Feature order matches extract_features() in conjunction_net.py:
            # [rp_x, rp_y, rp_z, rv_x, rv_y, rv_z, miss_km, tca_h, speed, angle, stale, danger_t]
            conj_approx = {
                'rel_pos':      features[0:3],
                'rel_vel':      features[3:6],
                'miss_km':      float(features[6]),
                'tca_hours':    float(features[7]),
                'tle_stale':    bool(features[10] > 0.5),
                'tle_age_hours':float(features[10] * 72.0),  # approximate
                'object_type':  'UNKNOWN',
                'object_id':    'UNKNOWN',
            }
            return self.predict_pc_from_conjunction(conj_approx)

        # Fallback
        return self._physics_fallback_from_array(features)

    def _run_lgbm(self, cdm: dict) -> float:
        """Run the LightGBM regressor and return linear Pc."""
        if not _HAS_PANDAS:
            raise ImportError("pandas required — pip install pandas")

        import pandas as pd

        df = pd.DataFrame([cdm])

        if self._pipeline_ready:
            df = self._impute(df)
            df = self._clip(df)
            df = self._engineer(df)
            if self.encoders:
                df, _ = self._encode(df, encoders=self.encoders, fit=False)
            else:
                df, _ = self._encode(df, fit=True)

            # Align features
            feat = self.regressor.feature_names
            for col in feat:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[feat]

            risk_score = float(self.regressor.predict(df)[0])
        else:
            # Raw LightGBM booster
            import lightgbm as lgb
            X = df.values
            risk_score = float(self.regressor.predict(X)[0])

        # Convert log-space risk score → linear probability
        raw_pc = float(10 ** risk_score)
        raw_pc = max(0.0, min(1.0, raw_pc))   # clamp to [0,1]
        return raw_pc

    def _physics_fallback(self, conj: dict) -> float:
        """
        Physics formula fallback — same as original OnnxInferenceEngine.
        Used when LightGBM models are not available.
        """
        miss_km = conj.get('miss_km', 1.0)
        speed   = float(np.linalg.norm(conj.get('rel_vel', [7.8, 0, 0])))
        pc      = min((0.01 / (miss_km + 1e-10))**2 * speed / 7.8, 1.0)
        if conj.get('tle_stale'):
            pc = min(pc * 2.5, 1.0)
        return float(pc)

    def _physics_fallback_from_array(self, features: np.ndarray) -> float:
        """Fallback for 12-element array input."""
        miss_km = float(features[6]) if len(features) > 6 else 1.0
        speed   = float(features[8]) if len(features) > 8 else 7.8
        return min((0.01 / (miss_km + 1e-10))**2 * speed / 7.8, 1.0)

    @property
    def is_loaded(self) -> bool:
        return not self.fallback

    def status(self) -> str:
        if self.fallback:
            return "PHYSICS_FALLBACK (LightGBM models not found)"
        return (
            f"LGBM_ACTIVE | "
            f"Reg iter={getattr(self.regressor, 'best_iteration', '?')} | "
            f"Clf threshold={getattr(self.classifier, 'optimal_threshold', 0.65):.3f}"
        )