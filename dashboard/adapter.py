"""
dashboard/adapter.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    Normalises dashboard form data into a conjunction dict, runs LightGBM
    prediction, and returns a result dict the dashboard renders directly.

CALLED FROM
    dashboard/app.py   assess_threat_from_dashboard(form_data, sat_state)
                        assess_scenario(SCENARIOS[i])

CALLS INTO
    model/lgbm_engine.py  LGBMInferenceEngine (singleton, created once)
    core/risk_scorer.py   RiskScorer, SatState, Alert
    data_files/satellite_model.json  _read_sat_state_from_json()

WHAT IT PROVIDES
    build_conjunction_from_form(form_data) → dict
        Normalises slider form data (miss_km, tca_h, rp, rv) into
        the standard conjunction dict format.

    assess_threat_from_dashboard(threat_data, sat_state, post_path_safe)
        → { assessment, lgbm, conjunction, sat_state }
        lgbm dict contains:
            engine_status, raw_pc, risk_score, probability_fmt,
            is_lgbm_active, alert_level, alert_colour,
            dv_magnitude_ms, fuel_cost_pct, limitations, decision, thresholds

    assess_scenario(scenario_dict) → same return dict
        Accepts one SCENARIOS entry from dashboard/app.py.

MIGRATED FROM  satellite-acas/dashboard_lgbm_adapter.py
IMPORT CHANGES
    from lgbm_inference_engine import  →  from model.lgbm_engine import
    from models.risk_scorer    import  →  from core.risk_scorer  import
══════════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger("dashboard_adapter")

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.lgbm_engine import LGBMInferenceEngine
from core.risk_scorer  import RiskScorer, SatState, Alert

# Singleton engine — loaded once when dashboard starts
_engine  = None
_scorer  = RiskScorer()

def _get_engine() -> LGBMInferenceEngine:
    global _engine
    if _engine is None:
        _engine = LGBMInferenceEngine()
    return _engine


# ─────────────────────────────────────────────────────────────────────────────
# Threat dict formats the dashboard can send
# ─────────────────────────────────────────────────────────────────────────────
"""
Dashboard manual form produces:
    {
        "object_id":    "CUSTOM-THREAT-001",
        "object_name":  "FENGYUN 1C DEB",
        "object_type":  "DEBRIS",
        "miss_km":      0.6,
        "tca_hours":    2.5,
        "rel_pos":      [0.36, -0.30, 0.12],   # km
        "rel_vel":      [-7.20, 3.50, 1.50],   # km/s
        "tle_stale":    False,
        "tle_age_hours":8.0,
        "norad":        "28682",               # optional
    }

Dashboard TLE injection produces (via tle_processor.py):
    Full CDM feature dict with 103 features

Both are handled automatically below.
"""


def build_conjunction_from_form(form_data: dict) -> dict:
    """
    Normalise dashboard form data into the standard conjunction dict format.
    Handles both the simple slider form and pre-computed TLE data.
    """
    rel_pos = form_data.get('rel_pos') or form_data.get('rp') or [0.0, 0.0, 0.0]
    rel_vel = form_data.get('rel_vel') or form_data.get('rv') or [0.0, 0.0, 0.0]

    if not isinstance(rel_pos, np.ndarray):
        rel_pos = np.array(rel_pos, dtype=float)
    if not isinstance(rel_vel, np.ndarray):
        rel_vel = np.array(rel_vel, dtype=float)

    miss_km = form_data.get('miss_km', float(np.linalg.norm(rel_pos)))
    tca_h   = form_data.get('tca_hours') or form_data.get('tca_h', 1.0)

    return {
        'object_id':     form_data.get('object_id',     'CUSTOM-001'),
        'object_name':   form_data.get('object_name',   'MANUAL THREAT'),
        'object_type':   form_data.get('object_type',   'UNKNOWN'),
        'miss_km':       float(miss_km),
        'tca_hours':     float(tca_h),
        'rel_pos':       rel_pos,
        'rel_vel':       rel_vel,
        'rel_speed_kms': float(np.linalg.norm(rel_vel)),
        'tle_stale':     bool(form_data.get('tle_stale', False)),
        'tle_age_hours': float(form_data.get('tle_age_hours', form_data.get('tle_age', 0.0))),
    }


def assess_threat_from_dashboard(
    threat_data:    dict,
    sat_state:      SatState = None,
    post_path_safe: bool     = True,
) -> dict:
    """
    ╔═══════════════════════════════════════════════════════════╗
    ║  MAIN FUNCTION — called from dashboard/app.py            ║
    ║                                                           ║
    ║  Input : threat dict from dashboard form OR TLE injection ║
    ║  Output: complete assessment + LightGBM details           ║
    ╚═══════════════════════════════════════════════════════════╝

    Returns dict with:
        assessment  — RiskScorer Assessment object (same as before)
        lgbm        — LightGBM-specific detail (risk score, probability etc.)
        conjunction — normalised conjunction dict
        sat_state   — satellite state used
    """
    engine = _get_engine()

    # ── Normalise threat data ─────────────────────────────────────────────────
    conj = build_conjunction_from_form(threat_data)

    # ── Default sat state (from satellite_model.json if available) ────────────
    if sat_state is None:
        sat_state = _read_sat_state_from_json()

    # ── Run LightGBM prediction ───────────────────────────────────────────────
    raw_pc = engine.predict_pc_from_conjunction(conj)

    # ── Risk scorer ───────────────────────────────────────────────────────────
    assessment = _scorer.assess(conj, raw_pc, sat_state, post_path_safe)

    # ── Build LightGBM-specific response for dashboard ────────────────────────
    risk_score = np.log10(max(raw_pc, 1e-30))   # linear Pc → log10 risk score

    lgbm_info = {
        "engine_status":    engine.status(),
        "raw_pc":           raw_pc,
        "risk_score":       round(risk_score, 4),   # log10(Pc)
        "probability_fmt":  f"{raw_pc:.2e}",
        "is_lgbm_active":   engine.is_loaded,
        "alert_level":      assessment.alert.value,
        "alert_colour": {
            Alert.GREEN:  "#00ff88",
            Alert.YELLOW: "#ffd700",
            Alert.ORANGE: "#ff8c00",
            Alert.RED:    "#ff2244",
        }.get(assessment.alert, "#ffffff"),
        "dv_magnitude_ms":  round(assessment.dv_magnitude_ms, 3),
        "fuel_cost_pct":    round(assessment.fuel_cost_pct, 3),
        "limitations":      assessment.limitations_hit,
        "decision":         assessment.decision,
        "thresholds": {
            "CRITICAL":  {"score": -4.0, "prob": "1e-4"},
            "HIGH":      {"score": -5.0, "prob": "1e-5"},
            "ELEVATED":  {"score": -6.0, "prob": "1e-6"},
            "LOW":       {"score": -8.0, "prob": "1e-8"},
            "NOMINAL":   {"score": -99,  "prob": "<1e-8"},
        },
    }

    return {
        "assessment":  assessment,
        "lgbm":        lgbm_info,
        "conjunction": conj,
        "sat_state":   sat_state,
    }


def assess_scenario(scenario: dict) -> dict:
    """
    Assess one of the built-in SCENARIOS from dashboard/app.py.

    Your dashboard already has:
        SCENARIOS = [
            {"name":"SENTINEL-2 DEB","miss_km":3.8,"tca_h":28.0,
             "rp":[2.28,-1.90,0.30],"rv":[-2.10,0.80,0.30], ...},
            ...
        ]

    Call: result = assess_scenario(SCENARIOS[i])
    """
    threat = {
        'object_name':   scenario.get('name', 'UNKNOWN'),
        'object_id':     scenario.get('norad', 'UNKNOWN'),
        'object_type':   'DEBRIS' if 'DEB' in scenario.get('name', '') else 'UNKNOWN',
        'miss_km':       scenario['miss_km'],
        'tca_hours':     scenario.get('tca_h', scenario.get('tca_hours', 1.0)),
        'rel_pos':       scenario.get('rp', scenario.get('rel_pos', [0, 0, 0])),
        'rel_vel':       scenario.get('rv', scenario.get('rel_vel', [0, 0, 0])),
        'tle_stale':     scenario.get('stale', False),
        'tle_age_hours': scenario.get('tle_age', 24.0),
    }
    return assess_threat_from_dashboard(threat)


def _read_sat_state_from_json() -> SatState:
    """Read satellite state from satellite_model.json (live sim data)."""
    try:
        json_path = os.path.join(_ROOT, "satellite_model.json")
        if os.path.exists(json_path):
            import json
            with open(json_path) as f:
                m = json.load(f)
            h  = m.get('health', {})
            c  = m.get('communications', {})
            d  = m.get('derived_position', {})
            ms = m.get('mission', {})
            return SatState(
                fuel_pct       = h.get('fuel_pct', 75.0),
                battery_pct    = h.get('battery_pct', 88.0),
                altitude_km    = d.get('altitude_km', 550.0),
                ground_contact = c.get('ground_contact', True),
                mission_phase  = ms.get('phase', 'nominal'),
                min_altitude_km= 300.0,
                total_fuel_kg  = h.get('fuel_kg_total', 2.0),
            )
    except Exception:
        pass

    return SatState(
        fuel_pct=75.0, battery_pct=88.0, altitude_km=550.0,
        ground_contact=True, mission_phase='nominal',
        min_altitude_km=300.0, total_fuel_kg=2.0
    )


# ─────────────────────────────────────────────────────────────────────────────
# SNIPPET: how to add to dashboard/app.py
# ─────────────────────────────────────────────────────────────────────────────
"""
In dashboard/app.py, find the section where it calls:
    raw_pc     = engine.predict_pc(extract_features(c))
    assessment = scorer.assess(c, raw_pc, sat_state)

Replace with:
    from dashboard_lgbm_adapter import assess_threat_from_dashboard, assess_scenario

    # For manual form threats (custom threat section in sidebar):
    if manual_threat_submitted:
        result     = assess_threat_from_dashboard(custom_threat_dict, sat_state)
        assessment = result["assessment"]
        lgbm       = result["lgbm"]
        # Show lgbm["probability_fmt"] and lgbm["risk_score"] in the dashboard

    # For built-in scenarios:
    result     = assess_scenario(SCENARIOS[selected_idx])
    assessment = result["assessment"]
    lgbm       = result["lgbm"]

    # In the right panel burn output card, add:
    st.markdown(f'''
        <div class="trow">
          <span class="tkey">RISK SCORE</span>
          <span class="tval">{lgbm["risk_score"]:.4f}</span>
        </div>
        <div class="trow">
          <span class="tkey">COLLISION Pc</span>
          <span class="tval">{lgbm["probability_fmt"]}</span>
        </div>
        <div class="trow">
          <span class="tkey">MODEL</span>
          <span class="tval">{"LightGBM" if lgbm["is_lgbm_active"] else "Physics"}</span>
        </div>
    ''', unsafe_allow_html=True)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  Dashboard LightGBM Adapter — Test")
    print("=" * 60)

    # Test with the COSMOS 954 DEB scenario (RED alert in original dashboard)
    red_scenario = {
        "name": "COSMOS 954 DEB",
        "norad": "10440",
        "miss_km": 0.15,
        "tca_h": 1.2,
        "rp": [0.09, -0.075, 0.03],
        "rv": [-13.50, 6.00, 2.50],
        "stale": False,
        "tle_age": 4.0,
    }

    result = assess_scenario(red_scenario)
    a  = result["assessment"]
    lg = result["lgbm"]

    print(f"\n  Scenario   : {red_scenario['name']}")
    print(f"  Alert      : {a.alert.value}")
    print(f"  Raw Pc     : {lg['raw_pc']:.2e}")
    print(f"  Risk Score : {lg['risk_score']:.4f}")
    print(f"  Prob Fmt   : {lg['probability_fmt']}")
    print(f"  ΔV needed  : {a.dv_magnitude_ms:.2f} m/s")
    print(f"  Fuel cost  : {a.fuel_cost_pct:.3f}%")
    print(f"  Decision   : {a.decision}")
    print(f"  Engine     : {lg['engine_status']}")
    print("\n  ✅ Dashboard adapter working")