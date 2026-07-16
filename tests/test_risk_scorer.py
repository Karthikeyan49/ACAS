"""
tests/test_risk_scorer.py
══════════════════════════════════════════════════════════════════════════════
Unit tests for core/risk_scorer.py — every alert threshold, every fuel-scaling
step, every operational limitation, and the manoeuvre-planning geometry.
Deterministic and dependency-light (numpy only) — no model loading needed.

    pytest tests/test_risk_scorer.py -v
══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
import pytest

from core.risk_scorer import RiskScorer, SatState, Alert


# ── fixtures / helpers ────────────────────────────────────────────────────────
@pytest.fixture
def scorer():
    return RiskScorer()


def sat(fuel=85.0, batt=85.0, alt=550.0, gc=True, phase="nominal"):
    return SatState(fuel, batt, alt, gc, phase,
                    min_altitude_km=300.0, total_fuel_kg=2.0)


def conj(miss=0.3, tca=2.0, stale=False, age=6.0):
    return {
        "object_id":     "TEST-OBJ",
        "object_name":   "TEST OBJECT",
        "object_type":   "DEBRIS",
        "miss_km":       miss,
        "tca_hours":     tca,
        "rel_pos":       np.array([0.20, -0.15, 0.08]),
        "rel_vel":       np.array([-11.0, 4.0, 2.0]),
        "tle_stale":     stale,
        "tle_age_hours": age,
    }


def alert_of(scorer, pc, s=None, c=None, pps=True):
    return scorer.assess(c or conj(), pc, s or sat(), pps).alert


# ── alert thresholds (healthy fuel → base thresholds) ─────────────────────────
@pytest.mark.parametrize("pc,expected", [
    (5e-6, Alert.GREEN),
    (1e-5, Alert.YELLOW),   # boundary — inclusive
    (5e-5, Alert.YELLOW),
    (1e-4, Alert.ORANGE),   # boundary
    (5e-4, Alert.ORANGE),
    (1e-3, Alert.RED),      # boundary
    (5e-3, Alert.RED),
])
def test_alert_thresholds(scorer, pc, expected):
    assert alert_of(scorer, pc) == expected


# ── fuel scaling raises thresholds as propellant drops ────────────────────────
def test_fuel_thresholds_table(scorer):
    assert scorer._fuel_thresholds(60) == RiskScorer.BASE_THRESHOLDS          # 1x
    assert scorer._fuel_thresholds(40) == {"yellow": 1e-5, "orange": 3e-4, "red": 3e-3}
    assert scorer._fuel_thresholds(20) == {"yellow": 5e-5, "orange": 1e-3, "red": 8e-3}
    assert scorer._fuel_thresholds(10) == {"yellow": 1e-4, "orange": 5e-3, "red": 5e-2}


def test_low_fuel_deescalates_alert(scorer):
    # A fixed Pc that is ORANGE when healthy must NOT stay ORANGE on low fuel.
    assert alert_of(scorer, 2e-4, sat(fuel=85)) == Alert.ORANGE
    assert alert_of(scorer, 2e-4, sat(fuel=40)) == Alert.YELLOW


# ── the six operational limitations ───────────────────────────────────────────
def test_l1_tle_staleness_inflates_pc(scorer):
    # age 72h → inflation min(72/24, 5) = 3 → Pc × (1 + 3) = ×4
    a = scorer.assess(conj(stale=True, age=72.0), 3e-5, sat(), True)
    assert a.adjusted_pc == pytest.approx(3e-5 * 4.0, rel=1e-6)
    assert a.alert == Alert.ORANGE
    assert any("TLE" in h for h in a.limitations_hit)


def test_l3_low_battery_inflates_pc(scorer):
    a = scorer.assess(conj(), 8e-5, sat(batt=15), True)
    assert a.adjusted_pc == pytest.approx(8e-5 * 1.5, rel=1e-6)
    assert a.alert == Alert.ORANGE


def test_l5_post_path_unsafe_doubles_pc(scorer):
    a = scorer.assess(conj(), 6e-4, sat(), post_path_safe=False)
    assert a.adjusted_pc == pytest.approx(6e-4 * 2.0, rel=1e-6)
    assert a.alert == Alert.RED


def test_l6_critical_phase_inflates_pc(scorer):
    a = scorer.assess(conj(), 8e-4, sat(phase="critical"), True)
    assert a.adjusted_pc == pytest.approx(8e-4 * 1.5, rel=1e-6)
    assert a.alert == Alert.RED


def test_l4_altitude_floor_flagged_near_reentry(scorer):
    a = scorer.assess(conj(), 5e-3, sat(alt=310), True)   # margin 10 km < 20 km
    assert any("ALTITUDE" in h for h in a.limitations_hit)


# ── manoeuvre planning geometry ───────────────────────────────────────────────
def test_no_burn_when_already_safe(scorer):
    a = scorer.assess(conj(miss=10.0), 5e-3, sat(), True)
    assert a.dv_magnitude_ms == 0.0


def test_burn_is_perpendicular_to_relative_velocity(scorer):
    c = conj(miss=0.3)
    a = scorer.assess(c, 5e-3, sat(), True)
    assert a.dv_magnitude_ms > 0.0
    assert abs(float(np.dot(a.dv_vector, c["rel_vel"]))) == pytest.approx(0.0, abs=1e-6)


def test_tsiolkovsky_fuel_cost_reasonable(scorer):
    a = scorer.assess(conj(miss=0.3), 5e-3, sat(), True)
    assert 0.0 < a.fuel_cost_pct < 5.0     # small burn should cost little fuel


# ── decision text ─────────────────────────────────────────────────────────────
def test_green_decision_text(scorer):
    a = scorer.assess(conj(), 5e-6, sat(), True)
    assert "No action" in a.decision


def test_red_autonomous_when_no_ground_contact(scorer):
    a = scorer.assess(conj(), 5e-3, sat(gc=False), True)
    assert a.alert == Alert.RED
    assert "AUTONOMOUS" in a.decision.upper()
