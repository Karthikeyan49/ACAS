"""
tests/test_model_accuracy.py
══════════════════════════════════════════════════════════════════════════════
Behavioural tests for the LightGBM collision-probability path
(model/lgbm_engine.py) as it is actually driven in flight — a conjunction dict
in, a calibrated raw_pc out.

These lock in the guarantees the engine must uphold:
    • determinism   — identical input → identical Pc (no random features)
    • monotonicity  — smaller miss distance → higher Pc
    • safety floor  — Pc never falls below the physics estimate
    • no dangerous false-negatives — close, fast approaches never read GREEN

Skipped automatically if the ML dependencies are not installed.

    pytest tests/test_model_accuracy.py -v
══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
import pytest

lgbm_engine = pytest.importorskip("model.lgbm_engine")
from core.risk_scorer import RiskScorer, SatState, Alert   # noqa: E402


@pytest.fixture(scope="module")
def engine():
    return lgbm_engine.LGBMInferenceEngine()


def mk(miss_km, speed_kms=14.0, otype="DEBRIS", tca=1.5, stale=False, age=6.0):
    u  = np.array([0.6, -0.5, 0.2]);  u  = u / np.linalg.norm(u)
    vu = np.array([-0.9, 0.4, 0.17]); vu = vu / np.linalg.norm(vu)
    return {
        "object_id":     "X", "object_name": "X", "object_type": otype,
        "miss_km":       miss_km, "tca_hours": tca,
        "rel_pos":       u * miss_km, "rel_vel": vu * speed_kms,
        "tle_stale":     stale, "tle_age_hours": age,
    }


def test_prediction_is_deterministic(engine):
    c = mk(0.15)
    vals = [engine.predict_pc_from_conjunction(c) for _ in range(5)]
    assert max(vals) - min(vals) < 1e-12, f"non-deterministic Pc: {vals}"


def test_pc_monotonic_in_miss_distance(engine):
    pcs = [engine.predict_pc_from_conjunction(mk(m)) for m in (5.0, 1.0, 0.2, 0.05)]
    for closer, farther in zip(pcs[1:], pcs[:-1]):
        assert closer >= farther, f"Pc must rise as miss shrinks: {pcs}"


def test_pc_never_below_physics_floor(engine):
    c = mk(0.2)
    assert engine.predict_pc_from_conjunction(c) >= engine._physics_fallback(c) - 1e-12


def test_pc_bounded_unit_interval(engine):
    for m in (50.0, 5.0, 0.5, 0.01):
        pc = engine.predict_pc_from_conjunction(mk(m))
        assert 0.0 <= pc <= 1.0


@pytest.mark.parametrize("miss,speed", [(0.05, 14.0), (0.2, 14.0), (1.0, 14.9)])
def test_no_dangerous_false_negative(engine, miss, speed):
    """A close, fast debris approach must never be assessed GREEN."""
    scorer = RiskScorer()
    s = SatState(75, 85, 550, True, "nominal", 300.0, 2.0)
    c = mk(miss, speed_kms=speed)
    a = scorer.assess(c, engine.predict_pc_from_conjunction(c), s, True)
    assert a.alert != Alert.GREEN, f"miss={miss}km speed={speed}km/s read GREEN"


def test_far_safe_miss_is_green(engine):
    """A genuinely safe far approach must stay GREEN (no over-firing)."""
    scorer = RiskScorer()
    s = SatState(80, 85, 550, True, "nominal", 300.0, 2.0)
    c = mk(8.0, speed_kms=1.0, otype="PAYLOAD")
    a = scorer.assess(c, engine.predict_pc_from_conjunction(c), s, True)
    assert a.alert == Alert.GREEN
