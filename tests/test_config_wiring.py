"""
tests/test_config_wiring.py
══════════════════════════════════════════════════════════════════════════════
Proves that core/risk_scorer.py reads its alert bands and fuel-scale multipliers
from config/thresholds.yaml (via core.config_loader) rather than hardcoded
literals: point the loader at a temp YAML with DIFFERENT values, reload, and
assert the RiskScorer classification changes accordingly.

    pytest tests/test_config_wiring.py -v
══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
import pytest

from core import config_loader
from core.risk_scorer import RiskScorer, SatState, Alert


def _sat(fuel=85.0):
    return SatState(fuel, 85.0, 550.0, True, "nominal",
                    min_altitude_km=300.0, total_fuel_kg=2.0)


def _conj():
    return {
        "object_id":     "CFG-OBJ",
        "object_name":   "CFG OBJECT",
        "object_type":   "DEBRIS",
        "miss_km":       0.3,
        "tca_hours":     2.0,
        "rel_pos":       np.array([0.2, -0.15, 0.08]),
        "rel_vel":       np.array([-11.0, 4.0, 2.0]),
        "tle_stale":     False,
        "tle_age_hours": 6.0,
    }


def _write_yaml(path, yellow, orange, red, below_15=50.0):
    path.write_text(
        "alerts:\n"
        f"  yellow: {yellow}\n"
        f"  orange: {orange}\n"
        f"  red:    {red}\n"
        "fuel_threshold_scale:\n"
        "  fuel_above_50pct: 1.0\n"
        "  fuel_above_30pct: 3.0\n"
        "  fuel_above_15pct: 8.0\n"
        f"  fuel_below_15pct: {below_15}\n"
        "hard_body_radius_m: 20.0\n"
    )


@pytest.fixture
def point_config_at(tmp_path, monkeypatch):
    """Return a helper that repoints config_loader at a written temp YAML and
    reloads. The original config path is restored on teardown."""
    original = config_loader._CONFIG_PATH

    def _use(**kwargs):
        p = tmp_path / "thresholds.yaml"
        _write_yaml(p, **kwargs)
        monkeypatch.setattr(config_loader, "_CONFIG_PATH", str(p))
        config_loader.reload()
        return p

    yield _use

    monkeypatch.setattr(config_loader, "_CONFIG_PATH", original)
    config_loader.reload()


# ── alert bands come from YAML ────────────────────────────────────────────────
def test_alert_bands_are_read_from_yaml(point_config_at):
    # Shipped defaults: 5e-4 is ORANGE (>= 1e-4 orange, < 1e-3 red).
    point_config_at(yellow="1.0e-5", orange="1.0e-4", red="1.0e-3")
    assert RiskScorer().assess(_conj(), 5e-4, _sat(), True).alert == Alert.ORANGE

    # Shift every band up 10x: now 5e-4 is only YELLOW (>= 1e-4 yellow,
    # < 1e-3 orange). Same Pc, different classification → config drives it.
    point_config_at(yellow="1.0e-4", orange="1.0e-3", red="1.0e-2")
    assert RiskScorer().assess(_conj(), 5e-4, _sat(), True).alert == Alert.YELLOW


def test_red_band_change_reclassifies(point_config_at):
    # Raise only the RED band far above the Pc: 5e-3 drops from RED to ORANGE.
    point_config_at(yellow="1.0e-5", orange="1.0e-4", red="1.0e-3")
    assert RiskScorer().assess(_conj(), 5e-3, _sat(), True).alert == Alert.RED

    point_config_at(yellow="1.0e-5", orange="1.0e-4", red="1.0e-1")
    assert RiskScorer().assess(_conj(), 5e-3, _sat(), True).alert == Alert.ORANGE


# ── fuel-scale multipliers come from YAML ─────────────────────────────────────
def test_fuel_scale_multiplier_is_read_from_yaml(point_config_at):
    # RED (burn) threshold at critical fuel = base_red x fuel_below_15pct.
    point_config_at(yellow="1.0e-5", orange="1.0e-4", red="1.0e-3", below_15=50.0)
    assert RiskScorer()._fuel_thresholds(10.0)["red"] == pytest.approx(5e-2)

    point_config_at(yellow="1.0e-5", orange="1.0e-4", red="1.0e-3", below_15=100.0)
    assert RiskScorer()._fuel_thresholds(10.0)["red"] == pytest.approx(1e-1)


# ── graceful fallback when a key/file is absent ───────────────────────────────
def test_missing_config_falls_back_to_defaults(tmp_path, monkeypatch):
    empty = tmp_path / "empty.yaml"
    empty.write_text("screening_km: 5.0\n")   # no alerts / fuel_threshold_scale
    monkeypatch.setattr(config_loader, "_CONFIG_PATH", str(empty))
    config_loader.reload()
    try:
        s = RiskScorer()
        assert s.base_thresholds == RiskScorer.BASE_THRESHOLDS
        assert s._fuel_thresholds(10.0)["red"] == pytest.approx(5e-2)
    finally:
        config_loader.reload()
