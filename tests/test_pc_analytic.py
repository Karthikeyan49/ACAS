"""
tests/test_pc_analytic.py
══════════════════════════════════════════════════════════════════════════════
Unit tests for core/pc_analytic.py — the Foster 2D analytic collision
probability and the ML-vs-analytic arbiter. Pure numpy + stdlib, no model
loading, deterministic.

    pytest tests/test_pc_analytic.py -v
══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
import pytest

from core.pc_analytic import (
    compute_pc_foster,
    pc_arbiter,
    default_hard_body_radius,
)


# Encounter velocity used throughout (14 km/s, a typical LEO closing speed).
VEL = np.array([14000.0, 0.0, 0.0])


def iso_cov(sigma_m):
    """Isotropic 3x3 position covariance with the given 1-sigma (m)."""
    return np.eye(3) * (sigma_m ** 2)


# ── configuration ─────────────────────────────────────────────────────────────
def test_default_hard_body_radius_from_config():
    r = default_hard_body_radius()
    assert r == pytest.approx(20.0)


# ── known-geometry sanity ─────────────────────────────────────────────────────
def test_huge_miss_is_essentially_zero():
    # 50 km miss with 200 m uncertainty — no chance of collision.
    pc = compute_pc_foster([0.0, 50_000.0, 0.0], VEL, iso_cov(200.0))
    assert pc < 1e-12


def test_dead_center_small_sigma_is_high():
    # Perfect hit, uncertainty (5 m) far smaller than the 20 m hard body:
    # almost all probability mass lies inside the keep-out disk.
    pc = compute_pc_foster([0.0, 0.0, 0.0], VEL, iso_cov(5.0))
    assert pc > 0.9


def test_pc_bounded_unit_interval():
    for miss in (0.0, 20.0, 200.0, 2000.0):
        for sigma in (5.0, 50.0, 500.0):
            pc = compute_pc_foster([0.0, miss, 0.0], VEL, iso_cov(sigma))
            assert 0.0 <= pc <= 1.0


# ── monotonicity ──────────────────────────────────────────────────────────────
def test_monotonic_decreasing_in_miss_distance():
    misses = [0.0, 50.0, 100.0, 300.0, 800.0, 2000.0]
    pcs = [compute_pc_foster([0.0, m, 0.0], VEL, iso_cov(200.0)) for m in misses]
    for closer, farther in zip(pcs[:-1], pcs[1:]):
        assert closer >= farther, f"Pc must fall as miss grows: {pcs}"


def test_larger_uncertainty_spreads_probability():
    # At a fixed offset well outside the hard body, a tighter covariance
    # concentrates mass away from the disk → lower Pc than a broad one.
    off = [0.0, 400.0, 0.0]
    pc_tight = compute_pc_foster(off, VEL, iso_cov(50.0))
    pc_broad = compute_pc_foster(off, VEL, iso_cov(400.0))
    assert pc_broad > pc_tight


# ── determinism ───────────────────────────────────────────────────────────────
def test_deterministic():
    args = ([0.0, 150.0, 0.0], VEL, iso_cov(200.0))
    vals = [compute_pc_foster(*args) for _ in range(5)]
    assert max(vals) - min(vals) < 1e-15


# ── degenerate inputs ─────────────────────────────────────────────────────────
def test_degenerate_covariance_falls_back_gracefully():
    # Singular (all-zero) covariance must not raise or return NaN.
    pc = compute_pc_foster([0.0, 100.0, 0.0], VEL, np.zeros((3, 3)))
    assert 0.0 <= pc <= 1.0
    assert np.isfinite(pc)


def test_zero_relative_velocity_graceful():
    pc = compute_pc_foster([0.0, 100.0, 0.0], [0.0, 0.0, 0.0], iso_cov(200.0))
    assert 0.0 <= pc <= 1.0
    assert np.isfinite(pc)


def test_none_covariance_uses_isotropic_fallback():
    pc = compute_pc_foster([0.0, 100.0, 0.0], VEL, None)
    assert 0.0 <= pc <= 1.0
    assert np.isfinite(pc)


# ── arbiter ───────────────────────────────────────────────────────────────────
def test_arbiter_is_conservative_max():
    r = pc_arbiter(3e-5, 3e-3)
    assert r["pc"] == 3e-3
    assert r["source"] == "analytic"


def test_arbiter_ml_wins_when_higher():
    r = pc_arbiter(1e-2, 1e-6)
    assert r["pc"] == 1e-2
    assert r["source"] == "ml"


def test_arbiter_tie():
    r = pc_arbiter(1e-4, 1e-4)
    assert r["source"] == "tie"
    assert r["pc"] == 1e-4


def test_arbiter_flags_large_disagreement():
    # 3 orders of magnitude apart → flagged.
    assert pc_arbiter(3e-6, 3e-3)["disagreement"] is True
    # within 2 orders → not flagged.
    assert pc_arbiter(2e-4, 4e-4)["disagreement"] is False


def test_arbiter_handles_zero_and_nonfinite():
    r = pc_arbiter(0.0, 0.0)
    assert r["pc"] == 0.0
    assert r["disagreement"] is False
    r2 = pc_arbiter(float("nan"), 1e-4)
    assert r2["pc"] == 1e-4
    assert np.isfinite(r2["pc"])


def test_arbiter_result_shape():
    r = pc_arbiter(1e-5, 1e-4)
    assert set(r) == {"pc", "ml_pc", "analytic_pc", "source", "disagreement"}
