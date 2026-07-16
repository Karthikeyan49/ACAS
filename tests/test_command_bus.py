"""
tests/test_command_bus.py
══════════════════════════════════════════════════════════════════════════════
Tests for the file command bus that closes the hardware-in-the-loop:
core/controller.py writes a BurnCommand, simulator/orbital.py picks it up on a
tick, applies the ΔV, deducts fuel, and writes a BurnAck.

Drives the simulator's PowerHouseSatellite directly (no 100×-real-time waits):
    • rv2kep round-trip sanity (keplerian_to_eci ↔ eci_to_keplerian)
    • prograde ΔV ⇒ semi-major axis increases, fuel decreases
    • BurnAck written with matching command_id, command file consumed
    • stale command rejected, orbit untouched

The bus file paths are module-level constants in orbital.py, monkeypatched to
tmp_path so the real data_files/ are never touched.

    pytest tests/test_command_bus.py -v
══════════════════════════════════════════════════════════════════════════════
"""
import os
import json
import math
from datetime import datetime, timezone, timedelta

import pytest

import simulator.orbital as orbital
from simulator.orbital import (
    PowerHouseSatellite, keplerian_to_eci, eci_to_keplerian,
)
from core.schemas import BurnCommand, BurnAck, AckStatus


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def bus(tmp_path, monkeypatch):
    """Point the command-bus files at a tmp dir and return the satellite."""
    cmd = tmp_path / "burn_command.json"
    ack = tmp_path / "burn_ack.json"
    rej = tmp_path / "burn_command.rejected.json"
    monkeypatch.setattr(orbital, "BURN_COMMAND_FILE", str(cmd))
    monkeypatch.setattr(orbital, "BURN_ACK_FILE", str(ack))
    monkeypatch.setattr(orbital, "BURN_REJECTED_FILE", str(rej))
    sat = PowerHouseSatellite()
    return sat, cmd, ack, rej


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _current_velocity_unit(sat):
    """Unit velocity vector (ECI) at the satellite's current state."""
    n_deg_s = 360.0 / sat.T
    M_now = (sat.M0 + n_deg_s * sat.sim_time) % 360.0
    eci = keplerian_to_eci(sat.a, sat.e, sat.i, sat.raan, sat.w, M_now)
    v = [eci['vel_x'], eci['vel_y'], eci['vel_z']]
    vmag = math.sqrt(sum(c * c for c in v))
    return [c / vmag for c in v]


def _write_cmd(path, command_id, dv_eci_ms, issued_at_iso, duration_s=200.0):
    cmd = BurnCommand(
        command_id=command_id,
        issued_at_iso=issued_at_iso,
        dv_eci_ms=dv_eci_ms,
        duration_s=duration_s,
        label="TEST",
        alert_level="RED",
    )
    path.write_text(cmd.to_json())


# ── rv2kep round-trip sanity ──────────────────────────────────────────────────
def test_rv2kep_roundtrip_recovers_position():
    a, e, i, raan, w, M = 6921.0, 0.0012, 97.6, 45.0, 90.0, 248.0
    fwd = keplerian_to_eci(a, e, i, raan, w, M)
    el = eci_to_keplerian(fwd['pos_x'], fwd['pos_y'], fwd['pos_z'],
                          fwd['vel_x'], fwd['vel_y'], fwd['vel_z'])
    assert el['a'] == pytest.approx(a, rel=1e-6)
    assert el['e'] == pytest.approx(e, abs=1e-6)
    assert el['i'] == pytest.approx(i, abs=1e-4)
    assert el['raan'] == pytest.approx(raan, abs=1e-4)
    # forward again from recovered elements → same ECI position
    fwd2 = keplerian_to_eci(el['a'], el['e'], el['i'], el['raan'], el['w'], el['M'])
    for k in ('pos_x', 'pos_y', 'pos_z'):
        assert fwd2[k] == pytest.approx(fwd[k], abs=1e-3)


# ── prograde burn raises semi-major axis and burns fuel ───────────────────────
def test_prograde_burn_raises_sma_and_burns_fuel(bus):
    sat, cmd, ack, _ = bus
    a_before = sat.a
    fuel_before = sat.fuel_kg

    v_unit = _current_velocity_unit(sat)
    dv = [1.0 * c for c in v_unit]      # 1 m/s prograde
    _write_cmd(cmd, "cmd-prograde", dv, _now_iso())

    sat.step(orbital.UPDATE_INTERVAL_S * orbital.SIM_SPEED_FACTOR)

    assert sat.a > a_before + 0.5          # prograde ⇒ SMA up by ~1.8 km
    assert sat.fuel_kg < fuel_before       # Tsiolkovsky drain
    assert sat.burn_active is True
    assert sat.burn_timer_s > 0.0
    assert not cmd.exists()                # command consumed
    assert ack.exists()                    # ack written


def test_ack_matches_command_and_reports_dv(bus):
    sat, cmd, ack, _ = bus
    v_unit = _current_velocity_unit(sat)
    dv = [0.8 * c for c in v_unit]
    _write_cmd(cmd, "cmd-abc123", dv, _now_iso())

    sat.step(orbital.UPDATE_INTERVAL_S * orbital.SIM_SPEED_FACTOR)

    parsed = BurnAck.from_json(ack.read_text())
    assert parsed.command_id == "cmd-abc123"
    assert parsed.status == AckStatus.OK
    assert parsed.achieved_dv_ms == pytest.approx(0.8, abs=0.05)
    assert parsed.fuel_used_kg > 0.0


def test_retrograde_burn_lowers_sma(bus):
    sat, cmd, ack, _ = bus
    a_before = sat.a
    v_unit = _current_velocity_unit(sat)
    dv = [-1.0 * c for c in v_unit]     # 1 m/s retrograde
    _write_cmd(cmd, "cmd-retro", dv, _now_iso())

    sat.step(orbital.UPDATE_INTERVAL_S * orbital.SIM_SPEED_FACTOR)
    assert sat.a < a_before - 0.5


# ── stale + malformed guards ──────────────────────────────────────────────────
def test_stale_command_rejected(bus):
    sat, cmd, ack, _ = bus
    a_before = sat.a
    fuel_before = sat.fuel_kg
    old_iso = (datetime.now(timezone.utc) - timedelta(seconds=600)) \
        .strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    v_unit = _current_velocity_unit(sat)
    _write_cmd(cmd, "cmd-stale", [1.0 * c for c in v_unit], old_iso)

    sat.step(orbital.UPDATE_INTERVAL_S * orbital.SIM_SPEED_FACTOR)

    # Orbit untouched by drag-only step (SMA barely moves), no burn applied.
    assert sat.a == pytest.approx(a_before, abs=1e-3)
    assert sat.fuel_kg <= fuel_before          # only passive drain
    assert sat.burn_active is False
    assert not cmd.exists()                    # consumed
    parsed = BurnAck.from_json(ack.read_text())
    assert parsed.command_id == "cmd-stale"
    assert parsed.status == AckStatus.REJECTED_STALE


def test_malformed_command_quarantined(bus):
    sat, cmd, ack, rej = bus
    cmd.write_text("{ this is not valid json ]")

    sat.step(orbital.UPDATE_INTERVAL_S * orbital.SIM_SPEED_FACTOR)

    assert not cmd.exists()                    # moved out of the way
    assert rej.exists()                        # quarantined
    assert sat.burn_active is False
    parsed = BurnAck.from_json(ack.read_text())
    assert parsed.status == AckStatus.REJECTED_MALFORMED


def test_command_applied_only_once(bus):
    sat, cmd, ack, _ = bus
    v_unit = _current_velocity_unit(sat)
    _write_cmd(cmd, "cmd-once", [0.5 * c for c in v_unit], _now_iso())
    dt = orbital.UPDATE_INTERVAL_S * orbital.SIM_SPEED_FACTOR

    sat.step(dt)
    a_after_first = sat.a
    sat.burn_active = False   # let the burn window lapse
    sat.step(dt)              # second tick: nothing to pick up
    # SMA changes only by drag on the second step, not another burn.
    assert sat.a == pytest.approx(a_after_first, abs=1e-2)
