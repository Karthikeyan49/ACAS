"""
tests/test_decision_fsm.py
══════════════════════════════════════════════════════════════════════════════
Unit tests for core/decision_fsm.py — the autonomy governance gate.

Every transition path is exercised with a fake clock (no sleeps):
    GREEN/YELLOW/ORANGE routing, RED autonomous-when-no-link, RED ground-veto
    with cancel, RED ground-veto expiry ⇒ execute, oversize-ΔV hold + ack,
    and the full EXECUTE→VERIFY→RECOVER→NOMINAL lifecycle.

    pytest tests/test_decision_fsm.py -v
══════════════════════════════════════════════════════════════════════════════
"""
import pytest

from core.risk_scorer import Alert
from core.decision_fsm import DecisionFSM, DecisionState


class FakeClock:
    """Deterministic, injectable time source."""
    def __init__(self, t=1_000_000.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_fsm(clock=None, autonomous_if_no_link=True,
             veto_window_s=300.0, max_autonomous_dv_ms=2.0):
    return DecisionFSM(
        time_source=clock or FakeClock(),
        veto_window_s=veto_window_s,
        autonomous_if_no_link=autonomous_if_no_link,
        max_autonomous_dv_ms=max_autonomous_dv_ms,
    )


# ── trivial alert routing ─────────────────────────────────────────────────────
def test_green_goes_nominal():
    fsm = make_fsm()
    assert fsm.submit(Alert.GREEN, ground_contact=True, dv_ms=0.0) == DecisionState.NOMINAL


def test_yellow_goes_alert():
    fsm = make_fsm()
    assert fsm.submit(Alert.YELLOW, ground_contact=True, dv_ms=0.0) == DecisionState.ALERT


def test_orange_plans_but_never_executes():
    fsm = make_fsm()
    # ground contact or not, ORANGE only prepares.
    assert fsm.submit(Alert.ORANGE, ground_contact=True, dv_ms=0.5) == DecisionState.PLAN
    fsm2 = make_fsm()
    assert fsm2.submit(Alert.ORANGE, ground_contact=False, dv_ms=0.5) == DecisionState.PLAN
    assert fsm2.pending is not None


# ── RED, no ground link ───────────────────────────────────────────────────────
def test_red_no_link_executes_autonomously():
    fsm = make_fsm(autonomous_if_no_link=True)
    assert fsm.submit(Alert.RED, ground_contact=False, dv_ms=1.0) == DecisionState.EXECUTE
    assert fsm.should_execute


def test_red_no_link_autonomy_disabled_holds_in_plan():
    fsm = make_fsm(autonomous_if_no_link=False)
    assert fsm.submit(Alert.RED, ground_contact=False, dv_ms=1.0) == DecisionState.PLAN


def test_red_no_link_executes_even_when_dv_oversize():
    # Behavioural contract: RED + no ground contact ⇒ act autonomously,
    # regardless of ΔV budget (the veto/ack budget only gates ground-linked burns).
    fsm = make_fsm(autonomous_if_no_link=True, max_autonomous_dv_ms=2.0)
    assert fsm.submit(Alert.RED, ground_contact=False, dv_ms=9.9) == DecisionState.EXECUTE


# ── RED, ground link, ΔV in budget ⇒ await veto ───────────────────────────────
def test_red_ground_awaits_veto_with_deadline():
    clock = FakeClock()
    fsm = make_fsm(clock=clock, veto_window_s=300.0, max_autonomous_dv_ms=2.0)
    assert fsm.submit(Alert.RED, ground_contact=True, dv_ms=1.5) == DecisionState.AWAIT_GROUND_VETO
    assert fsm.veto_deadline == pytest.approx(clock.t + 300.0)


def test_veto_cancels_pending_burn():
    fsm = make_fsm()
    fsm.submit(Alert.RED, ground_contact=True, dv_ms=1.5)
    assert fsm.veto() == DecisionState.NOMINAL
    assert fsm.pending is None
    assert fsm.veto_deadline is None


def test_poll_before_deadline_keeps_waiting():
    clock = FakeClock()
    fsm = make_fsm(clock=clock, veto_window_s=300.0)
    fsm.submit(Alert.RED, ground_contact=True, dv_ms=1.5)
    clock.advance(299.0)
    assert fsm.poll() == DecisionState.AWAIT_GROUND_VETO


def test_veto_expiry_triggers_execute():
    clock = FakeClock()
    fsm = make_fsm(clock=clock, veto_window_s=300.0)
    fsm.submit(Alert.RED, ground_contact=True, dv_ms=1.5)
    clock.advance(300.1)
    assert fsm.poll() == DecisionState.EXECUTE
    assert fsm.should_execute


def test_veto_after_expiry_is_ignored():
    clock = FakeClock()
    fsm = make_fsm(clock=clock, veto_window_s=300.0)
    fsm.submit(Alert.RED, ground_contact=True, dv_ms=1.5)
    clock.advance(300.1)
    fsm.poll()  # → EXECUTE
    # A late veto can no longer stop the burn.
    assert fsm.veto() == DecisionState.EXECUTE


# ── RED, ground link, ΔV over budget ⇒ hold for explicit ack ──────────────────
def test_red_oversize_dv_holds_for_ground_ack():
    fsm = make_fsm(max_autonomous_dv_ms=2.0)
    assert fsm.submit(Alert.RED, ground_contact=True, dv_ms=5.0) == DecisionState.PLAN
    assert fsm.veto_deadline is None  # not an autonomous veto window
    assert fsm.ack() == DecisionState.EXECUTE


def test_ack_on_await_veto_also_executes():
    fsm = make_fsm(max_autonomous_dv_ms=2.0)
    fsm.submit(Alert.RED, ground_contact=True, dv_ms=1.0)  # AWAIT_GROUND_VETO
    assert fsm.ack() == DecisionState.EXECUTE


# ── execution lifecycle ───────────────────────────────────────────────────────
def test_full_execute_verify_recover_cycle():
    fsm = make_fsm(autonomous_if_no_link=True)
    fsm.submit(Alert.RED, ground_contact=False, dv_ms=1.0)
    assert fsm.state == DecisionState.EXECUTE
    assert fsm.begin_verify() == DecisionState.VERIFY
    assert fsm.finish_verify(verified=True) == DecisionState.RECOVER
    assert fsm.recover() == DecisionState.NOMINAL
    assert fsm.pending is None


def test_submit_ignored_while_busy_executing():
    fsm = make_fsm(autonomous_if_no_link=True)
    fsm.submit(Alert.RED, ground_contact=False, dv_ms=1.0)  # EXECUTE
    # A new assessment mid-burn must not stomp the in-flight decision.
    assert fsm.submit(Alert.GREEN, ground_contact=True, dv_ms=0.0) == DecisionState.EXECUTE


def test_transition_records_reason():
    fsm = make_fsm(autonomous_if_no_link=True)
    fsm.submit(Alert.RED, ground_contact=False, dv_ms=1.0)
    assert "autonomous" in fsm.last_reason.lower()


# ── config-driven defaults (no explicit args) ─────────────────────────────────
def test_defaults_come_from_config():
    fsm = DecisionFSM(time_source=FakeClock())
    # config/thresholds.yaml decision_gate.*
    assert fsm.veto_window_s == 300
    assert fsm.autonomous_if_no_link is True
    assert fsm.max_autonomous_dv_ms == 2.0
