"""
core/decision_fsm.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    The autonomy governance gate. An explicit finite state machine that sits
    between "the risk scorer says RED" and "fire the thruster". It encodes the
    decide-and-act policy that used to be inline if/else in the controller, so
    the escalation path (plan → wait for ground → veto/expire → execute →
    verify → recover) is auditable and testable rather than implicit.

CALLED FROM
    core/controller.py     one FSM instance, driven each 60-s loop
    tests/test_decision_fsm.py

CALLS INTO
    core/risk_scorer.py    Alert enum (GREEN/YELLOW/ORANGE/RED)
    core/config_loader.py  decision_gate.* timing parameters
    logging, enum, time    standard library only.

WHAT IT PROVIDES
    DecisionState  Enum: NOMINAL ALERT PLAN AWAIT_GROUND_VETO EXECUTE
                         VERIFY RECOVER
    DecisionFSM    class — inject a time source (callable → float epoch) so
                   tests never sleep. Every transition is logged with a reason.

TRANSITION POLICY
    GREEN            → NOMINAL     (log only)
    YELLOW           → ALERT       (log only)
    ORANGE           → PLAN        (prepare burn, do NOT execute)
    RED              → PLAN, then one of:
        no ground link + autonomous_if_no_link  → EXECUTE  (immediately)
        no ground link + autonomy disabled       → PLAN     (hold for link)
        ground link + dv ≤ max_autonomous_dv_ms → AWAIT_GROUND_VETO
                                    (deadline = now + veto_window_s;
                                     veto() cancels → NOMINAL;
                                     poll() past deadline → EXECUTE)
        ground link + dv >  max_autonomous_dv_ms → PLAN
                                    (hold for explicit ground ack() → EXECUTE)

    Execution lifecycle (driven by the controller after a burn is issued):
        EXECUTE → begin_verify() → VERIFY → finish_verify() → RECOVER
                → recover() → NOMINAL
══════════════════════════════════════════════════════════════════════════════
"""
import time
import logging
from enum import Enum
from typing import Callable, Optional

from core.risk_scorer import Alert

log = logging.getLogger("ACAS.decision")


# ── default timing (overridden by config/thresholds.yaml decision_gate.*) ──────
_DEFAULT_VETO_WINDOW_S        = 300.0
_DEFAULT_AUTONOMOUS_IF_NO_LINK = True
_DEFAULT_MAX_AUTONOMOUS_DV_MS  = 2.0


class DecisionState(Enum):
    NOMINAL           = "NOMINAL"
    ALERT             = "ALERT"
    PLAN              = "PLAN"
    AWAIT_GROUND_VETO = "AWAIT_GROUND_VETO"
    EXECUTE           = "EXECUTE"
    VERIFY            = "VERIFY"
    RECOVER           = "RECOVER"


class DecisionFSM:
    """
    Governance state machine for a single satellite. Stateful across loops:
    an AWAIT_GROUND_VETO entered in one loop is serviced by poll()/veto()/ack()
    in later loops.
    """

    def __init__(self,
                 time_source:           Callable[[], float] = time.time,
                 veto_window_s:         Optional[float] = None,
                 autonomous_if_no_link: Optional[bool]  = None,
                 max_autonomous_dv_ms:  Optional[float] = None):
        self._now = time_source

        # Config with sane fallbacks. Explicit args always win.
        self.veto_window_s = (
            veto_window_s if veto_window_s is not None
            else _cfg("decision_gate.veto_window_s", _DEFAULT_VETO_WINDOW_S))
        self.autonomous_if_no_link = (
            autonomous_if_no_link if autonomous_if_no_link is not None
            else _cfg("decision_gate.autonomous_if_no_link", _DEFAULT_AUTONOMOUS_IF_NO_LINK))
        self.max_autonomous_dv_ms = (
            max_autonomous_dv_ms if max_autonomous_dv_ms is not None
            else _cfg("decision_gate.max_autonomous_dv_ms", _DEFAULT_MAX_AUTONOMOUS_DV_MS))

        self.state         = DecisionState.NOMINAL
        self.veto_deadline: Optional[float] = None
        self.pending:       Optional[dict]  = None  # {alert, dv_ms, object_id}
        self.last_reason    = "initialised"

    # ── internal transition primitive ────────────────────────────────────────
    def _to(self, new_state: DecisionState, reason: str) -> DecisionState:
        old = self.state
        self.state = new_state
        self.last_reason = reason
        log.info(f"[FSM] {old.value} → {new_state.value} : {reason}")
        return new_state

    # ── main entry: feed a fresh assessment ───────────────────────────────────
    def submit(self, alert: Alert, ground_contact: bool, dv_ms: float,
               object_id: str = "") -> DecisionState:
        """
        Process one risk assessment. Returns the resulting DecisionState.
        Reaching EXECUTE means the controller should fire the thruster now;
        AWAIT_GROUND_VETO / PLAN mean hold and service later.
        """
        # Do not stomp a decision that is already mid-execution/verification.
        if self.state in (DecisionState.EXECUTE, DecisionState.VERIFY,
                          DecisionState.RECOVER):
            log.info(f"[FSM] submit ignored — busy in {self.state.value}")
            return self.state

        # GREEN / YELLOW clear or downgrade any pending plan.
        if alert == Alert.GREEN:
            self.pending = None
            self.veto_deadline = None
            return self._to(DecisionState.NOMINAL, "GREEN — no actionable threat")

        if alert == Alert.YELLOW:
            self.pending = None
            self.veto_deadline = None
            return self._to(DecisionState.ALERT,
                            "YELLOW — monitor, alert downlinked (log only)")

        # ORANGE / RED both prepare a burn first.
        self.pending = {"alert": alert, "dv_ms": float(dv_ms),
                        "object_id": object_id}
        self.veto_deadline = None
        self._to(DecisionState.PLAN,
                 f"{alert.value} — maneuver prepared (ΔV={dv_ms:.3f} m/s)")

        if alert == Alert.ORANGE:
            # Prepare only. Never auto-execute an ORANGE.
            return self.state

        # ── RED escalation ────────────────────────────────────────────────────
        if not ground_contact:
            if self.autonomous_if_no_link:
                return self._to(
                    DecisionState.EXECUTE,
                    "RED + no ground link → autonomous execution (black box)")
            return self._to(
                DecisionState.PLAN,
                "RED + no ground link but autonomy disabled → hold for link")

        # ground contact present
        if dv_ms <= self.max_autonomous_dv_ms:
            self.veto_deadline = self._now() + self.veto_window_s
            return self._to(
                DecisionState.AWAIT_GROUND_VETO,
                f"RED + ground link, ΔV={dv_ms:.3f} ≤ {self.max_autonomous_dv_ms} m/s "
                f"→ await ground veto for {self.veto_window_s:.0f}s")

        return self._to(
            DecisionState.PLAN,
            f"RED + ground link, ΔV={dv_ms:.3f} > {self.max_autonomous_dv_ms} m/s "
            f"→ oversize burn, hold for explicit ground ack")

    # ── service an in-flight veto window ──────────────────────────────────────
    def poll(self) -> DecisionState:
        """Advance a pending veto window. If the deadline has passed with no
        veto, escalate to EXECUTE."""
        if self.state == DecisionState.AWAIT_GROUND_VETO and self.veto_deadline is not None:
            remaining = self.veto_deadline - self._now()
            if remaining <= 0.0:
                self.veto_deadline = None
                return self._to(DecisionState.EXECUTE,
                                "veto window expired with no veto → execute")
            log.info(f"[FSM] awaiting veto — {remaining:.0f}s remaining")
        return self.state

    def veto(self) -> DecisionState:
        """Ground cancels a pending autonomous burn."""
        if self.state != DecisionState.AWAIT_GROUND_VETO:
            log.info(f"[FSM] veto ignored — not awaiting veto (state={self.state.value})")
            return self.state
        self.pending = None
        self.veto_deadline = None
        return self._to(DecisionState.NOMINAL, "ground VETO received → burn cancelled")

    def ack(self) -> DecisionState:
        """Ground explicitly approves a held burn (oversize ΔV, or a burn
        awaiting confirmation) → execute now."""
        if self.state in (DecisionState.PLAN, DecisionState.AWAIT_GROUND_VETO):
            self.veto_deadline = None
            return self._to(DecisionState.EXECUTE, "ground ACK received → execute")
        log.info(f"[FSM] ack ignored — nothing awaiting ack (state={self.state.value})")
        return self.state

    # ── execution lifecycle ───────────────────────────────────────────────────
    def begin_verify(self) -> DecisionState:
        if self.state == DecisionState.EXECUTE:
            return self._to(DecisionState.VERIFY,
                            "burn issued → verifying ack and re-checking threat")
        return self.state

    def finish_verify(self, verified: bool, threat_clear: bool = True) -> DecisionState:
        if self.state == DecisionState.VERIFY:
            return self._to(
                DecisionState.RECOVER,
                f"verify complete (ack_ok={verified}, threat_clear={threat_clear}) "
                f"→ recover")
        return self.state

    def recover(self) -> DecisionState:
        if self.state == DecisionState.RECOVER:
            self.pending = None
            self.veto_deadline = None
            return self._to(DecisionState.NOMINAL, "recovery complete → nominal")
        return self.state

    # ── convenience ───────────────────────────────────────────────────────────
    @property
    def should_execute(self) -> bool:
        return self.state == DecisionState.EXECUTE

    def reset(self) -> DecisionState:
        self.pending = None
        self.veto_deadline = None
        return self._to(DecisionState.NOMINAL, "reset")


def _cfg(dotted_key: str, default):
    """Read a decision-gate parameter from config_loader, falling back to the
    default if config is unavailable (flight software must not hard-fail on a
    missing config file)."""
    try:
        from core.config_loader import get
        val = get(dotted_key, default)
        return default if val is None else val
    except Exception:
        return default
