"""
core/schemas.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    Typed interface contracts for the data that crosses ACAS module and
    process boundaries. Plain stdlib dataclasses (NO pydantic — this lives in
    core/ under the flight software constraint: numpy + stdlib only), each with
    an explicit validate() that raises ValueError on bad or missing fields.

    These are the wire formats of the file command bus that closes the
    hardware-in-the-loop between core/controller.py (the brain) and
    simulator/orbital.py (the physics engine):

        controller  --BurnCommand (burn_command.json)-->  simulator
        simulator   --BurnAck     (burn_ack.json)     -->  controller

CALLED FROM
    core/controller.py    builds BurnCommand, parses BurnAck
    tests/test_command_bus.py, tests/test_decision_fsm.py
    (simulator/orbital.py validates the same fields inline — it stays fully
     standalone and does not import this module.)

CALLS INTO
    json, dataclasses, datetime  — standard library only.
    numpy — vector coercion only.

WHAT IT PROVIDES
    ConjunctionEvent   mirrors the dict data/conjunction_finder.py produces
    BurnCommand        command_id, issued_at_iso, dv_eci_ms(3), duration_s,
                       label, alert_level, veto_deadline_iso|None
    BurnAck            command_id, executed_at_iso, achieved_dv_ms,
                       fuel_used_kg, status
    Each dataclass has .validate() → self (raises ValueError on bad input).
    BurnCommand / BurnAck additionally have to_dict/from_dict and
    to_json/from_json helpers for the file bus.
══════════════════════════════════════════════════════════════════════════════
"""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _as_vec3(value, name: str) -> List[float]:
    """Coerce value into a plain list of three finite floats or raise."""
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric 3-vector, got {value!r}")
    if arr.shape[0] != 3:
        raise ValueError(f"{name} must have exactly 3 components, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: {value!r}")
    return [float(x) for x in arr]


def _require_str(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_number(value, name: str, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {v}")
    return v


# ─────────────────────────────────────────────────────────────────────────────
# ConjunctionEvent — mirrors one dict from ConjunctionFinder.find_all()
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ConjunctionEvent:
    object_id:      str
    object_name:    str
    object_type:    str
    miss_km:        float
    tca_hours:      float
    rel_pos:        List[float]           # relative position at TCA (km)
    rel_vel:        List[float]           # relative velocity at TCA (km/s)
    rel_speed_kms:  float = 0.0
    tle_stale:      bool = False
    tle_age_hours:  float = 0.0
    tca_time:       Optional[str] = None  # ISO string (datetime not JSON-safe)

    def validate(self) -> "ConjunctionEvent":
        _require_str(self.object_id, "object_id")
        _require_str(self.object_name, "object_name")
        _require_str(self.object_type, "object_type")
        _require_number(self.miss_km, "miss_km", minimum=0.0)
        _require_number(self.tca_hours, "tca_hours")
        self.rel_pos = _as_vec3(self.rel_pos, "rel_pos")
        self.rel_vel = _as_vec3(self.rel_vel, "rel_vel")
        _require_number(self.rel_speed_kms, "rel_speed_kms", minimum=0.0)
        if not isinstance(self.tle_stale, bool):
            raise ValueError(f"tle_stale must be bool, got {self.tle_stale!r}")
        _require_number(self.tle_age_hours, "tle_age_hours", minimum=0.0)
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "ConjunctionEvent":
        """Build from a raw conjunction dict (rel_pos/rel_vel may be np arrays,
        tca_time may be a datetime)."""
        missing = [k for k in ("object_id", "object_name", "object_type",
                               "miss_km", "tca_hours", "rel_pos", "rel_vel")
                   if k not in d]
        if missing:
            raise ValueError(f"conjunction dict missing fields: {missing}")
        tca_time = d.get("tca_time")
        if tca_time is not None and not isinstance(tca_time, str):
            tca_time = getattr(tca_time, "isoformat", lambda: str(tca_time))()
        rel_vel = d["rel_vel"]
        rel_speed = d.get("rel_speed_kms")
        if rel_speed is None:
            rel_speed = float(np.linalg.norm(np.asarray(rel_vel, dtype=float)))
        return cls(
            object_id     = d["object_id"],
            object_name   = d["object_name"],
            object_type   = d["object_type"],
            miss_km       = float(d["miss_km"]),
            tca_hours     = float(d["tca_hours"]),
            rel_pos       = _as_vec3(d["rel_pos"], "rel_pos"),
            rel_vel       = _as_vec3(rel_vel, "rel_vel"),
            rel_speed_kms = float(rel_speed),
            tle_stale     = bool(d.get("tle_stale", False)),
            tle_age_hours = float(d.get("tle_age_hours", 0.0)),
            tca_time      = tca_time,
        ).validate()


# ─────────────────────────────────────────────────────────────────────────────
# BurnCommand — controller → simulator (data_files/burn_command.json)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BurnCommand:
    command_id:        str
    issued_at_iso:     str
    dv_eci_ms:         List[float]          # commanded ΔV in ECI (m/s), 3-vector
    duration_s:        float
    label:             str = ""
    alert_level:       str = ""
    veto_deadline_iso: Optional[str] = None

    def validate(self) -> "BurnCommand":
        _require_str(self.command_id, "command_id")
        _require_str(self.issued_at_iso, "issued_at_iso")
        self.dv_eci_ms = _as_vec3(self.dv_eci_ms, "dv_eci_ms")
        _require_number(self.duration_s, "duration_s", minimum=0.0)
        if self.duration_s <= 0.0:
            raise ValueError(f"duration_s must be > 0, got {self.duration_s}")
        if not isinstance(self.label, str):
            raise ValueError(f"label must be a string, got {self.label!r}")
        if not isinstance(self.alert_level, str):
            raise ValueError(f"alert_level must be a string, got {self.alert_level!r}")
        if self.veto_deadline_iso is not None:
            _require_str(self.veto_deadline_iso, "veto_deadline_iso")
        return self

    @property
    def magnitude_ms(self) -> float:
        return float(np.linalg.norm(np.asarray(self.dv_eci_ms, dtype=float)))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "BurnCommand":
        try:
            obj = cls(
                command_id        = d["command_id"],
                issued_at_iso     = d["issued_at_iso"],
                dv_eci_ms         = d["dv_eci_ms"],
                duration_s        = d["duration_s"],
                label             = d.get("label", ""),
                alert_level       = d.get("alert_level", ""),
                veto_deadline_iso = d.get("veto_deadline_iso"),
            )
        except KeyError as e:
            raise ValueError(f"BurnCommand missing required field: {e}")
        return obj.validate()

    @classmethod
    def from_json(cls, text: str) -> "BurnCommand":
        return cls.from_dict(json.loads(text))


# ─────────────────────────────────────────────────────────────────────────────
# BurnAck — simulator → controller (data_files/burn_ack.json)
# ─────────────────────────────────────────────────────────────────────────────
class AckStatus:
    OK                  = "OK"
    REJECTED_STALE      = "REJECTED_STALE"
    REJECTED_MALFORMED  = "REJECTED_MALFORMED"
    FAILED              = "FAILED"


@dataclass
class BurnAck:
    command_id:      str
    executed_at_iso: str
    achieved_dv_ms:  float
    fuel_used_kg:    float
    status:          str = AckStatus.OK

    def validate(self) -> "BurnAck":
        _require_str(self.command_id, "command_id")
        _require_str(self.executed_at_iso, "executed_at_iso")
        _require_number(self.achieved_dv_ms, "achieved_dv_ms", minimum=0.0)
        _require_number(self.fuel_used_kg, "fuel_used_kg", minimum=0.0)
        _require_str(self.status, "status")
        return self

    @property
    def ok(self) -> bool:
        return self.status == AckStatus.OK

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "BurnAck":
        try:
            obj = cls(
                command_id      = d["command_id"],
                executed_at_iso = d["executed_at_iso"],
                achieved_dv_ms  = d["achieved_dv_ms"],
                fuel_used_kg    = d["fuel_used_kg"],
                status          = d.get("status", AckStatus.OK),
            )
        except KeyError as e:
            raise ValueError(f"BurnAck missing required field: {e}")
        return obj.validate()

    @classmethod
    def from_json(cls, text: str) -> "BurnAck":
        return cls.from_dict(json.loads(text))
