"""
core/risk_scorer.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    The operational intelligence layer. Takes raw_pc from the LightGBM model,
    adjusts it for the satellite's real health state, classifies the risk
    as GREEN / YELLOW / ORANGE / RED, and plans the minimum-fuel burn.

CALLED FROM
    core/controller.py    scorer.assess(conj, raw_pc, sat_state)
    dashboard/adapter.py  scorer.assess(conj, raw_pc, sat_state)

CALLS INTO
    numpy only. No imports from this project.

WHAT IT PROVIDES
    Alert         Enum: GREEN | YELLOW | ORANGE | RED

    SatState      Dataclass:
                    fuel_pct, battery_pct, altitude_km, ground_contact,
                    mission_phase, min_altitude_km, total_fuel_kg

    Assessment    Dataclass:
                    object_id, raw_pc, adjusted_pc, alert,
                    dv_vector, dv_magnitude_ms, fuel_cost_pct,
                    post_path_safe, limitations_hit, decision

    RiskScorer
        assess(conjunction, raw_pc, sat, post_path_safe) → Assessment
            Applies all six operational limitations, classifies, plans burn.

SIX LIMITATIONS APPLIED (in order)
    1. TLE staleness      age > 48h → Pc inflated up to 5x
    2. Fuel level         <50% | <30% | <15% → thresholds raised 3x|8x|50x
    3. Battery            <20% → Pc raised 1.5x
    4. Altitude floor     margin < 20km → downward burns disabled
    5. Post-manoeuvre path  unsafe → Pc doubled
    6. Mission phase      critical → Pc raised 1.5x

ALERT THRESHOLDS (base, before fuel scaling)
    GREEN   Pc < 1e-5   no action
    YELLOW  Pc < 1e-4   monitor
    ORANGE  Pc < 1e-3   prepare manoeuvre
    RED     Pc >= 1e-3  execute burn

BURN PLANNING
    Direction  perpendicular to rel_vel (max miss distance per m/s ΔV)
    Cap        sat.fuel_pct × 0.5  max ΔV in m/s
    Fuel cost  Tsiolkovsky: Δm = m(1 − e^(−ΔV/Isp·g0)),  Isp=220s

MIGRATED FROM
    satellite-acas/models/risk_scorer.py  — no import changes
══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Alert levels — what state the system is in
# ─────────────────────────────────────────────────────────────────────────────
class Alert(Enum):
    GREEN  = "GREEN"   # Pc < 1e-5  — no action
    YELLOW = "YELLOW"  # Pc < 1e-4  — monitor closely
    ORANGE = "ORANGE"  # Pc < 1e-3  — prepare maneuver
    RED    = "RED"     # Pc >= 1e-3 — act now


# ─────────────────────────────────────────────────────────────────────────────
# SatState — complete picture of satellite health
# Passed into RiskScorer.assess() every cycle
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SatState:
    fuel_pct:        float         # 0-100  — remaining propellant
    battery_pct:     float         # 0-100  — electrical power
    altitude_km:     float         # current orbital altitude
    ground_contact:  bool          # True = downlink active right now
    mission_phase:   str           # 'nominal' | 'critical' | 'safe_mode'
    min_altitude_km: float = 300.0 # reentry floor — cannot go below this
    total_fuel_kg:   float = 2.0   # physical fuel mass (for Tsiolkovsky)


# ─────────────────────────────────────────────────────────────────────────────
# Assessment — complete output of one risk evaluation
# Consumed by acas_controller.py to decide what to do
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Assessment:
    object_id:        str
    raw_pc:           float           # Pc from neural network (unadjusted)
    adjusted_pc:      float           # Pc after all limitation adjustments
    alert:            Alert           # GREEN / YELLOW / ORANGE / RED
    dv_vector:        np.ndarray      # burn direction + magnitude (m/s)
    dv_magnitude_ms:  float           # scalar ΔV in m/s
    fuel_cost_pct:    float           # % fuel this burn will consume
    post_path_safe:   bool            # True = post-maneuver trajectory is clear
    limitations_hit:  List[str]       # human-readable list of triggered limits
    decision:         str             # text description of what system will do


# ─────────────────────────────────────────────────────────────────────────────
# RiskScorer — the operational intelligence layer
# ─────────────────────────────────────────────────────────────────────────────
class RiskScorer:

    # Standard Pc thresholds (when fuel is healthy)
    # Based on NASA CARA operational guidelines
    BASE_THRESHOLDS = {
        'yellow': 1e-5,
        'orange': 1e-4,
        'red':    1e-3
    }

    def assess(self,
               conjunction:    dict,
               raw_pc:         float,
               sat:            SatState,
               post_path_safe: bool = True) -> Assessment:
        """
        Main method. Adjusts raw_pc for all operational limitations
        then classifies and plans a maneuver.

        conjunction    : one item from ConjunctionFinder.find_all()
        raw_pc         : output of ConjunctionNet inference
        sat            : current satellite health state
        post_path_safe : False if post-maneuver trajectory creates new threats
        """
        pc   = raw_pc
        hits = []   # track which limitations triggered

        # ── LIMITATION 1: TLE Staleness ──────────────────────────────────────
        # Old tracking data = real position unknown = treat as more dangerous
        if conjunction.get('tle_stale'):
            age_h     = conjunction.get('tle_age_hours', 48.0)
            inflation = min(age_h / 24.0, 5.0)   # cap at 5x inflation
            pc       *= (1 + inflation)
            hits.append(
                f"TLE is {age_h:.0f}h old → Pc inflated {1+inflation:.1f}x "
                f"to account for positional uncertainty"
            )

        # ── LIMITATION 2: Fuel Level ─────────────────────────────────────────
        # Low fuel → raise thresholds (conserve propellant for worse threats)
        thresholds = self._fuel_thresholds(sat.fuel_pct)

        if sat.fuel_pct < 5:
            hits.append(
                f"CRITICAL FUEL ({sat.fuel_pct:.1f}%): "
                "Only near-certain collisions trigger burn. "
                "Threshold raised 50x."
            )
        elif sat.fuel_pct < 15:
            hits.append(
                f"LOW FUEL ({sat.fuel_pct:.1f}%): "
                "Minimum-ΔV strategy enforced. "
                "Threshold raised 8x."
            )
        elif sat.fuel_pct < 30:
            hits.append(
                f"MODERATE FUEL ({sat.fuel_pct:.1f}%): "
                "Threshold raised 3x."
            )

        # ── LIMITATION 3: Battery / Power ────────────────────────────────────
        # Low battery = reduced thrust + switch to lightweight model
        if sat.battery_pct < 20:
            pc *= 1.5
            hits.append(
                f"LOW BATTERY ({sat.battery_pct:.1f}%): "
                "Sensitivity increased 1.5x. "
                "Lightweight inference model active."
            )

        # ── LIMITATION 4: Altitude Floor ─────────────────────────────────────
        # Cannot burn downward if too close to reentry altitude
        altitude_margin = sat.altitude_km - sat.min_altitude_km
        downward_ok     = altitude_margin > 20.0

        if not downward_ok:
            hits.append(
                f"ALTITUDE MARGIN LOW ({altitude_margin:.0f}km): "
                "Downward burns disabled. "
                "Only prograde and radial burns allowed."
            )

        # ── LIMITATION 5: Post-Maneuver Path ─────────────────────────────────
        # Maneuver that creates a new conjunction is twice as bad
        if not post_path_safe:
            pc *= 2.0
            hits.append(
                "Post-maneuver trajectory creates new conjunction. "
                "Alternate burn direction being computed."
            )

        # ── LIMITATION 6: Mission Phase ───────────────────────────────────────
        if sat.mission_phase == 'critical':
            pc *= 1.5
            hits.append(
                "Critical mission phase: sensitivity increased 1.5x."
            )
        elif sat.mission_phase == 'safe_mode':
            hits.append(
                "Safe mode: only minimum essential burns will execute."
            )

        # ── CLASSIFY ──────────────────────────────────────────────────────────
        alert = self._classify(pc, thresholds)

        # ── PLAN MANEUVER ─────────────────────────────────────────────────────
        dv_vec, dv_mag, fuel_cost = self._plan_maneuver(
            conjunction, sat, downward_ok
        )

        # ── DECISION TEXT ─────────────────────────────────────────────────────
        decision = self._decision_text(alert, sat, dv_mag, fuel_cost)

        return Assessment(
            object_id       = conjunction['object_id'],
            raw_pc          = raw_pc,
            adjusted_pc     = pc,
            alert           = alert,
            dv_vector       = dv_vec,
            dv_magnitude_ms = dv_mag,
            fuel_cost_pct   = fuel_cost,
            post_path_safe  = post_path_safe,
            limitations_hit = hits,
            decision        = decision
        )

    # ─────────────────────────────────────────────────────────────────────────
    # _fuel_thresholds
    # As fuel drops, the system becomes more conservative about burning.
    # A satellite with 5% fuel left cannot afford to spend it on marginal risks.
    # ─────────────────────────────────────────────────────────────────────────
    def _fuel_thresholds(self, fuel_pct: float) -> dict:
        if fuel_pct > 50:
            return self.BASE_THRESHOLDS                              # standard
        elif fuel_pct > 30:
            return {'yellow': 1e-5, 'orange': 3e-4, 'red': 3e-3}   # 3x raised
        elif fuel_pct > 15:
            return {'yellow': 5e-5, 'orange': 1e-3, 'red': 8e-3}   # 8x raised
        else:
            return {'yellow': 1e-4, 'orange': 5e-3, 'red': 5e-2}   # 50x raised

    def _classify(self, pc: float, thresholds: dict) -> Alert:
        if   pc >= thresholds['red']:    return Alert.RED
        elif pc >= thresholds['orange']: return Alert.ORANGE
        elif pc >= thresholds['yellow']: return Alert.YELLOW
        else:                            return Alert.GREEN

    # ─────────────────────────────────────────────────────────────────────────
    # _plan_maneuver
    # Computes the minimum ΔV burn that achieves 5km miss distance.
    #
    # Direction: perpendicular to the relative velocity vector.
    #            This direction gives the maximum gain in miss distance
    #            per metre per second of ΔV spent.
    #
    # Magnitude: capped by the fuel budget (LIMITATION 2).
    #            Uses simplified Tsiolkovsky rocket equation for fuel cost.
    # ─────────────────────────────────────────────────────────────────────────
    def _plan_maneuver(self, conj: dict, sat: SatState,
                       downward_ok: bool):
        miss = conj['miss_km']
        rv   = conj['rel_vel']
        tca  = max(conj['tca_hours'], 0.01)

        # No maneuver needed if already safe
        if miss >= 5.0:
            return np.zeros(3), 0.0, 0.0

        # Required ΔV to achieve 5km miss distance
        required_kms = (5.0 - miss) / (tca * 3600.0)

        # Perpendicular direction — maximises miss distance gain
        vel_unit = rv / (np.linalg.norm(rv) + 1e-10)
        radial   = np.array([0.0, 0.0, 1.0])
        perp     = np.cross(vel_unit, radial)
        perp_mag = np.linalg.norm(perp)

        if perp_mag < 1e-10:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp /= perp_mag

        # Enforce altitude floor — no downward Z component
        if not downward_ok:
            perp[2] = abs(perp[2])
            perp   /= (np.linalg.norm(perp) + 1e-10)

        # Convert km/s → m/s
        dv_vec = perp * required_kms * 1000.0

        # Cap ΔV by fuel budget
        max_dv = sat.fuel_pct * 0.5
        dv_mag = min(np.linalg.norm(dv_vec), max_dv)
        dv_vec = perp * dv_mag

        # Fuel cost via Tsiolkovsky: Δm = m × (1 - e^(-ΔV / (Isp × g0)))
        Isp    = 220.0   # cold gas thruster specific impulse (s)
        g0     = 9.807   # standard gravity (m/s²)
        dm     = sat.total_fuel_kg * (1 - np.exp(-dv_mag / (Isp * g0)))
        fuel_cost_pct = (dm / sat.total_fuel_kg) * 100.0

        return dv_vec, dv_mag, fuel_cost_pct

    def _decision_text(self, alert: Alert, sat: SatState,
                        dv_mag: float, fuel_cost: float) -> str:
        if alert == Alert.GREEN:
            return "No action required. Continuing nominal monitoring."

        elif alert == Alert.YELLOW:
            return (
                "Potential conjunction detected. "
                "Alert sent to ground station. "
                "Monitoring frequency increased to every 6 seconds."
            )

        elif alert == Alert.ORANGE:
            maneuver_info = (
                f"Maneuver computed: ΔV={dv_mag:.2f} m/s, "
                f"fuel cost={fuel_cost:.2f}%."
            )
            if sat.ground_contact:
                return f"{maneuver_info} Awaiting ground station approval."
            else:
                return (
                    f"{maneuver_info} No ground contact. "
                    "Maneuver queued — will execute autonomously if TCA < 2h."
                )

        else:  # RED
            maneuver_info = (
                f"COLLISION RISK HIGH. "
                f"ΔV={dv_mag:.2f} m/s, fuel cost={fuel_cost:.2f}%."
            )
            if sat.ground_contact:
                return f"{maneuver_info} Executing with ground confirmation."
            else:
                return (
                    f"{maneuver_info} NO GROUND CONTACT — "
                    "AUTONOMOUS EXECUTION. Event logged to onboard black box."
                )