"""
core/controller.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    The main brain of ACAS. Runs a 60-second autonomous loop onboard the
    satellite. Every cycle it fetches fresh TLE data, scans for threats,
    predicts collision probability, and decides whether to execute a burn.

CALLED FROM
    Terminal:  python core/controller.py
    Entry point in pyproject.toml as "acas-controller"

CALLS INTO
    data/tle_fetcher.py         TLEFetcher, OrbitPropagator
    data/conjunction_finder.py  ConjunctionFinder
    core/risk_scorer.py         RiskScorer, SatState, Alert
    model/lgbm_engine.py        LGBMInferenceEngine
    core/maneuver_planner.py    RLManeuverAgent

READS
    data_files/satellite_model.json  written every second by simulator/orbital.py
    .env                             SPACETRACK_USER, SPACETRACK_PASS

WRITES
    onboard_blackbox.log   every burn event and loop result

WHAT IT PROVIDES
    ACASController              Main class — call .run_forever()
    SatelliteHardwareInterface  Reads SatState from satellite_model.json
    RLManeuverAgent             Loads PPO policy, falls back to geometric burn

LOOP LOGIC (every 60 seconds)
    1.  read SatState from satellite_model.json
    2.  fetch / refresh TLE catalog (every 90 min)
    3.  propagate your satellite 24h forward
    4.  ConjunctionFinder scans catalog → all events with miss_km < 5
    5.  for each event:
            LGBMInferenceEngine.predict_pc_from_conjunction(conj) → raw_pc
            RiskScorer.assess(conj, raw_pc, sat) → Assessment
    6.  take worst Assessment by adjusted_pc
    7.  act:
            GREEN  → no action
            YELLOW → log, notify ground
            ORANGE → queue burn, auto-execute if TCA < 2h and no contact
            RED    → execute burn immediately, log to black box

IMPORT CHANGES FROM ORIGINAL (onboard/acas_controller_lgbm.py)
    from models.risk_scorer    →  from core.risk_scorer
    from lgbm_inference_engine →  from model.lgbm_engine
    LGBM_MODEL_DIR             →  trained_models/lgbm/
══════════════════════════════════════════════════════════════════════════════
"""
"""
acas_controller.py  — PATCHED for LightGBM
═══════════════════════════════════════════
Changes from original (marked with # ← CHANGED):
  1. Import LGBMInferenceEngine instead of using OnnxInferenceEngine
  2. In run_once(): call predict_pc_from_conjunction(conj) instead of
     extract_features(conj) + predict_pc(features)
  3. ONNX_MODEL_PATH constant kept for reference but no longer used

All other logic — RiskScorer, RL agent, hardware interface, loop — is UNCHANGED.
"""

import sys
import os
import time
import json
import logging
import numpy as np
from datetime import datetime, timedelta


from data.tle_fetcher        import TLEFetcher, OrbitPropagator
from data.conjunction_finder import ConjunctionFinder
# ← CHANGED: removed extract_features import (no longer needed in run_once)
from core.risk_scorer     import RiskScorer, SatState, Alert

# ← CHANGED: import LightGBM engine
from model.lgbm_engine     import LGBMInferenceEngine 


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

SPACETRACK_USER   = "your_email@example.com"     # ← set your credentials
SPACETRACK_PASS   = "your_password"

MY_TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00005764  00000-0  10780-3 0  9993"
MY_TLE_LINE2 = "2 25544  51.6416 290.0015 0002627  55.4917 344.9690 15.49960988432698"

CATALOG_LIMIT    = 200
LOOP_INTERVAL_SEC = 60
TLE_REFRESH_MIN  = 90

ONNX_MODEL_PATH  = "trained_models/conjunction_model.onnx"   # kept for reference
RL_MODEL_PATH    = "trained_models/maneuver_policy"

# ← CHANGED: path to LightGBM models
LGBM_MODEL_DIR = os.path.join(                                 # NEW
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trained_models", "lgbm"
)

LOG_FILE = "onboard_blackbox.log"


logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("ACAS")


# ─────────────────────────────────────────────────────────────────────────────
# RLManeuverAgent — UNCHANGED from original
# ─────────────────────────────────────────────────────────────────────────────
class RLManeuverAgent:

    def __init__(self, model_path: str):
        self.model    = None
        self.fallback = False

        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
            log.info(f"✅ RL maneuver agent loaded: {model_path}.zip")

        except Exception:
            log.warning(
                f"⚠️  RL model not found at {model_path}.zip. "
                f"Using geometric maneuver fallback. "
                f"Run 'python models/maneuver_planner.py' to train."
            )
            self.fallback = True

    def predict_burn(self, conjunction: dict, sat: SatState) -> np.ndarray:
        if self.fallback:
            rv       = conjunction['rel_vel']
            vel_unit = rv / (np.linalg.norm(rv) + 1e-10)
            perp     = np.cross(vel_unit, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(perp) < 1e-10:
                perp = np.array([1.0, 0.0, 0.0])
            else:
                perp /= np.linalg.norm(perp)
            miss  = conjunction['miss_km']
            tca   = max(conjunction['tca_hours'], 0.01)
            dv_ms = (5.0 - miss) / (tca * 3.6) * 1000.0
            dv_ms = min(dv_ms, sat.fuel_pct * 0.5)
            return perp * dv_ms

        obs = np.array([
            *conjunction['rel_pos'],
            *conjunction['rel_vel'],
            sat.fuel_pct,
            sat.battery_pct,
            conjunction['tca_hours'],
            sat.altitude_km - sat.min_altitude_km
        ], dtype=np.float32)

        action, _ = self.model.predict(obs, deterministic=True)
        return action


# ─────────────────────────────────────────────────────────────────────────────
# SatelliteHardwareInterface — UNCHANGED from original
# ─────────────────────────────────────────────────────────────────────────────
class SatelliteHardwareInterface:

    def __init__(self):
        self._fuel_pct       = 75.0
        self._battery_pct    = 88.0
        self._altitude_km    = 550.0
        self._ground_contact = True
        self._mission_phase  = 'nominal'

    def read_state(self) -> SatState:
        # Try to read from satellite_model.json (live sim data)
        try:
            model_path = os.path.join(os.path.dirname(...), "data_files", "satellite_model.json") 
            
            if os.path.exists(model_path):
                with open(model_path) as f:
                    m = json.load(f)
                h = m.get('health', {})
                c = m.get('communications', {})
                d = m.get('derived_position', {})
                ms = m.get('mission', {})
                return SatState(
                    fuel_pct       = h.get('fuel_pct', self._fuel_pct),
                    battery_pct    = h.get('battery_pct', self._battery_pct),
                    altitude_km    = d.get('altitude_km', self._altitude_km),
                    ground_contact = c.get('ground_contact', self._ground_contact),
                    mission_phase  = ms.get('phase', self._mission_phase),
                    min_altitude_km= 300.0,
                    total_fuel_kg  = h.get('fuel_kg_total', 2.0),
                )
        except Exception:
            pass

        self._fuel_pct    = max(0, self._fuel_pct - 0.001)
        self._battery_pct = max(20, min(100,
            self._battery_pct + np.random.uniform(-0.5, 0.5)))
        self._ground_contact = np.random.random() > 0.3

        return SatState(
            fuel_pct       = self._fuel_pct,
            battery_pct    = self._battery_pct,
            altitude_km    = self._altitude_km,
            ground_contact = self._ground_contact,
            mission_phase  = self._mission_phase,
            min_altitude_km= 300.0,
            total_fuel_kg  = 2.0
        )

    def execute_burn(self, dv_vector: np.ndarray, label: str = "") -> bool:
        dv_mag = np.linalg.norm(dv_vector)
        fuel_cost = dv_mag * 0.12
        self._fuel_pct    = max(0, self._fuel_pct - fuel_cost)
        self._altitude_km += dv_vector[2] * 0.1

        log.info(
            f"🔥 THRUSTER BURN EXECUTED {label} | "
            f"ΔV = [{dv_vector[0]:.3f}, {dv_vector[1]:.3f}, {dv_vector[2]:.3f}] m/s | "
            f"Magnitude = {dv_mag:.3f} m/s | "
            f"Fuel cost = {fuel_cost:.3f}% | "
            f"Fuel remaining = {self._fuel_pct:.2f}%"
        )
        return True

    def verify_burn(self, expected_dv: np.ndarray) -> bool:
        success = np.random.random() > 0.05
        if not success:
            log.warning("⚠️  THRUSTER ANOMALY: Actual position deviates > 500m from expected")
        return success


# ─────────────────────────────────────────────────────────────────────────────
# ACASController — ONE CHANGE in __init__ and run_once()
# ─────────────────────────────────────────────────────────────────────────────
class ACASController:

    def __init__(self):
        log.info("="*60)
        log.info("ACAS — Autonomous Collision Avoidance System")
        log.info("Initialising all components...")
        log.info("="*60)

        # ← CHANGED: LightGBM engine replaces OnnxInferenceEngine
        self.onnx_engine  = LGBMInferenceEngine(model_dir=LGBM_MODEL_DIR)
        log.info(f"  Inference engine: {self.onnx_engine.status()}")

        self.rl_agent     = RLManeuverAgent(RL_MODEL_PATH)
        self.scorer       = RiskScorer()
        self.finder       = ConjunctionFinder()
        self.hardware     = SatelliteHardwareInterface()

        self.my_propagator = OrbitPropagator(MY_TLE_LINE1, MY_TLE_LINE2)

        self.catalog              = []
        self.last_tle_refresh     = datetime.utcnow() - timedelta(hours=2)
        self.maneuver_log         = []

        self._init_catalog()

        log.info("✅ ACAS fully initialised. Starting autonomous loop.")

    def _init_catalog(self):
        try:
            self.fetcher = TLEFetcher(SPACETRACK_USER, SPACETRACK_PASS)
            raw          = self.fetcher.get_leo_debris(limit=CATALOG_LIMIT)
            self.catalog = self.fetcher.parse_to_propagators(raw)
            self.last_tle_refresh = datetime.utcnow()
            log.info(f"✅ Catalog loaded: {len(self.catalog)} objects")
        except Exception as e:
            log.error(f"❌ Could not load catalog: {e}")
            log.warning("Running with empty catalog — no conjunctions will be detected")
            self.catalog = []

    def _maybe_refresh_catalog(self):
        age_min = (datetime.utcnow() - self.last_tle_refresh).total_seconds() / 60
        if age_min > TLE_REFRESH_MIN:
            log.info(f"🔄 TLE data is {age_min:.0f} min old — refreshing catalog...")
            try:
                self.catalog = self.fetcher.refresh_catalog(self.catalog)
                self.last_tle_refresh = datetime.utcnow()
            except Exception as e:
                log.warning(f"⚠️  TLE refresh failed: {e}. Continuing with existing data.")

    def run_once(self) -> list:
        sat = self.hardware.read_state()
        log.info(
            f"📡 Satellite state | "
            f"Fuel={sat.fuel_pct:.1f}% | Battery={sat.battery_pct:.1f}% | "
            f"Alt={sat.altitude_km:.0f}km | Ground={'YES' if sat.ground_contact else 'NO'}"
        )

        my_traj = self.my_propagator.get_trajectory(hours=24, step_min=1)

        if not self.catalog:
            log.warning("⚠️  Empty catalog — skipping conjunction check")
            return []

        conjunctions = self.finder.find_all(my_traj, self.catalog)

        if not conjunctions:
            log.info("✅ No conjunctions detected. All clear.")
            return []

        log.info(f"⚠️  {len(conjunctions)} conjunction(s) detected within 5km")

        assessments = []

        for conj in conjunctions:
            # ← CHANGED: use predict_pc_from_conjunction instead of
            #   extract_features(conj) → predict_pc(12-array)
            #   This gives LightGBM the full CDM feature set for best accuracy
            raw_pc = self.onnx_engine.predict_pc_from_conjunction(conj)

            post_path_safe = self._check_post_maneuver_path(conj, sat)

            assessment = self.scorer.assess(conj, raw_pc, sat, post_path_safe)
            assessments.append(assessment)

            log.info(
                f"  [{assessment.alert.value}] "
                f"{conj['object_name'][:20]:20s} | "
                f"Miss={conj['miss_km']:.3f}km | "
                f"TCA={conj['tca_hours']:.2f}h | "
                f"Pc(raw)={raw_pc:.2e} | "
                f"Pc(adj)={assessment.adjusted_pc:.2e}"
            )

            if assessment.limitations_hit:
                for lim in assessment.limitations_hit:
                    log.info(f"    ⚡ {lim}")

        assessments.sort(key=lambda a: a.adjusted_pc, reverse=True)
        top = assessments[0]
        self._act(top, sat, conjunctions[0])

        return assessments

    def _act(self, assessment, sat: SatState, conjunction: dict):
        if assessment.alert == Alert.GREEN:
            return

        elif assessment.alert == Alert.YELLOW:
            log.info(
                f"🟡 YELLOW ALERT: {assessment.object_id} | "
                f"Increasing scan frequency. Alert downlinked."
            )

        elif assessment.alert == Alert.ORANGE:
            dv = self.rl_agent.predict_burn(conjunction, sat)
            log.info(
                f"🟠 ORANGE ALERT: {assessment.object_id} | "
                f"Maneuver computed: ΔV={np.linalg.norm(dv):.2f} m/s"
            )
            if sat.ground_contact:
                self._downlink_maneuver_request(assessment, dv)
            else:
                if conjunction['tca_hours'] < 2.0:
                    log.info(
                        f"  TCA={conjunction['tca_hours']:.2f}h < 2h threshold "
                        f"and no ground contact → EXECUTING AUTONOMOUSLY"
                    )
                    self._execute_maneuver(dv, assessment, sat, "ORANGE-AUTO")
                else:
                    log.info(
                        f"  Maneuver queued. TCA={conjunction['tca_hours']:.2f}h. "
                        f"Will auto-execute if contact not restored."
                    )

        else:  # RED
            dv = self.rl_agent.predict_burn(conjunction, sat)
            log.warning(
                f"🔴 RED ALERT: {assessment.object_id} | "
                f"Pc={assessment.adjusted_pc:.2e} | "
                f"TCA={conjunction['tca_hours']:.2f}h | "
                f"ΔV={np.linalg.norm(dv):.2f} m/s"
            )
            if sat.ground_contact:
                log.warning("  Ground contact active. Executing with ground confirmation.")
                self._execute_maneuver(dv, assessment, sat, "RED-GROUND")
            else:
                log.warning(
                    "  NO GROUND CONTACT. "
                    "EXECUTING AUTONOMOUSLY. "
                    "Event logged to black box."
                )
                self._execute_maneuver(dv, assessment, sat, "RED-AUTONOMOUS")

    def _execute_maneuver(self, dv: np.ndarray, assessment,
                           sat: SatState, label: str):
        success = self.hardware.execute_burn(dv, label)
        if success:
            burn_ok = self.hardware.verify_burn(dv)
            if burn_ok:
                log.info("✅ Burn verified. Trajectory updated.")
            else:
                log.error(
                    "❌ BURN ANOMALY: Position mismatch > 500m. "
                    "Flagged for ground investigation on next pass."
                )
            self.maneuver_log.append({
                'time':       datetime.utcnow().isoformat(),
                'object':     assessment.object_id,
                'alert':      assessment.alert.value,
                'label':      label,
                'dv_ms':      float(np.linalg.norm(dv)),
                'fuel_cost':  assessment.fuel_cost_pct,
                'autonomous': 'AUTONOMOUS' in label
            })

    def _check_post_maneuver_path(self, conjunction: dict, sat: SatState) -> bool:
        risk_factor = max(0, (2.0 - conjunction['tca_hours']) / 2.0)
        return np.random.random() > risk_factor * 0.1

    def _downlink_maneuver_request(self, assessment, dv: np.ndarray):
        log.info(
            f"  📤 DOWNLINK REQUEST | "
            f"Object={assessment.object_id} | "
            f"ΔV={np.linalg.norm(dv):.2f} m/s | "
            f"Fuel cost={assessment.fuel_cost_pct:.2f}% | "
            f"Awaiting ground confirmation..."
        )

    def run_forever(self):
        log.info("🚀 ACAS autonomous loop started")
        loop_number = 0

        while True:
            loop_start  = time.time()
            loop_number += 1
            log.info(f"\n{'─'*60}")
            log.info(f"Loop #{loop_number} | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

            try:
                self._maybe_refresh_catalog()
                assessments = self.run_once()
                elapsed = time.time() - loop_start
                log.info(f"Loop completed in {elapsed:.1f}s")

            except KeyboardInterrupt:
                log.info("\n🛑 ACAS shutdown requested by operator")
                self._print_session_summary()
                break

            except Exception as e:
                log.error(f"❌ Loop error: {e}")
                log.info("Continuing — errors are isolated to this cycle")

            elapsed   = time.time() - loop_start
            sleep_for = max(0, LOOP_INTERVAL_SEC - elapsed)
            if sleep_for > 0:
                log.info(f"⏳ Next cycle in {sleep_for:.0f}s")
                time.sleep(sleep_for)

    def _print_session_summary(self):
        log.info("\n" + "="*60)
        log.info("SESSION SUMMARY")
        log.info("="*60)
        log.info(f"Total maneuvers executed: {len(self.maneuver_log)}")
        for m in self.maneuver_log:
            log.info(
                f"  {m['time']} | {m['label']} | "
                f"Object={m['object']} | "
                f"ΔV={m['dv_ms']:.2f} m/s | "
                f"Fuel={m['fuel_cost']:.2f}%"
            )


if __name__ == "__main__":
    controller = ACASController()
    controller.run_forever()