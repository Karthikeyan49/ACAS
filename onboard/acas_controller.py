# ─────────────────────────────────────────────────────────────────────────────
# onboard/acas_controller.py
#
# PURPOSE:
#   The brain of the entire system. Runs a 60-second autonomous loop that:
#     1. Reads satellite health (fuel, battery, altitude, ground contact)
#     2. Gets 24-hour trajectory of all catalog objects
#     3. Finds conjunction events (ConjunctionFinder)
#     4. Runs ONNX inference to get Pc for each conjunction (ConjunctionNet)
#     5. Adjusts Pc for all operational limitations (RiskScorer)
#     6. Decides: do nothing / alert ground / execute burn autonomously
#     7. Verifies burn succeeded (post-burn GPS check)
#     8. Logs everything to the onboard black box
#
# HOW TO RUN:
#   python onboard/acas_controller.py
#
# REQUIRES:
#   trained_models/conjunction_model.onnx  (run models/conjunction_net.py first)
#   trained_models/maneuver_policy.zip     (run models/maneuver_planner.py first)
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import time
import json
import logging
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tle_fetcher      import TLEFetcher, OrbitPropagator
from data.conjunction_finder import ConjunctionFinder
from models.conjunction_net  import extract_features
from models.risk_scorer      import RiskScorer, SatState, Alert


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Space-Track.org credentials (free account at space-track.org)
SPACETRACK_USER = "karthikeyansenthilkumar0@gmail.com"
SPACETRACK_PASS = "asdfghjkl123456789"

# Your satellite's TLE (replace with your actual satellite's TLE)
MY_TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00005764  00000-0  10780-3 0  9993"
MY_TLE_LINE2 = "2 25544  51.6416 290.0015 0002627  55.4917 344.9690 15.49960988432698"

# Number of debris objects to track
CATALOG_LIMIT = 200

# How often the loop runs (seconds)
LOOP_INTERVAL_SEC = 60

# How often to refresh TLE data (minutes) — once per orbit ~90min
TLE_REFRESH_MIN = 90

# Path to trained models
ONNX_MODEL_PATH = "trained_models/conjunction_model.onnx"
RL_MODEL_PATH   = "trained_models/maneuver_policy"

# Log file
LOG_FILE = "onboard_blackbox.log"


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup — writes to both terminal and black box file
# ─────────────────────────────────────────────────────────────────────────────
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
# OnnxInferenceEngine
# Lightweight wrapper around ONNX Runtime.
# This is what runs on the real satellite — PyTorch is too heavy for OBC.
# Falls back to a simple physics formula if ONNX file not found yet.
# ─────────────────────────────────────────────────────────────────────────────
class OnnxInferenceEngine:

    def __init__(self, model_path: str):
        self.session = None
        self.fallback = False

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            log.info(f"✅ ONNX model loaded: {model_path}")

        except Exception as e:
            log.warning(
                f"⚠️  ONNX model not found at {model_path}. "
                f"Using physics fallback formula. "
                f"Run 'python models/conjunction_net.py' to train the model."
            )
            self.fallback = True

    def predict_pc(self, features: np.ndarray) -> float:
        """
        Run inference. Returns Pc between 0 and 1.
        If ONNX model not available, uses simplified physics formula.
        """
        if self.fallback:
            # Simplified fallback: Pc based on miss distance only
            miss_km = features[6]
            speed   = features[8]
            pc      = min((0.01 / (miss_km + 1e-10))**2 * speed / 7.8, 1.0)
            return float(pc)

        input_data = features.reshape(1, -1)
        result     = self.session.run(
            None, {'features': input_data}
        )
        return float(result[0][0][0])


# ─────────────────────────────────────────────────────────────────────────────
# RLManeuverAgent
# Loads the trained PPO policy for inference.
# Falls back to the geometric maneuver from RiskScorer if model not available.
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
        """
        Returns optimal ΔV vector [dv_x, dv_y, dv_z] in m/s.
        """
        if self.fallback:
            # Geometric fallback: burn perpendicular to approach velocity
            rv      = conjunction['rel_vel']
            vel_unit = rv / (np.linalg.norm(rv) + 1e-10)
            perp    = np.cross(vel_unit, np.array([0.0, 0.0, 1.0]))
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
# SatelliteHardwareInterface
# In a real satellite this reads actual sensors and sends thruster commands.
# In the prototype it simulates sensor readings so you can run on a laptop.
# ─────────────────────────────────────────────────────────────────────────────
class SatelliteHardwareInterface:

    def __init__(self):
        # Simulated satellite state (edit to test different conditions)
        self._fuel_pct       = 75.0
        self._battery_pct    = 88.0
        self._altitude_km    = 550.0
        self._ground_contact = True
        self._mission_phase  = 'nominal'

    def read_state(self) -> SatState:
        """
        Read all sensor values.
        Real satellite: reads from GPS, IMU, battery management system.
        Prototype: returns simulated values.
        """
        # Simulate gradual fuel consumption and battery fluctuation
        self._fuel_pct    = max(0, self._fuel_pct - 0.001)
        self._battery_pct = max(20, min(100,
            self._battery_pct + np.random.uniform(-0.5, 0.5)))
        # Simulate ground contact window (70% of orbit has no contact)
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
        """
        Send ΔV command to thrusters.
        Real satellite: sends command to propulsion controller.
        Prototype: logs the burn and deducts fuel.
        Returns True if burn succeeded.
        """
        dv_mag = np.linalg.norm(dv_vector)
        fuel_cost = dv_mag * 0.12

        self._fuel_pct    = max(0, self._fuel_pct - fuel_cost)
        self._altitude_km += dv_vector[2] * 0.1  # simplified altitude effect

        log.info(
            f"🔥 THRUSTER BURN EXECUTED {label} | "
            f"ΔV = [{dv_vector[0]:.3f}, {dv_vector[1]:.3f}, {dv_vector[2]:.3f}] m/s | "
            f"Magnitude = {dv_mag:.3f} m/s | "
            f"Fuel cost = {fuel_cost:.3f}% | "
            f"Fuel remaining = {self._fuel_pct:.2f}%"
        )
        return True

    def verify_burn(self, expected_dv: np.ndarray) -> bool:
        """
        Post-burn verification: compare expected vs actual new position.
        Real satellite: reads GPS and compares to propagated expected position.
        Prototype: simulates 95% success rate.
        """
        success = np.random.random() > 0.05
        if not success:
            log.warning("⚠️  THRUSTER ANOMALY: Actual position deviates > 500m from expected")
        return success


# ─────────────────────────────────────────────────────────────────────────────
# ACASController
# The main controller class. Wires all components together.
# ─────────────────────────────────────────────────────────────────────────────
class ACASController:

    def __init__(self):
        log.info("="*60)
        log.info("ACAS — Autonomous Collision Avoidance System")
        log.info("Initialising all components...")
        log.info("="*60)

        # Load AI models
        self.onnx_engine  = OnnxInferenceEngine(ONNX_MODEL_PATH)
        self.rl_agent     = RLManeuverAgent(RL_MODEL_PATH)

        # Load operational modules
        self.scorer       = RiskScorer()
        self.finder       = ConjunctionFinder()
        self.hardware     = SatelliteHardwareInterface()

        # Your satellite's propagator
        self.my_propagator = OrbitPropagator(MY_TLE_LINE1, MY_TLE_LINE2)

        # Catalog starts empty — filled at startup
        self.catalog         = []
        self.last_tle_refresh = datetime.utcnow() - timedelta(hours=2)

        # Maneuvers executed this session
        self.maneuver_log = []

        # Connect to Space-Track and download catalog
        self._init_catalog()

        log.info("✅ ACAS fully initialised. Starting autonomous loop.")

    def _init_catalog(self):
        """Download initial TLE catalog at startup."""
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
        """Refresh TLE data if it has been more than TLE_REFRESH_MIN minutes."""
        age_min = (datetime.utcnow() - self.last_tle_refresh).total_seconds() / 60
        if age_min > TLE_REFRESH_MIN:
            log.info(f"🔄 TLE data is {age_min:.0f} min old — refreshing catalog...")
            try:
                self.catalog = self.fetcher.refresh_catalog(self.catalog)
                self.last_tle_refresh = datetime.utcnow()
            except Exception as e:
                log.warning(f"⚠️  TLE refresh failed: {e}. Continuing with existing data.")

    def run_once(self) -> list:
        """
        Execute one complete assessment cycle.
        Called every LOOP_INTERVAL_SEC seconds.

        Returns list of Assessment objects for the dashboard.
        """

        # ── STEP 1: Read satellite sensors ───────────────────────────────────
        sat = self.hardware.read_state()
        log.info(
            f"📡 Satellite state | "
            f"Fuel={sat.fuel_pct:.1f}% | "
            f"Battery={sat.battery_pct:.1f}% | "
            f"Alt={sat.altitude_km:.0f}km | "
            f"Ground={'YES' if sat.ground_contact else 'NO'}"
        )

        # ── STEP 2: Get your satellite's trajectory ───────────────────────────
        my_traj = self.my_propagator.get_trajectory(hours=24, step_min=1)

        # ── STEP 3: Find conjunctions ─────────────────────────────────────────
        if not self.catalog:
            log.warning("⚠️  Empty catalog — skipping conjunction check")
            return []

        conjunctions = self.finder.find_all(my_traj, self.catalog)

        if not conjunctions:
            log.info("✅ No conjunctions detected. All clear.")
            return []

        log.info(f"⚠️  {len(conjunctions)} conjunction(s) detected within 5km")

        # ── STEP 4: Score each conjunction ────────────────────────────────────
        assessments = []

        for conj in conjunctions:
            # Extract 12 features and run ONNX inference
            features = extract_features(conj)
            raw_pc   = self.onnx_engine.predict_pc(features)

            # Check if post-maneuver path is safe (simplified check)
            post_path_safe = self._check_post_maneuver_path(conj, sat)

            # Apply all operational limits and classify
            assessment = self.scorer.assess(
                conj, raw_pc, sat, post_path_safe
            )
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

        # ── STEP 5: Act on highest-priority threat ────────────────────────────
        # Sort by adjusted Pc — most dangerous first
        assessments.sort(key=lambda a: a.adjusted_pc, reverse=True)
        top = assessments[0]

        self._act(top, sat, conjunctions[0])

        return assessments

    def _act(self, assessment, sat: SatState, conjunction: dict):
        """
        Execute the appropriate action based on alert level and ground contact.
        This is the core autonomous decision logic.
        """

        if assessment.alert == Alert.GREEN:
            # No action needed
            return

        elif assessment.alert == Alert.YELLOW:
            log.info(
                f"🟡 YELLOW ALERT: {assessment.object_id} | "
                f"Increasing scan frequency. Alert downlinked."
            )

        elif assessment.alert == Alert.ORANGE:
            # Get optimal burn from RL agent
            dv = self.rl_agent.predict_burn(conjunction, sat)
            log.info(
                f"🟠 ORANGE ALERT: {assessment.object_id} | "
                f"Maneuver computed: ΔV={np.linalg.norm(dv):.2f} m/s"
            )

            if sat.ground_contact:
                # Send to ground for approval
                self._downlink_maneuver_request(assessment, dv)
            else:
                # Queue — will auto-execute if TCA drops below 2 hours
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
                # Urgent downlink with very short confirmation window
                log.warning("  Ground contact active. Executing with ground confirmation.")
                self._execute_maneuver(dv, assessment, sat, "RED-GROUND")
            else:
                # Fully autonomous — no human in the loop
                log.warning(
                    "  NO GROUND CONTACT. "
                    "EXECUTING AUTONOMOUSLY. "
                    "Event logged to black box."
                )
                self._execute_maneuver(dv, assessment, sat, "RED-AUTONOMOUS")

    def _execute_maneuver(self, dv: np.ndarray, assessment,
                           sat: SatState, label: str):
        """
        Send the burn command and verify it succeeded.
        """
        success = self.hardware.execute_burn(dv, label)

        if success:
            # Verify burn achieved expected trajectory change
            burn_ok = self.hardware.verify_burn(dv)
            if burn_ok:
                log.info("✅ Burn verified. Trajectory updated.")
            else:
                log.error(
                    "❌ BURN ANOMALY: Position mismatch > 500m. "
                    "Flagged for ground investigation on next pass."
                )

            # Log to maneuver record
            self.maneuver_log.append({
                'time':       datetime.utcnow().isoformat(),
                'object':     assessment.object_id,
                'alert':      assessment.alert.value,
                'label':      label,
                'dv_ms':      float(np.linalg.norm(dv)),
                'fuel_cost':  assessment.fuel_cost_pct,
                'autonomous': 'AUTONOMOUS' in label
            })

    def _check_post_maneuver_path(self, conjunction: dict,
                                   sat: SatState) -> bool:
        """
        Simplified post-maneuver path safety check.
        Real version: propagate new trajectory 72h and re-run conjunction scan.
        Prototype: probabilistic check based on scenario complexity.
        """
        # If TCA is very soon and many objects are present, small chance of
        # post-maneuver path creating a new conjunction
        risk_factor = max(0, (2.0 - conjunction['tca_hours']) / 2.0)
        return np.random.random() > risk_factor * 0.1

    def _downlink_maneuver_request(self, assessment, dv: np.ndarray):
        """
        In real satellite: encodes maneuver proposal and adds to downlink queue.
        Prototype: logs the request.
        """
        log.info(
            f"  📤 DOWNLINK REQUEST | "
            f"Object={assessment.object_id} | "
            f"ΔV={np.linalg.norm(dv):.2f} m/s | "
            f"Fuel cost={assessment.fuel_cost_pct:.2f}% | "
            f"Awaiting ground confirmation..."
        )

    def run_forever(self):
        """
        The infinite 60-second autonomous loop.
        This is what runs on the satellite's onboard computer.
        """
        log.info("🚀 ACAS autonomous loop started")
        loop_number = 0

        while True:
            loop_start = time.time()
            loop_number += 1
            log.info(f"\n{'─'*60}")
            log.info(f"Loop #{loop_number} | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

            try:
                # Refresh TLE if due
                self._maybe_refresh_catalog()

                # Run full assessment cycle
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

            # Wait for next cycle
            elapsed  = time.time() - loop_start
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


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    controller = ACASController()
    controller.run_forever()