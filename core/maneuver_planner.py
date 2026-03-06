"""
core/maneuver_planner.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    Computes the optimal thruster burn (ΔV vector) to resolve a conjunction.
    Primary: PPO reinforcement learning agent. Fallback: geometric formula.

CALLED FROM
    core/controller.py   RLManeuverAgent(RL_PATH).predict_burn(conj, sat)

CALLS INTO
    stable_baselines3   PPO.load(), model.predict()
    gymnasium           ManeuverEnv base class
    numpy
    Nothing from this project.

WHAT IT PROVIDES
    ManeuverEnv (gymnasium.Env)
        Sandbox for RL training. One episode = one conjunction scenario.
        State  (10): rel_pos[3] km, rel_vel[3] km/s, fuel%, battery%,
                     tca_hours, alt_margin_km
        Action (3):  [dv_x, dv_y, dv_z] m/s  range ±15 m/s
        Reward:      +12/km miss  −6/%fuel  −25 if still <5km  −150 if fuel<3%

    train_rl_agent()
        Trains PPO 400,000 steps on 4 parallel envs (~25 min on CPU).
        Saves: trained_models/rl/maneuver_policy.zip
        Run:   python core/maneuver_planner.py

GEOMETRIC FALLBACK
    Burns perpendicular to rel_vel: cross(vel_unit, [0,0,1])
    Same direction as RiskScorer._plan_maneuver.

MIGRATED FROM
    satellite-acas/models/maneuver_planner.py  — no import changes
══════════════════════════════════════════════════════════════════════════════
"""
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class ManeuverEnv(gym.Env):
    """
    Gymnasium environment for collision avoidance maneuver training.

    The agent lives in a simplified orbital mechanics simulation.
    Each episode is one conjunction scenario. The agent learns which
    burns are effective and fuel-efficient across thousands of scenarios.

    STATE (10 numbers the agent sees):
      rel_pos[0,1,2]  — relative position of threat (km)
      rel_vel[0,1,2]  — relative velocity of threat (km/s)
      fuel            — remaining propellant (%)
      battery         — electrical power (%)
      tca_hours       — hours until closest approach
      alt_margin      — km above reentry altitude floor

    ACTION (3 numbers the agent outputs):
      [dv_x, dv_y, dv_z] — thruster burn in each axis (m/s)
      Range: [-15, +15] m/s per axis (before fuel capping)

    REWARD:
      +12 per km of miss distance achieved    → maximize miss
      -6  per % of fuel consumed              → minimize fuel
      -25 per km still below 5km safety       → enforce 5km target
      -150 if fuel drops below 3%             → protect reserves
      -5  if miss distance > 50km             → no over-burns
    """

    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()

        # Action: ΔV in 3 axes, ±15 m/s each
        self.action_space = gym.spaces.Box(
            low=-15.0, high=15.0, shape=(3,), dtype=np.float32
        )

        # State: 10 numbers
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Start a new random conjunction scenario.
        Randomising the scenario parameters forces the agent to learn
        a general policy, not just memorise one situation.
        """
        super().reset(seed=seed)

        self.rel_pos = np.random.uniform(-4.0, 4.0, 3)       # km
        self.rel_vel = np.random.uniform(-2.0, 2.0, 3)       # km/s
        self.fuel    = np.random.uniform(10.0, 100.0)         # %
        self.battery = np.random.uniform(20.0, 100.0)         # %
        self.tca     = np.random.uniform(0.5, 48.0)           # hours
        self.alt_mgn = np.random.uniform(10.0, 400.0)         # km
        self.steps   = 0

        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        return np.array([
            *self.rel_pos,
            *self.rel_vel,
            self.fuel,
            self.battery,
            self.tca,
            self.alt_mgn
        ], dtype=np.float32)

    def step(self, action: np.ndarray):
        """
        Apply a thruster burn and update the simulation state.
        All limitation constraints are enforced here.
        """

        # ── CONSTRAINT 1: Fuel budget ─────────────────────────────────────────
        # Cannot burn more than the fuel budget allows.
        # max_dv scales with remaining fuel — less fuel = less allowed burn.
        max_dv = self.fuel * 0.08
        action = np.clip(action, -max_dv, max_dv)

        # ── CONSTRAINT 2: Battery reduces effective thrust ────────────────────
        # Low battery = less electrical power to thrusters
        if self.battery < 20.0:
            throttle_factor = self.battery / 20.0
            action = action * throttle_factor

        # ── CONSTRAINT 3: Altitude floor — no downward burns ─────────────────
        # If altitude margin is critically low, force Z component upward.
        if self.alt_mgn < 15.0:
            action[2] = abs(action[2])

        # Compute actual ΔV magnitude after all constraints
        dv_mag = np.linalg.norm(action)

        # Apply burn — update velocity and consume fuel + battery
        fuel_cost      = dv_mag * 0.12
        self.rel_vel  += action * 0.001    # convert m/s → km/s effect
        self.fuel     -= fuel_cost
        self.battery  -= fuel_cost * 0.3
        self.tca      -= 0.5               # time advances each step
        self.steps    += 1

        # Predict miss distance at TCA after this burn
        final_miss = np.linalg.norm(
            self.rel_pos + self.rel_vel * self.tca * 3600.0
        )

        # ── REWARD FUNCTION ───────────────────────────────────────────────────
        reward = (
             final_miss * 12.0             # ✅ reward miss distance
            - fuel_cost * 6.0              # ✅ penalise fuel spent
            - max(0, 5.0 - final_miss)*25  # ✅ heavy penalty if still < 5km
            - (self.fuel < 3.0) * 150.0   # ✅ catastrophic penalty for fuel out
            - (final_miss > 50.0) * 5.0   # ✅ penalise wasteful over-burns
        )

        # Episode ends when TCA passes, fuel runs out, or step limit reached
        done = (
            self.tca   <= 0 or
            self.fuel  <= 0 or
            self.steps >= 60
        )

        return self._obs(), reward, done, False, {}


def train_rl_agent():
    """
    Train the PPO agent on the ManeuverEnv.

    PPO (Proximal Policy Optimisation) is chosen because:
    - Stable for continuous action spaces (3D ΔV)
    - Limits policy change per update → prevents catastrophic forgetting
    - Battle-tested for space control problems

    4 parallel environments speed up training by 4x.
    400,000 total steps → approximately 25 minutes on CPU.
    """
    os.makedirs("trained_models", exist_ok=True)

    # 4 parallel training environments
    env = make_vec_env(ManeuverEnv, n_envs=4)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate = 3e-4,   # small learning rate = stable training
        n_steps       = 1024,   # collect 1024 steps before each update
        batch_size    = 64,     # update in mini-batches of 64
        n_epochs      = 10,     # reuse each batch 10 times
        verbose       = 1
    )

    print("⏳ Training RL Maneuver Agent (400,000 steps, ~25 min)...")
    model.learn(total_timesteps=400_000)
    model.save("trained_models/maneuver_policy")

    print("✅ RL agent saved to trained_models/maneuver_policy.zip")
    return model


if __name__ == "__main__":
    train_rl_agent()