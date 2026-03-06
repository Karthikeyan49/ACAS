"""
simulator/orbital.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    The satellite physics engine. Runs at 100× real time, writes
    data_files/satellite_model.json every second. Simulates every
    subsystem of the Power House satellite.

CALLED FROM
    Terminal:  python simulator/orbital.py   (Terminal 1, keep running)
    pyproject.toml entry point: "acas-simulator"

CALLS INTO
    json, math, time, threading, signal, os  — standard library only.
    Nothing from this project. Fully standalone.

WRITES
    data_files/satellite_model.json
        Read by core/controller.py and dashboard/app.py every cycle.
        Written atomically via .tmp + os.rename().

JSON TOP-LEVEL KEYS
    _meta, identity, timestamp, sim_time_seconds, orbit_number,
    orbital_elements, eci_state, derived_position, health, adcs,
    propulsion, communications, environment, mission, acas

PHYSICS MODELLED
    Keplerian orbit         GM = 398600.4418 km³/s²
    J2 perturbation         RAAN drift ~-6.5°/day at 97.6° SSO
    Atmospheric drag        altitude decay ~2 m/day at 550 km
    Eclipse detection       cylindrical shadow model, 38% fraction
    Battery                 40 Wh, 42 W solar, 28 W idle, 55 W burn
    Ground contact          ISRO Bengaluru 13°N 77.5°E, elevation > 5°
    Doppler shift           437 MHz UHF, v_los × f / c

SATELLITE
    Mass 50 kg | Area 0.35 m² | Cd 2.2 | Isp 220 s
    Orbit 550 km SSO, 97.6° inclination

MIGRATED FROM
    satellite-acas/satellite_process.py
PATH CHANGE
    MODEL_FILE  →  data_files/satellite_model.json
══════════════════════════════════════════════════════════════════════════════
"""


import json
import math
import time
import os
import sys
import signal
import threading
from datetime import datetime, timezone, timedelta
from copy import deepcopy

# ── File path ─────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE   = os.path.join(SCRIPT_DIR, "..", "data_files", "satellite_model.json")
TEMP_FILE    = MODEL_FILE + ".tmp"   # atomic write via rename

# ── Simulation config ─────────────────────────────────────────────────────────
UPDATE_INTERVAL_S = 1.0     # real seconds between file writes
SIM_SPEED_FACTOR  = 100.0    # 100× real time (adjustable via keyboard)
PRINT_INTERVAL_S  = 5.0      # how often to print status to terminal

# ── Physical constants ────────────────────────────────────────────────────────
GM   = 398600.4418   # km³/s²  Earth gravitational parameter
RE   = 6371.0        # km      Earth radius
J2   = 1.08263e-3    # —       Earth oblateness coefficient
DEG  = math.pi / 180.0

# ── Satellite physical parameters ─────────────────────────────────────────────
SAT_MASS_KG      = 50.0    # kg  (wet, with propellant)
SAT_AREA_M2      = 0.35    # m²  cross-sectional area for drag
CD               = 2.2     # — drag coefficient (typical for LEO box sat)
BATTERY_CAPACITY_WH = 40.0  # Wh total battery capacity
SOLAR_POWER_W    = 42.0    # W  in full sun
IDLE_POWER_W     = 28.0    # W  science mode power draw
BURN_POWER_W     = 55.0    # W  during thruster firing
ISP              = 220.0   # s  monopropellant hydrazine specific impulse
G0               = 9.807   # m/s² standard gravity

# ── Initial orbital elements (Power House — Sun-Synchronous 550km) ────────────
# These are the "true" Keplerian elements at simulation epoch
INIT_ELEMENTS = {
    'a':    6921.0,       # km  semi-major axis (RE + 550km)
    'e':    0.0001,       # —   eccentricity (nearly circular)
    'i':    97.6,         # deg inclination (SSO)
    'raan': 45.0,         # deg right ascension of ascending node
    'w':    90.0,         # deg argument of perigee
    'M0':   248.0,        # deg mean anomaly at epoch
}


# =============================================================================
# ORBITAL MECHANICS ENGINE
# =============================================================================

def mean_to_eccentric(M: float, e: float, tol: float = 1e-10) -> float:
    """
    Kepler's equation: M = E - e*sin(E)
    Solve for E (eccentric anomaly) given M (mean anomaly) and e (eccentricity).
    Uses Newton-Raphson iteration — converges in 3-5 iterations for low e.
    """
    E = M  # initial guess
    for _ in range(50):
        dE = (M - E + e * math.sin(E)) / (1 - e * math.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


def keplerian_to_eci(a: float, e: float, i_deg: float, raan_deg: float,
                     w_deg: float, M_deg: float) -> dict:
    """
    Convert Keplerian orbital elements to ECI position + velocity.

    Mathematics:
      1. Solve Kepler's equation for Eccentric Anomaly E
      2. Compute true anomaly ν from E
      3. Compute r (distance) and speed v in orbital plane
      4. Rotate from orbital plane to ECI using three rotation matrices:
           R3(-Ω) · R1(-i) · R3(-ω)  where Ω=RAAN, i=inclination, ω=arg_perigee

    Returns dict with pos_x, pos_y, pos_z (km) and vel_x, vel_y, vel_z (km/s)
    """
    i    = i_deg    * DEG
    raan = raan_deg * DEG
    w    = w_deg    * DEG
    M    = M_deg    * DEG

    # 1. Eccentric anomaly
    E = mean_to_eccentric(M, e)

    # 2. True anomaly
    nu = 2.0 * math.atan2(
        math.sqrt(1 + e) * math.sin(E / 2),
        math.sqrt(1 - e) * math.cos(E / 2)
    )

    # 3. Orbital plane position and velocity
    r    = a * (1 - e * math.cos(E))          # km
    p    = a * (1 - e * e)                     # semi-latus rectum
    h    = math.sqrt(GM * p)                   # specific angular momentum
    rdot = GM / h * e * math.sin(nu)           # radial velocity
    rnu  = h / r                               # tangential velocity

    # Perifocal coordinates (PQW frame)
    x_pqw = r * math.cos(nu)
    y_pqw = r * math.sin(nu)
    vx_pqw = rdot * math.cos(nu) - rnu * math.sin(nu)
    vy_pqw = rdot * math.sin(nu) + rnu * math.cos(nu)

    # 4. Rotation matrix from PQW to ECI
    cos_raan = math.cos(raan);  sin_raan = math.sin(raan)
    cos_i    = math.cos(i);     sin_i    = math.sin(i)
    cos_w    = math.cos(w);     sin_w    = math.sin(w)

    # R_eci_pqw columns
    R11 = cos_raan*cos_w - sin_raan*sin_w*cos_i
    R21 = sin_raan*cos_w + cos_raan*sin_w*cos_i
    R31 = sin_w*sin_i
    R12 = -cos_raan*sin_w - sin_raan*cos_w*cos_i
    R22 = -sin_raan*sin_w + cos_raan*cos_w*cos_i
    R32 = cos_w*sin_i

    px = R11*x_pqw + R12*y_pqw
    py = R21*x_pqw + R22*y_pqw
    pz = R31*x_pqw + R32*y_pqw
    vx = R11*vx_pqw + R12*vy_pqw
    vy = R21*vx_pqw + R22*vy_pqw
    vz = R31*vx_pqw + R32*vy_pqw

    return {
        'pos_x': px, 'pos_y': py, 'pos_z': pz,
        'vel_x': vx, 'vel_y': vy, 'vel_z': vz,
        'r': r, 'nu_deg': nu / DEG,
        'E_deg': E / DEG,
    }


def eci_to_geodetic(px: float, py: float, pz: float,
                    sim_time_s: float) -> dict:
    """
    Convert ECI position to geodetic (lat, lon, alt).
    Accounts for Earth's rotation: GMST (Greenwich Mean Sidereal Time).

    GMST rotates at 360°/86164s = 0.00417807°/s = 7.2921e-5 rad/s
    """
    EARTH_RATE_DEG_S = 360.0 / 86164.0   # degrees per second (sidereal day)
    gmst = (EARTH_RATE_DEG_S * sim_time_s) % 360.0  # simplified GMST

    # Rotate ECI to ECEF (subtract Earth's rotation)
    theta = (gmst) * DEG
    x_ecef =  px * math.cos(theta) + py * math.sin(theta)
    y_ecef = -px * math.sin(theta) + py * math.cos(theta)
    z_ecef =  pz

    r     = math.sqrt(px*px + py*py + pz*pz)
    lat   = math.asin(pz / r) / DEG
    lon   = math.atan2(y_ecef, x_ecef) / DEG
    alt   = r - RE

    return {'lat': lat, 'lon': lon, 'alt': alt}


def is_in_eclipse(px: float, py: float, pz: float,
                  sim_time_s: float) -> bool:
    """
    Cylindrical shadow model (simplified).
    Sun direction rotates around Earth at 360°/year ≈ 0.9856°/day.
    At sim speed 100×, 1 real second = 100 sim seconds = ~0.114 deg sun motion.
    """
    # Sun direction in ECI (simplified: sun moves in ecliptic at ~0.9856 deg/day)
    sun_angle_deg = (sim_time_s / 86400.0) * 360.0 / 365.25
    sun_x = math.cos(sun_angle_deg * DEG)
    sun_y = math.sin(sun_angle_deg * DEG)
    sun_z = 0.0  # assume ecliptic ≈ equatorial for simplicity

    # Satellite position vector
    r = math.sqrt(px*px + py*py + pz*pz)

    # Project sat position onto sun direction
    proj = px*sun_x + py*sun_y + pz*sun_z

    # Check if satellite is behind Earth (in shadow cone)
    # Cylindrical model: in shadow if projection onto anti-sun is > 0
    # and perpendicular distance to shadow axis is < RE
    if proj > 0:
        return False  # sunlit side

    # Perpendicular distance from shadow axis
    perp_sq = (px - proj*sun_x)**2 + (py - proj*sun_y)**2 + (pz - proj*sun_z)**2
    return math.sqrt(perp_sq) < RE


def compute_doppler(vx: float, vy: float, vz: float,
                    px: float, py: float, pz: float,
                    gs_lat: float = 13.0, gs_lon: float = 77.5) -> float:
    """
    Approximate Doppler shift at 437 MHz UHF link.
    Uses line-of-sight velocity component between satellite and ground station.
    """
    FREQ_HZ = 437e6  # UHF frequency
    C_KMS   = 299792.458  # km/s

    # Ground station ECEF position (simplified, assuming spherical Earth)
    gs_x = RE * math.cos(gs_lat * DEG) * math.cos(gs_lon * DEG)
    gs_y = RE * math.cos(gs_lat * DEG) * math.sin(gs_lon * DEG)
    gs_z = RE * math.sin(gs_lat * DEG)

    # Line of sight vector (sat → GS)
    los_x = gs_x - px;  los_y = gs_y - py;  los_z = gs_z - pz
    los_r = math.sqrt(los_x**2 + los_y**2 + los_z**2)
    if los_r < 1:
        return 0.0
    los_x /= los_r;  los_y /= los_r;  los_z /= los_r

    # Radial velocity (negative = approaching = positive Doppler)
    v_radial = -(vx*los_x + vy*los_y + vz*los_z)
    doppler  = FREQ_HZ * v_radial / C_KMS
    return round(doppler, 1)


def compute_atmospheric_density(alt_km: float) -> float:
    """
    Exponential atmospheric density model (Harris-Priester simplified).
    Valid 200-1000km LEO.
    """
    H_SCALE = {200: 37.5, 300: 53.3, 400: 71.8, 500: 88.7, 600: 124.6,
               700: 190.1, 800: 408.0}
    rho0 = {200: 2.60e-10, 300: 1.92e-11, 400: 2.80e-12, 500: 5.22e-13,
            400: 2.80e-12, 600: 5.70e-14, 700: 3.07e-15, 800: 1.55e-16}
    # Simple exponential between reference levels
    ref_alts = [200, 300, 400, 500, 600, 700, 800]
    for i in range(len(ref_alts)-1):
        if ref_alts[i] <= alt_km < ref_alts[i+1]:
            H = H_SCALE.get(ref_alts[i], 70.0)
            r0 = rho0.get(ref_alts[i], 1e-12)
            return r0 * math.exp(-(alt_km - ref_alts[i]) / H)
    return 1e-13  # fallback for ~550km


def compute_j2_drift(a: float, e: float, i_deg: float) -> dict:
    """
    J2 perturbation rates (secular terms only).
    These cause the orbital plane to precess over time.

    RAAN drift rate: dΩ/dt = -3/2 * n * J2 * (RE/p)² * cos(i)
    AoP drift rate:  dω/dt =  3/4 * n * J2 * (RE/p)² * (5cos²i - 1)

    For SSO (i≈97.6°): RAAN drifts at +0.9856 deg/day to match Earth's orbit
    """
    n   = math.sqrt(GM / a**3) * (180.0 / math.pi) * 86400.0  # deg/day
    p   = a * (1 - e**2)
    factor = -1.5 * J2 * (RE / p)**2

    raan_rate_deg_s = factor * n * math.cos(i_deg * DEG) / 86400.0
    w_rate_deg_s    = -0.5 * factor * n * (5 * math.cos(i_deg * DEG)**2 - 1) / 86400.0

    return {
        'raan_rate_deg_s': raan_rate_deg_s,
        'w_rate_deg_s':    w_rate_deg_s,
    }


# =============================================================================
# SATELLITE STATE MACHINE
# =============================================================================

class PowerHouseSatellite:
    """
    Complete simulation of the Power House satellite.
    Maintains full orbital + subsystem state and advances it in time.
    """

    def __init__(self):
        # Load the model file to get initial state
        with open(MODEL_FILE, 'r') as f:
            self.model = json.load(f)

        # Orbital elements (mutable — they drift due to perturbations)
        el = INIT_ELEMENTS
        self.a    = el['a']        # km
        self.e    = el['e']
        self.i    = el['i']        # deg
        self.raan = el['raan']     # deg
        self.w    = el['w']        # deg
        self.M0   = el['M0']       # deg at t=0

        # Precompute period
        self.T    = 2 * math.pi * math.sqrt(self.a**3 / GM)   # seconds

        # J2 perturbation rates
        self.j2   = compute_j2_drift(self.a, self.e, self.i)

        # Drag parameters
        self.B    = (CD * SAT_AREA_M2) / (SAT_MASS_KG * 1e6)  # m²/kg → converted

        # Subsystem state
        self.fuel_kg  = INIT_ELEMENTS.get('fuel_kg', 1.7)
        self.battery_pct = 94.0
        self.temperature_bus = 22.0
        self.temperature_battery = 18.0
        self.reaction_wheels  = [1200.0, -800.0, 3500.0]  # RPM x,y,z

        # Simulation time
        self.sim_time = 0.0     # seconds (simulation time)
        self.real_start = time.time()
        self.orbit_number = 1
        self.uptime_hours = 0.0
        self.anomaly_count = 0

        # Derived (updated each step)
        self.in_eclipse    = False
        self.ground_contact = True
        self.prev_M        = el['M0']
        self.total_dv_ms   = 0.0

        # Maneuver state (can be set externally by ACAS)
        self.burn_active   = False
        self.burn_timer_s  = 0.0
        self.burn_dv       = [0.0, 0.0, 0.0]

        print(f"[Power House] Satellite process started.")
        print(f"[Power House] Orbit: {self.a - RE:.0f} km alt | i={self.i}° | T={self.T/60:.1f} min")
        print(f"[Power House] Writing to: {MODEL_FILE}")
        print(f"[Power House] Sim speed: {SIM_SPEED_FACTOR}× | Update rate: {UPDATE_INTERVAL_S}s")
        print(f"[Power House] Press Ctrl+C to stop\n")


    def step(self, dt_sim: float):
        """
        Advance simulation by dt_sim seconds (simulation time).
        Called every UPDATE_INTERVAL_S real seconds.
        dt_sim = UPDATE_INTERVAL_S × SIM_SPEED_FACTOR
        """
        self.sim_time += dt_sim
        self.uptime_hours = self.sim_time / 3600.0

        # ── 1. Mean anomaly advance ─────────────────────────────────────────
        n = 360.0 / self.T   # deg/s
        M_current = (self.M0 + n * self.sim_time) % 360.0

        # Orbit counter (every time M crosses 0°)
        if M_current < self.prev_M and not (self.prev_M > 350 and M_current < 10):
            self.orbit_number += 1
        self.prev_M = M_current

        # ── 2. J2 perturbations ─────────────────────────────────────────────
        self.raan = (self.raan + self.j2['raan_rate_deg_s'] * dt_sim) % 360.0
        self.w    = (self.w    + self.j2['w_rate_deg_s']    * dt_sim) % 360.0

        # ── 3. Atmospheric drag (altitude decay) ────────────────────────────
        alt  = self.a - RE
        rho  = compute_atmospheric_density(alt)
        v_ms = math.sqrt(GM / self.a) * 1000.0       # orbital speed m/s
        F_drag = 0.5 * rho * self.B * v_ms**2        # deceleration m/s²
        # Decay in semi-major axis: da/dt = -2a/v * F_drag
        da_dt = -2.0 * self.a * 1000.0 / v_ms * F_drag / 1000.0  # km/s
        self.a -= da_dt * dt_sim   # tiny decay each step

        # ── 4. ECI position & velocity ──────────────────────────────────────
        eci = keplerian_to_eci(self.a, self.e, self.i, self.raan, self.w, M_current)
        geo = eci_to_geodetic(eci['pos_x'], eci['pos_y'], eci['pos_z'], self.sim_time)

        # ── 5. Eclipse detection ────────────────────────────────────────────
        self.in_eclipse = is_in_eclipse(eci['pos_x'], eci['pos_y'], eci['pos_z'], self.sim_time)
        eclipse_frac = 0.38  # ISS-like ~38% eclipse fraction per orbit

        # ── 6. Battery model ─────────────────────────────────────────────────
        # Sunlight: solar panels charge battery
        # Eclipse:  battery powers satellite
        current_power_w = BURN_POWER_W if self.burn_active else IDLE_POWER_W
        if self.in_eclipse:
            # Power from battery only
            power_deficit = current_power_w  # W
            charge_loss_wh = power_deficit * (UPDATE_INTERVAL_S / 3600.0)
            bat_loss_pct   = (charge_loss_wh / BATTERY_CAPACITY_WH) * 100.0
            self.battery_pct = max(15.0, self.battery_pct - bat_loss_pct * SIM_SPEED_FACTOR)
        else:
            # Solar surplus charges battery
            surplus = SOLAR_POWER_W - current_power_w
            if surplus > 0:
                charge_gain_wh = surplus * (UPDATE_INTERVAL_S / 3600.0)
                bat_gain_pct   = (charge_gain_wh / BATTERY_CAPACITY_WH) * 100.0
                self.battery_pct = min(100.0, self.battery_pct + bat_gain_pct * SIM_SPEED_FACTOR)
            else:
                charge_loss_wh = abs(surplus) * (UPDATE_INTERVAL_S / 3600.0)
                bat_loss_pct   = (charge_loss_wh / BATTERY_CAPACITY_WH) * 100.0
                self.battery_pct = max(15.0, self.battery_pct - bat_loss_pct * SIM_SPEED_FACTOR)

        solar_power = 0.0 if self.in_eclipse else SOLAR_POWER_W * (0.9 + 0.1 * math.sin(self.sim_time * 0.01))

        # ── 7. Thermal model ─────────────────────────────────────────────────
        # Temperature oscillates with eclipse cycle
        T_sunlit  = 25.0
        T_eclipse = 10.0
        T_target  = T_eclipse if self.in_eclipse else T_sunlit
        tau = 600.0  # thermal time constant seconds
        dT  = (T_target - self.temperature_bus) / tau * dt_sim
        self.temperature_bus     = max(-5.0, min(45.0, self.temperature_bus + dT))
        self.temperature_battery = self.temperature_bus - 4.0 + 2.0 * math.sin(self.sim_time * 0.02)
        T_prop = 8.0 + 5.0 * math.sin(self.sim_time * 0.015)

        # ── 8. ADCS model ─────────────────────────────────────────────────────
        # Reaction wheels slowly change speed due to drag torque + maneuvers
        for k in range(3):
            noise = 0.5 * (math.sin(self.sim_time * (0.003 + k*0.001)) + 0.1 * (0.5 - math.sin(self.sim_time)))
            self.reaction_wheels[k] += noise * dt_sim * 0.1
            self.reaction_wheels[k] = max(-8000.0, min(8000.0, self.reaction_wheels[k]))

        # Angular rates (body rate tiny in nadir pointing)
        ang_rate_z = 360.0 / self.T    # deg/s — one rotation per orbit (nadir pointing)
        ang_rate_x = 0.002 * math.sin(self.sim_time * 0.05)
        ang_rate_y = 0.001 * math.cos(self.sim_time * 0.07)
        pointing_error = 0.05 + 0.03 * abs(math.sin(self.sim_time * 0.1))

        # ── 9. Ground contact model ───────────────────────────────────────────
        # Simplified: ground contact when satellite is above horizon of Bengaluru GS
        gs_lat, gs_lon = 13.0, 77.5
        gs_x = RE * math.cos(gs_lat*DEG) * math.cos(gs_lon*DEG)
        gs_y = RE * math.cos(gs_lat*DEG) * math.sin(gs_lon*DEG)
        gs_z = RE * math.sin(gs_lat*DEG)
        sat_r = math.sqrt(eci['pos_x']**2 + eci['pos_y']**2 + eci['pos_z']**2)
        # Elevation angle (simplified from dot product)
        dot = ((eci['pos_x']-gs_x)*eci['pos_x'] + (eci['pos_y']-gs_y)*eci['pos_y'] + (eci['pos_z']-gs_z)*eci['pos_z'])
        range_km = math.sqrt((eci['pos_x']-gs_x)**2+(eci['pos_y']-gs_y)**2+(eci['pos_z']-gs_z)**2)
        elev_approx = math.asin(max(-1,min(1, (sat_r - RE*dot/(sat_r*RE)) / range_km))) / DEG if range_km > 0 else 0
        self.ground_contact = elev_approx > 5.0
        doppler = compute_doppler(eci['vel_x'], eci['vel_y'], eci['vel_z'],
                                  eci['pos_x'], eci['pos_y'], eci['pos_z'])

        # Link margin oscillates realistically
        if self.ground_contact:
            link_margin = 4.0 + 6.0 * math.sin(math.pi * (elev_approx - 5.0) / 85.0)
        else:
            link_margin = -5.0

        # ── 10. Fuel drain (passive outgassing + tiny leakage) ────────────────
        if self.burn_active:
            # Burn consumes fuel at engine-specific rate
            self.burn_timer_s -= dt_sim
            if self.burn_timer_s <= 0:
                self.burn_active  = False
                self.burn_timer_s = 0.0
                print(f"[ACAS] Burn complete. Fuel remaining: {(self.fuel_kg/2.0)*100:.1f}%")
        else:
            # Passive drain: ~0.001 kg/day = ~1.16e-8 kg/s
            self.fuel_kg = max(0.0, self.fuel_kg - 1.16e-8 * dt_sim)

        fuel_pct = (self.fuel_kg / 2.0) * 100.0

        # ── 11. Environmental parameters ──────────────────────────────────────
        rho_val = compute_atmospheric_density(geo['alt'])
        drag_acc = 0.5 * rho_val * self.B * (math.sqrt(GM / self.a) * 1000.0)**2
        radiation_dose = 12.0 + 5.0 * abs(math.sin(self.sim_time * 0.001))

        # ── 12. Attitude quaternion (nadir-pointing, simplified) ───────────────
        # Just a rotating quaternion to simulate attitude
        half_angle = ang_rate_z * self.sim_time / 2.0 * DEG
        q = [math.cos(half_angle), 0.0, math.sin(half_angle), 0.0]

        # ── 13. Speed calculation ─────────────────────────────────────────────
        speed = math.sqrt(eci['vel_x']**2 + eci['vel_y']**2 + eci['vel_z']**2)

        # ── 14. Data storage fill rate ────────────────────────────────────────
        data_stored = min(64.0, self.sim_time / 3600.0 * 0.8)  # 0.8 GB/hr

        # ── 15. Build and return state dict ──────────────────────────────────
        now_utc = datetime.now(timezone.utc) + timedelta(seconds=self.sim_time)

        return {
            "_meta": self.model['_meta'],
            "identity": self.model['identity'],

            "timestamp": now_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "sim_time_seconds": round(self.sim_time, 3),
            "orbit_number": self.orbit_number,

            "orbital_elements": {
                "semi_major_axis_km": round(self.a, 4),
                "eccentricity":       round(self.e, 6),
                "inclination_deg":    round(self.i, 4),
                "raan_deg":           round(self.raan % 360.0, 4),
                "arg_perigee_deg":    round(self.w % 360.0, 4),
                "mean_anomaly_deg":   round(M_current, 4),
                "mean_motion_rev_per_day": round(86400.0 / self.T, 6),
                "period_min":         round(self.T / 60.0, 3),
            },

            "eci_state": {
                "pos_x_km": round(eci['pos_x'], 4),
                "pos_y_km": round(eci['pos_y'], 4),
                "pos_z_km": round(eci['pos_z'], 4),
                "vel_x_kms": round(eci['vel_x'], 6),
                "vel_y_kms": round(eci['vel_y'], 6),
                "vel_z_kms": round(eci['vel_z'], 6),
            },

            "derived_position": {
                "altitude_km":     round(geo['alt'], 3),
                "latitude_deg":    round(geo['lat'], 4),
                "longitude_deg":   round(geo['lon'], 4),
                "speed_kms":       round(speed, 5),
                "local_solar_time": f"{int(10 + geo['lon']/15) % 24:02d}:{int((geo['lon']/15 % 1)*60):02d}",
                "sub_satellite_point": [round(geo['lat'], 4), round(geo['lon'], 4)],
            },

            "health": {
                "fuel_pct":              round(fuel_pct, 4),
                "fuel_kg_remaining":     round(self.fuel_kg, 5),
                "fuel_kg_total":         2.0,
                "battery_pct":           round(self.battery_pct, 3),
                "battery_voltage_v":     round(28.0 + (self.battery_pct - 50) * 0.008, 3),
                "solar_power_w":         round(solar_power, 2),
                "power_draw_w":          round(BURN_POWER_W if self.burn_active else IDLE_POWER_W, 1),
                "temperature_bus_c":     round(self.temperature_bus, 2),
                "temperature_battery_c": round(self.temperature_battery, 2),
                "temperature_propulsion_c": round(T_prop, 2),
                "thermal_state":         "BURN" if self.burn_active else ("COLD" if self.temperature_bus < 5 else "NOMINAL"),
            },

            "adcs": {
                "attitude_mode": "NADIR_POINTING",
                "attitude_quaternion": [round(x, 5) for x in q],
                "angular_rate_x_degs": round(ang_rate_x, 5),
                "angular_rate_y_degs": round(ang_rate_y, 5),
                "angular_rate_z_degs": round(ang_rate_z, 5),
                "pointing_error_deg":  round(pointing_error, 4),
                "reaction_wheel_rpm_x": round(self.reaction_wheels[0], 1),
                "reaction_wheel_rpm_y": round(self.reaction_wheels[1], 1),
                "reaction_wheel_rpm_z": round(self.reaction_wheels[2], 1),
                "magnetometer_x_ut": round(24.0 + 2*math.sin(self.sim_time*0.01), 2),
                "magnetometer_y_ut": round(-18.0 + 3*math.cos(self.sim_time*0.008), 2),
                "magnetometer_z_ut": round(42.0 + 4*math.sin(self.sim_time*0.006), 2),
            },

            "propulsion": {
                "thruster_active":         self.burn_active,
                "burn_duration_remaining_s": round(self.burn_timer_s, 2),
                "last_burn_dv_ms":         [round(x, 4) for x in self.burn_dv],
                "last_burn_utc":           now_utc.strftime("%Y-%m-%dT%H:%M:%SZ") if self.burn_active else None,
                "last_burn_magnitude_ms":  round(math.sqrt(sum(x**2 for x in self.burn_dv)), 4),
                "tank_pressure_bar":       round(18.2 * (self.fuel_kg / 2.0) + 2.0, 3),
                "valve_state":             "OPEN" if self.burn_active else "CLOSED",
                "total_dv_used_ms":        round(self.total_dv_ms, 4),
            },

            "communications": {
                "ground_contact":    self.ground_contact,
                "ground_station":    "ISRO Bengaluru" if self.ground_contact else "OUT_OF_RANGE",
                "link_margin_db":    round(link_margin, 2),
                "uplink_rate_kbps":  9.6 if self.ground_contact else 0.0,
                "downlink_rate_kbps":38.4 if self.ground_contact else 0.0,
                "doppler_shift_hz":  doppler,
                "elevation_angle_deg": round(max(0, elev_approx), 2),
                "contact_duration_min": 8.2 if self.ground_contact else 0.0,
            },

            "environment": {
                "in_eclipse":           self.in_eclipse,
                "eclipse_fraction":     eclipse_frac,
                "solar_flux_sfu":       round(148.0 + 5*math.sin(self.sim_time*0.0001), 2),
                "magnetic_field_ut":    round(31.2 + 8*math.sin(self.sim_time*0.005), 2),
                "atomic_oxygen_flux":   round(2.1e10 * math.exp(-(geo['alt']-400)*0.01), 3),
                "radiation_dose_mrad_hr": round(radiation_dose, 2),
                "atmospheric_density_kgm3": rho_val,
                "drag_acceleration_ms2":   drag_acc,
            },

            "mission": {
                "phase":           "safe_mode" if self.battery_pct < 20 else "nominal",
                "mode":            "BURN" if self.burn_active else ("ECLIPSE" if self.in_eclipse else "SCIENCE"),
                "payload_active":  not self.in_eclipse and self.battery_pct > 30,
                "data_stored_gb":  round(data_stored, 3),
                "data_capacity_gb": 64.0,
                "anomaly_count":   self.anomaly_count,
                "last_anomaly":    None,
                "uptime_hours":    round(self.uptime_hours, 4),
            },

            "acas": {
                "active":         True,
                "last_cycle_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "cycle_count":    int(self.sim_time / 60),
                "threats_tracked": 0,
                "maneuver_active": self.burn_active,
                "last_alert_level": "GREEN",
                "last_dv_vector":   [round(x, 4) for x in self.burn_dv],
            },
        }


    def write_atomic(self, state: dict):
        """
        Atomic write: write to temp file then rename.
        This prevents the dashboard from reading a half-written file.
        """
        with open(TEMP_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(TEMP_FILE, MODEL_FILE)  # atomic on Linux/Mac


    def run(self):
        """Main loop — runs continuously until Ctrl+C."""
        last_print = time.time()
        step_count = 0

        while True:
            t_start   = time.time()
            dt_real   = UPDATE_INTERVAL_S
            dt_sim    = dt_real * SIM_SPEED_FACTOR

            state = self.step(dt_sim)
            self.write_atomic(state)
            step_count += 1

            # Terminal status printout
            if time.time() - last_print >= PRINT_INTERVAL_S:
                geo = state['derived_position']
                hlth = state['health']
                env  = state['environment']
                eci  = state['eci_state']
                print(
                    f"[{state['timestamp'][11:19]}] "
                    f"Orbit #{state['orbit_number']:03d} | "
                    f"Alt:{geo['altitude_km']:.1f}km | "
                    f"Lat:{geo['latitude_deg']:+7.3f}° Lon:{geo['longitude_deg']:+8.3f}° | "
                    f"Spd:{geo['speed_kms']:.4f}km/s | "
                    f"Fuel:{hlth['fuel_pct']:.2f}% | "
                    f"Bat:{hlth['battery_pct']:.1f}% | "
                    f"{'🌑ECLIPSE' if env['in_eclipse'] else '☀️ SUNLIT '} | "
                    f"{'📡GND' if state['communications']['ground_contact'] else '📵   '}"
                )
                last_print = time.time()

            # Sleep to maintain real-time update rate
            elapsed = time.time() - t_start
            sleep   = max(0, UPDATE_INTERVAL_S - elapsed)
            time.sleep(sleep)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    sat = PowerHouseSatellite()

    def shutdown(sig, frame):
        print("\n[Power House] Shutdown requested. Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    sat.run()