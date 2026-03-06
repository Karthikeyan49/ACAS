"""
pipeline/tle_processor.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    Converts a pair of raw TLE strings into the 103-column CDM feature dict
    that LightGBM was trained on. Handles TLE parsing, SGP4 propagation,
    TCA finding, RIC frame decomposition, and covariance estimation.

CALLED FROM
    pipeline/model_bridge.py   tle_pair_to_cdm_features(sat_tle, obj_tle)
    Do not call directly from dashboard or controller — use model_bridge.

CALLS INTO
    sgp4     Satrec.twoline2rv(), sat.sgp4()
    numpy
    Nothing from this project.

WHAT IT PROVIDES
    parse_tle(tle_block) → dict
        {name, norad_id, line1, line2, epoch, mean_motion, eccentricity,
         inclination, raan, arg_perigee, mean_anomaly, bstar, tle_age_days}

    propagate(tle, dt_minutes) → (pos_km, vel_kms)
        SGP4 at dt_minutes from epoch. Keplerian fallback if sgp4 unavailable.

    find_tca(sat_tle, obj_tle) → dict
        7-day scan at 1-min, refine at 10-sec around minimum.
        {tca_datetime, time_to_tca_days, miss_distance_m,
         relative_speed_ms, sat_pos, obj_pos, sat_vel, obj_vel}

    eci_to_ric(rel_pos, sat_vel) → np.ndarray(3)
        ECI relative position → Radial-In-track-Cross-track frame.

    estimate_uncertainty(tle, object_type) → dict
        sigma_r/t/n from object type + TLE age.

    tle_pair_to_cdm_features(sat_tle_str, obj_tle_str,
                              object_type, **kwargs) → dict
        MASTER FUNCTION. Returns 103-column CDM feature dict for LightGBM.

MIGRATED FROM
    satellite_lgbm/tle_processor.py  — no import changes
══════════════════════════════════════════════════════════════════════════════
"""

"""
tle_processor.py
────────────────
Receives a pair of TLE inputs from the dashboard:
  - satellite_tle : your satellite (target)
  - object_tle    : threat object / debris (chaser)

Computes all CDM-equivalent features the LightGBM model needs.

Called by model_bridge.py — do not call directly from dashboard.
"""

import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger("tle_processor")

# ── Physical constants ────────────────────────────────────────────────────────
EARTH_RADIUS_KM  = 6371.0
MU_KM3_S2        = 398600.4418       # Earth gravitational parameter km³/s²
SECONDS_PER_DAY  = 86400.0
DEG2RAD          = np.pi / 180.0
RAD2DEG          = 180.0 / np.pi

# ── Conjunction search config ─────────────────────────────────────────────────
SEARCH_WINDOW_DAYS    = 7            # how many days ahead to search
TIME_STEP_MINUTES     = 1.0          # propagation step size (minutes)
FINE_STEP_SECONDS     = 10.0         # fine search step around TCA (seconds)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TLE PARSING
# ═════════════════════════════════════════════════════════════════════════════

def parse_tle(tle_block: str) -> Dict:
    """
    Parse a TLE block (2-line or 3-line format) into orbital elements.

    Input format accepted:
        "NAME\\n1 NNNNN...\\n2 NNNNN..."
        OR just two lines: "1 NNNNN...\\n2 NNNNN..."

    Returns dict of orbital elements + raw lines for SGP4.
    """
    lines = [l.strip() for l in tle_block.strip().splitlines() if l.strip()]

    if len(lines) == 2:
        name  = f"OBJECT_{lines[0][2:7].strip()}"
        line1, line2 = lines[0], lines[1]
    elif len(lines) >= 3:
        name  = lines[0]
        line1 = lines[1]
        line2 = lines[2]
    else:
        raise ValueError(f"Invalid TLE format — expected 2 or 3 lines, got {len(lines)}")

    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError("TLE lines must start with '1 ' and '2 '")

    try:
        # Line 1 fields
        norad_id     = int(line1[2:7])
        epoch_year   = int(line1[18:20])
        epoch_day    = float(line1[20:32])
        bstar_str    = line1[53:61].strip()

        # Parse B* drag term
        try:
            if len(bstar_str) >= 6:
                mantissa = float(bstar_str[:5]) * 1e-5
                exp      = int(bstar_str[5:])
                bstar    = mantissa * (10 ** exp)
            else:
                bstar = 0.0
        except Exception:
            bstar = 0.0

        # Line 2 fields
        inclination   = float(line2[8:16])
        raan          = float(line2[17:25])    # right ascension of ascending node
        eccentricity  = float("0." + line2[26:33])
        arg_perigee   = float(line2[34:42])
        mean_anomaly  = float(line2[43:51])
        mean_motion   = float(line2[52:63])    # revolutions per day
        rev_number    = int(line2[63:68])

        # Derived orbital elements
        n_rad_s  = mean_motion * 2 * np.pi / SECONDS_PER_DAY
        sma_km   = (MU_KM3_S2 / n_rad_s ** 2) ** (1.0 / 3.0)
        h_apo_km = sma_km * (1 + eccentricity) - EARTH_RADIUS_KM
        h_per_km = sma_km * (1 - eccentricity) - EARTH_RADIUS_KM

        # Epoch as datetime
        full_year = (2000 + epoch_year) if epoch_year < 57 else (1900 + epoch_year)
        epoch_dt  = datetime(full_year, 1, 1, tzinfo=timezone.utc) + \
                    timedelta(days=epoch_day - 1)

        return {
            "name":          name.strip(),
            "norad_id":      norad_id,
            "line1":         line1,
            "line2":         line2,
            "epoch":         epoch_dt,
            # Keplerian elements
            "inclination":   inclination,      # degrees
            "raan":          raan,             # degrees
            "eccentricity":  eccentricity,
            "arg_perigee":   arg_perigee,      # degrees
            "mean_anomaly":  mean_anomaly,     # degrees
            "mean_motion":   mean_motion,      # rev/day
            "bstar":         bstar,
            # Derived
            "sma_km":        sma_km,
            "h_apo_km":      h_apo_km,
            "h_per_km":      h_per_km,
            "period_min":    1440.0 / mean_motion,
        }

    except (ValueError, IndexError) as e:
        raise ValueError(f"TLE parsing failed: {e}\nLine1: {line1}\nLine2: {line2}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ORBIT PROPAGATION
# ═════════════════════════════════════════════════════════════════════════════

def propagate(tle: Dict, dt_minutes: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate TLE forward by dt_minutes.
    Uses SGP4 if installed, falls back to analytical Keplerian propagator.

    Returns:
        position_km  : np.array([x, y, z]) in ECI frame
        velocity_km_s: np.array([vx, vy, vz]) in ECI frame
    """
    try:
        return _propagate_sgp4(tle, dt_minutes)
    except ImportError:
        logger.debug("sgp4 not installed — using Keplerian propagator")
        return _propagate_keplerian(tle, dt_minutes)


def _propagate_sgp4(tle: Dict, dt_minutes: float) -> Tuple[np.ndarray, np.ndarray]:
    """High-accuracy SGP4 propagation. Requires: pip install sgp4"""
    from sgp4.api import Satrec, jday

    sat = Satrec.twoline2rv(tle["line1"], tle["line2"])
    t   = datetime.now(timezone.utc) + timedelta(minutes=dt_minutes)
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                  t.second + t.microsecond / 1e6)
    err, r, v = sat.sgp4(jd, fr)
    if err != 0:
        raise ValueError(f"SGP4 error code {err}")
    return np.array(r), np.array(v)


def _propagate_keplerian(tle: Dict, dt_minutes: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytical Keplerian propagation (fallback).
    Accurate for short propagation windows and near-circular orbits.
    """
    inc  = tle["inclination"]  * DEG2RAD
    raan = tle["raan"]         * DEG2RAD
    ecc  = tle["eccentricity"]
    w    = tle["arg_perigee"]  * DEG2RAD
    M0   = tle["mean_anomaly"] * DEG2RAD
    n    = tle["mean_motion"]  * 2 * np.pi / SECONDS_PER_DAY   # rad/s
    sma  = tle["sma_km"]

    # Propagate mean anomaly
    M = M0 + n * dt_minutes * 60.0

    # Solve Kepler's equation: M = E - e*sin(E)
    E = _solve_kepler(M, ecc)

    # True anomaly
    nu = 2.0 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(E / 2),
        np.sqrt(1 - ecc) * np.cos(E / 2)
    )

    # Radius
    r_mag = sma * (1 - ecc * np.cos(E))

    # Position in orbital plane (perifocal frame)
    x_pf = r_mag * np.cos(nu)
    y_pf = r_mag * np.sin(nu)

    # Velocity in perifocal frame
    p    = sma * (1 - ecc**2)
    vx_pf = -np.sqrt(MU_KM3_S2 / p) * np.sin(nu)
    vy_pf =  np.sqrt(MU_KM3_S2 / p) * (ecc + np.cos(nu))

    # Rotation matrix: perifocal → ECI
    R = _perifocal_to_eci(raan, inc, w)

    pos = R @ np.array([x_pf, y_pf, 0.0])
    vel = R @ np.array([vx_pf, vy_pf, 0.0])

    return pos, vel


def _solve_kepler(M: float, ecc: float, tol: float = 1e-10) -> float:
    """Newton-Raphson solver for Kepler's equation."""
    E = M.copy() if hasattr(M, 'copy') else float(M)
    for _ in range(50):
        dE = (M - E + ecc * np.sin(E)) / (1 - ecc * np.cos(E))
        E  += dE
        if abs(dE) < tol:
            break
    return E


def _perifocal_to_eci(raan: float, inc: float, w: float) -> np.ndarray:
    """3x3 rotation matrix from perifocal to ECI frame."""
    cr, sr = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc),  np.sin(inc)
    cw, sw = np.cos(w),    np.sin(w)

    return np.array([
        [ cr*cw - sr*sw*ci,  -cr*sw - sr*cw*ci,  sr*si],
        [ sr*cw + cr*sw*ci,  -sr*sw + cr*cw*ci, -cr*si],
        [ sw*si,              cw*si,              ci   ],
    ])


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CONJUNCTION GEOMETRY
# ═════════════════════════════════════════════════════════════════════════════

def eci_to_ric(
    r_ref: np.ndarray, v_ref: np.ndarray,
    r_obj: np.ndarray, v_obj: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert ECI relative state to RIC (Radial-In-track-Cross-track) frame.

    RIC matches exactly the CDM column naming:
        relative_position_r → radial
        relative_position_t → in-track (transverse)
        relative_position_n → cross-track (normal)
    """
    dr = r_obj - r_ref           # relative position (km)
    dv = v_obj - v_ref           # relative velocity (km/s)

    r_hat = r_ref / np.linalg.norm(r_ref)
    h_vec = np.cross(r_ref, v_ref)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)

    # Project into RIC
    r_ric = np.array([np.dot(dr, r_hat),
                      np.dot(dr, t_hat),
                      np.dot(dr, n_hat)]) * 1000.0    # km → m

    v_ric = np.array([np.dot(dv, r_hat),
                      np.dot(dv, t_hat),
                      np.dot(dv, n_hat)]) * 1000.0    # km/s → m/s

    return r_ric, v_ric


def find_tca(sat_tle: Dict, obj_tle: Dict) -> Dict:
    """
    Find Time of Closest Approach (TCA) between satellite and object.

    Two-phase search:
      Phase 1 — coarse scan at 1-minute steps over SEARCH_WINDOW_DAYS
      Phase 2 — fine scan at 10-second steps around coarse TCA

    Returns full conjunction geometry at TCA.
    """
    # ── Phase 1: Coarse scan ─────────────────────────────────────────────
    coarse_steps = np.arange(0, SEARCH_WINDOW_DAYS * 1440, TIME_STEP_MINUTES)
    min_dist_km  = np.inf
    tca_min_coarse = 0.0

    for t_min in coarse_steps:
        r_s, _ = propagate(sat_tle, t_min)
        r_o, _ = propagate(obj_tle, t_min)
        d = np.linalg.norm(r_s - r_o)
        if d < min_dist_km:
            min_dist_km     = d
            tca_min_coarse  = t_min

    # ── Phase 2: Fine scan ± 5 minutes around coarse TCA ─────────────────
    fine_start = max(0, tca_min_coarse - 5)
    fine_end   = tca_min_coarse + 5
    fine_steps = np.arange(fine_start * 60, fine_end * 60, FINE_STEP_SECONDS)

    min_dist_km = np.inf
    tca_sec     = fine_start * 60

    for t_sec in fine_steps:
        t_min_f = t_sec / 60.0
        r_s, _ = propagate(sat_tle, t_min_f)
        r_o, _ = propagate(obj_tle, t_min_f)
        d = np.linalg.norm(r_s - r_o)
        if d < min_dist_km:
            min_dist_km = d
            tca_sec     = t_sec

    # ── Final state at TCA ────────────────────────────────────────────────
    tca_min  = tca_sec / 60.0
    r_s, v_s = propagate(sat_tle, tca_min)
    r_o, v_o = propagate(obj_tle, tca_min)

    r_ric, v_ric = eci_to_ric(r_s, v_s, r_o, v_o)

    time_to_tca_days = tca_min / 1440.0
    miss_dist_m      = min_dist_km * 1000.0
    rel_speed_ms     = float(np.linalg.norm(v_ric))

    logger.info(f"TCA found at +{tca_min:.1f} min | "
                f"Miss distance: {miss_dist_m:.0f} m | "
                f"Relative speed: {rel_speed_ms:.0f} m/s")

    return {
        "time_to_tca":         time_to_tca_days,
        "miss_distance":       miss_dist_m,
        "relative_speed":      rel_speed_ms,
        "relative_position_r": float(r_ric[0]),
        "relative_position_t": float(r_ric[1]),
        "relative_position_n": float(r_ric[2]),
        "relative_velocity_r": float(v_ric[0]),
        "relative_velocity_t": float(v_ric[1]),
        "relative_velocity_n": float(v_ric[2]),
        "tca_minutes_from_now": tca_min,
        "tca_datetime":        (datetime.now(timezone.utc) +
                                timedelta(minutes=tca_min)).isoformat(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — UNCERTAINTY ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

# Sigma values (metres) by object type — from operational statistics
_SIGMA_TABLE = {
    "PAYLOAD":      {"r": 10,    "t": 400,   "n": 5,    "rdot": 0.5,   "tdot": 0.008,  "ndot": 0.015},
    "ROCKET BODY":  {"r": 50,    "t": 2000,  "n": 30,   "rdot": 2.5,   "tdot": 0.04,   "ndot": 0.09},
    "DEBRIS":       {"r": 200,   "t": 10000, "n": 100,  "rdot": 10.0,  "tdot": 0.20,   "ndot": 0.30},
    "UNKNOWN":      {"r": 300,   "t": 20000, "n": 150,  "rdot": 15.0,  "tdot": 0.40,   "ndot": 0.45},
}

def estimate_uncertainty(object_type: str, tle: Dict) -> Dict:
    """
    Estimate positional and velocity uncertainty from object type and TLE age.
    Scales uncertainty by how old the TLE epoch is (older = less certain).
    """
    s = _SIGMA_TABLE.get(object_type.upper(), _SIGMA_TABLE["UNKNOWN"])

    # Age scaling: uncertainty grows ~√t with TLE age
    tle_age_days = max(0, (datetime.now(timezone.utc) - tle["epoch"]).total_seconds() / 86400)
    age_scale    = max(1.0, np.sqrt(1 + tle_age_days / 3.0))

    r     = s["r"]    * age_scale
    t     = s["t"]    * age_scale
    n     = s["n"]    * age_scale
    rdot  = s["rdot"] * age_scale
    tdot  = s["tdot"] * age_scale
    ndot  = s["ndot"] * age_scale

    return {
        "sigma_r":    r,    "sigma_t":    t,    "sigma_n":    n,
        "sigma_rdot": rdot, "sigma_tdot": tdot, "sigma_ndot": ndot,
        "cov_det":    r**2 * t**2 * n**2,       # position covariance determinant
        "tle_age_days": tle_age_days,
    }


def estimate_physical(object_type: str, rcs_m2: Optional[float] = None) -> Dict:
    """Estimate drag/reflectivity properties from RCS and object type."""
    _defaults = {
        "PAYLOAD":     0.5,
        "ROCKET BODY": 3.0,
        "DEBRIS":      0.1,
        "UNKNOWN":     0.1,
    }
    rcs  = rcs_m2 if rcs_m2 is not None else _defaults.get(object_type.upper(), 0.1)
    area = rcs * 2.2
    mass = max(1.0, rcs * 80)
    return {
        "rcs_estimate":      rcs,
        "cd_area_over_mass": 2.2 * area / mass,
        "cr_area_over_mass": 1.3 * area / mass,
        "sedr":              2.2 * area / mass * 1e-5,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MASTER FUNCTION: TLE PAIR → CDM FEATURE DICT
# ═════════════════════════════════════════════════════════════════════════════

def tle_pair_to_cdm_features(
    satellite_tle_str:  str,
    object_tle_str:     str,
    object_type:        str   = "UNKNOWN",
    satellite_rcs_m2:   float = None,
    object_rcs_m2:      float = None,
    space_weather:      Dict  = None,
    mission_id:         int   = 1,
    event_id:           int   = 0,
) -> Dict:
    """
    ╔══════════════════════════════════════════════════════════════╗
    ║  MASTER FUNCTION — called by model_bridge.py                ║
    ║                                                              ║
    ║  Input : Two raw TLE strings from the dashboard              ║
    ║  Output: Full CDM feature dict ready for LightGBM            ║
    ╚══════════════════════════════════════════════════════════════╝

    Parameters
    ----------
    satellite_tle_str : raw TLE string of your satellite (target)
    object_tle_str    : raw TLE string of threat object (chaser)
    object_type       : "PAYLOAD" | "ROCKET BODY" | "DEBRIS" | "UNKNOWN"
    satellite_rcs_m2  : radar cross-section of satellite (m²), optional
    object_rcs_m2     : radar cross-section of object (m²), optional
    space_weather     : dict with F10, F3M, SSN, AP (uses defaults if None)
    mission_id        : mission identifier (integer)
    event_id          : event counter

    Returns
    -------
    dict : 100+ CDM features — directly consumable by LightGBM model
    """

    # ── Step A: Parse both TLEs ───────────────────────────────────────────
    logger.info("Parsing TLE data...")
    sat_tle = parse_tle(satellite_tle_str)
    obj_tle = parse_tle(object_tle_str)
    logger.info(f"  Satellite : {sat_tle['name']} (NORAD {sat_tle['norad_id']})")
    logger.info(f"  Object    : {obj_tle['name']} (NORAD {obj_tle['norad_id']})")

    # ── Step B: Find TCA and conjunction geometry ─────────────────────────
    logger.info("Computing Time of Closest Approach...")
    conj = find_tca(sat_tle, obj_tle)

    # ── Step C: Uncertainty estimates ─────────────────────────────────────
    sat_unc  = estimate_uncertainty("PAYLOAD", sat_tle)
    obj_unc  = estimate_uncertainty(object_type, obj_tle)

    # ── Step D: Physical properties ───────────────────────────────────────
    sat_phys = estimate_physical("PAYLOAD", satellite_rcs_m2)
    obj_phys = estimate_physical(object_type, object_rcs_m2)

    # ── Step E: Space weather ─────────────────────────────────────────────
    sw = space_weather or {"F10": 150.0, "F3M": 148.0, "SSN": 80.0, "AP": 12.0}

    # ── Step F: Derived geometry ──────────────────────────────────────────
    combined_sigma_r  = np.sqrt(sat_unc["sigma_r"]**2 + obj_unc["sigma_r"]**2)
    mahalanobis       = conj["miss_distance"] / (combined_sigma_r + 1e-9)
    geocentric_lat    = float(np.degrees(np.arcsin(np.clip(
        conj["relative_position_n"] / (conj["miss_distance"] + 1e-9), -1, 1
    ))))

    # ── Step G: Observation quality (realistic defaults) ─────────────────
    # Satellite (well-tracked payload — good OD quality)
    t_od_age = sat_unc["tle_age_days"]
    # Object (debris — poorer tracking)
    c_od_age = obj_unc["tle_age_days"]

    # ── Step H: Assemble full CDM feature dict ────────────────────────────
    features = {
        # Identifiers
        "event_id":                    event_id,
        "mission_id":                  mission_id,

        # ── Core conjunction geometry ─────────────────────────────────────
        "time_to_tca":                 conj["time_to_tca"],
        "miss_distance":               conj["miss_distance"],
        "relative_speed":              conj["relative_speed"],
        "relative_position_r":         conj["relative_position_r"],
        "relative_position_t":         conj["relative_position_t"],
        "relative_position_n":         conj["relative_position_n"],
        "relative_velocity_r":         conj["relative_velocity_r"],
        "relative_velocity_t":         conj["relative_velocity_t"],
        "relative_velocity_n":         conj["relative_velocity_n"],

        # ── Target (your satellite) orbital elements ───────────────────────
        "t_j2k_sma":                   sat_tle["sma_km"],
        "t_j2k_ecc":                   sat_tle["eccentricity"],
        "t_j2k_inc":                   sat_tle["inclination"],
        "t_h_apo":                     sat_tle["h_apo_km"],
        "t_h_per":                     sat_tle["h_per_km"],
        "t_span":                      1.5,

        # ── Chaser (threat object) orbital elements ───────────────────────
        "c_j2k_sma":                   obj_tle["sma_km"],
        "c_j2k_ecc":                   obj_tle["eccentricity"],
        "c_j2k_inc":                   obj_tle["inclination"],
        "c_h_apo":                     obj_tle["h_apo_km"],
        "c_h_per":                     obj_tle["h_per_km"],
        "c_span":                      2.0,
        "c_object_type":               object_type,

        # ── Target uncertainty (sigma) ────────────────────────────────────
        "t_sigma_r":                   sat_unc["sigma_r"],
        "t_sigma_t":                   sat_unc["sigma_t"],
        "t_sigma_n":                   sat_unc["sigma_n"],
        "t_sigma_rdot":                sat_unc["sigma_rdot"],
        "t_sigma_tdot":                sat_unc["sigma_tdot"],
        "t_sigma_ndot":                sat_unc["sigma_ndot"],

        # ── Chaser uncertainty ────────────────────────────────────────────
        "c_sigma_r":                   obj_unc["sigma_r"],
        "c_sigma_t":                   obj_unc["sigma_t"],
        "c_sigma_n":                   obj_unc["sigma_n"],
        "c_sigma_rdot":                obj_unc["sigma_rdot"],
        "c_sigma_tdot":                obj_unc["sigma_tdot"],
        "c_sigma_ndot":                obj_unc["sigma_ndot"],

        # ── Covariance determinants ───────────────────────────────────────
        "t_position_covariance_det":   sat_unc["cov_det"],
        "c_position_covariance_det":   obj_unc["cov_det"],
        "mahalanobis_distance":        mahalanobis,

        # ── Covariance matrix off-diagonal terms (set to 0) ───────────────
        **{col: 0.0 for col in [
            "t_ct_r","t_cn_r","t_cn_t",
            "t_crdot_r","t_crdot_t","t_crdot_n",
            "t_ctdot_r","t_ctdot_t","t_ctdot_n","t_ctdot_rdot",
            "t_cndot_r","t_cndot_t","t_cndot_n","t_cndot_rdot","t_cndot_tdot",
            "c_ct_r","c_cn_r","c_cn_t",
            "c_crdot_r","c_crdot_t","c_crdot_n",
            "c_ctdot_r","c_ctdot_t","c_ctdot_n","c_ctdot_rdot",
            "c_cndot_r","c_cndot_t","c_cndot_n","c_cndot_rdot","c_cndot_tdot",
        ]},

        # ── Physical properties ───────────────────────────────────────────
        "t_rcs_estimate":              sat_phys["rcs_estimate"],
        "c_rcs_estimate":              obj_phys["rcs_estimate"],
        "t_cd_area_over_mass":         sat_phys["cd_area_over_mass"],
        "c_cd_area_over_mass":         obj_phys["cd_area_over_mass"],
        "t_cr_area_over_mass":         sat_phys["cr_area_over_mass"],
        "c_cr_area_over_mass":         obj_phys["cr_area_over_mass"],
        "t_sedr":                      sat_phys["sedr"],
        "c_sedr":                      obj_phys["sedr"],

        # ── Observation quality ───────────────────────────────────────────
        "t_time_lastob_start":         max(0.5, t_od_age),
        "t_time_lastob_end":           0.0,
        "t_recommended_od_span":       7.5,
        "t_actual_od_span":            min(7.49, max(1.0, 7.49 - t_od_age * 0.1)),
        "t_obs_available":             max(50, 215 - int(t_od_age * 5)),
        "t_obs_used":                  max(48, 214 - int(t_od_age * 5)),
        "t_residuals_accepted":        max(85.0, 99.4 - t_od_age * 0.5),
        "t_weighted_rms":              min(3.0, 1.293 + t_od_age * 0.05),

        "c_time_lastob_start":         max(5.0, c_od_age * 10),
        "c_time_lastob_end":           2.0,
        "c_recommended_od_span":       29.45,
        "c_actual_od_span":            min(29.44, max(5.0, 29.44 - c_od_age * 0.5)),
        "c_obs_available":             max(5, 18 - int(c_od_age * 0.3)),
        "c_obs_used":                  max(4, 17 - int(c_od_age * 0.3)),
        "c_residuals_accepted":        max(60.0, 83.3 - c_od_age * 1.0),
        "c_weighted_rms":              min(8.0, 4.113 + c_od_age * 0.2),

        # ── Geometry ──────────────────────────────────────────────────────
        "geocentric_latitude":         geocentric_lat,
        "azimuth":                     0.0,
        "elevation":                   0.0,

        # ── Space weather ─────────────────────────────────────────────────
        "F10":                         sw.get("F10", 150.0),
        "F3M":                         sw.get("F3M", 148.0),
        "SSN":                         sw.get("SSN", 80.0),
        "AP":                          sw.get("AP",  12.0),

        # ── Extra metadata (not fed to model, used for dashboard response) ─
        "_meta": {
            "satellite_name":    sat_tle["name"],
            "satellite_norad":   sat_tle["norad_id"],
            "object_name":       obj_tle["name"],
            "object_norad":      obj_tle["norad_id"],
            "tca_datetime":      conj["tca_datetime"],
            "tca_minutes":       conj["tca_minutes_from_now"],
            "sat_tle_age_days":  sat_unc["tle_age_days"],
            "obj_tle_age_days":  obj_unc["tle_age_days"],
            "object_type":       object_type,
        }
    }

    return features