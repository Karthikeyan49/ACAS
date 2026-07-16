"""
core/pc_analytic.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    The analytical short-encounter collision-probability model (Foster's 2D
    method). Where the LightGBM engine LEARNS Pc from CDM features, this file
    COMPUTES Pc from first-principles geometry: it projects the miss vector and
    the combined position covariance onto the encounter plane (the plane
    perpendicular to the relative velocity) and integrates a 2D Gaussian over
    the circular hard-body-radius keep-out zone.

    It exists so the runtime never has to trust the ML model alone. The arbiter
    takes the conservative maximum of the ML Pc and this analytic Pc, giving a
    physically-grounded floor that is valid even when the ML model runs far
    outside its training distribution.

CALLED FROM
    model/lgbm_engine.py    pc_arbiter(model_pc, analytic_pc)  — after inference
    tests/test_pc_analytic.py

CALLS INTO
    numpy  (linear algebra + vectorised quadrature)
    core.config_loader  (default hard-body radius)          — stdlib-safe only

WHAT IT PROVIDES
    compute_pc_foster(miss_vector_m, rel_velocity_ms,
                      cov_combined_m2, hard_body_radius_m=None) -> float
        Foster 2D short-encounter Pc in [0, 1]. Deterministic, < 1 ms.

    pc_arbiter(ml_pc, analytic_pc) -> dict
        {pc, ml_pc, analytic_pc, source, disagreement}
        pc = max(ml_pc, analytic_pc)  (conservative); source names the winner;
        disagreement is True when the two differ by > 2 orders of magnitude.

    default_hard_body_radius() -> float
        Reads hard_body_radius_m from config/thresholds.yaml (fallback 20 m).

METHOD (Foster 1992, NASA CARA short-encounter assumption)
    1. Build an orthonormal basis (u1, u2) spanning the plane perpendicular to
       the relative velocity — the "encounter plane" the objects sweep past in.
    2. Project the 3D miss vector onto that plane  →  2D offset b.
    3. Project the 3D combined position covariance  →  2x2 in-plane covariance.
    4. Integrate the 2D Gaussian N(b, C_2d) over the disk of radius R (the
       combined hard-body radius) centred at the origin, in polar coordinates.

DEGENERATE HANDLING
    • singular / non-finite covariance  → isotropic fallback sigma
    • zero relative velocity            → arbitrary but deterministic plane
══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np

# Disagreement flag fires when ML and analytic Pc differ by more than this many
# orders of magnitude (base-10).
_DISAGREEMENT_DECADES = 2.0

# Floor used so log-ratio comparisons and empty integrals stay finite.
_TINY = 1e-300

# Polar quadrature resolution. R is small relative to sigma in the common case,
# so the integrand is smooth; 64x64 midpoint nodes converge well under 1 ms and
# stay accurate even when sigma ~ R (near-certain hits).
_N_R = 64
_N_THETA = 64


def default_hard_body_radius() -> float:
    """Combined hard-body radius (m) from config, with a 20 m fallback."""
    try:
        from core import config_loader
        r = config_loader.get("hard_body_radius_m", 20.0)
        return float(r) if r is not None else 20.0
    except Exception:
        return 20.0


def _encounter_plane_basis(rel_velocity_ms: np.ndarray):
    """Two orthonormal vectors spanning the plane perpendicular to velocity.

    For a zero (or negligible) relative velocity the encounter plane is
    undefined; we return an arbitrary but deterministic orthonormal basis so
    the computation degrades gracefully instead of raising.
    """
    v = np.asarray(rel_velocity_ms, dtype=float).reshape(3)
    speed = float(np.linalg.norm(v))

    if not np.isfinite(speed) or speed < 1e-9:
        # No well-defined encounter plane — pick a fixed canonical basis.
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

    v_hat = v / speed
    # Seed vector least parallel to v_hat, chosen deterministically.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(v_hat[0]) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])

    u1 = seed - np.dot(seed, v_hat) * v_hat
    u1 /= (np.linalg.norm(u1) + 1e-12)
    u2 = np.cross(v_hat, u1)
    u2 /= (np.linalg.norm(u2) + 1e-12)
    return u1, u2


def _sanitise_cov_2d(c2d: np.ndarray, fallback_sigma: float) -> np.ndarray:
    """Return a symmetric positive-definite 2x2 covariance.

    Falls back to an isotropic covariance when the projected covariance is
    non-finite, non-positive, or numerically singular.
    """
    iso = np.eye(2) * (fallback_sigma ** 2)

    if c2d is None or not np.all(np.isfinite(c2d)):
        return iso

    c2d = 0.5 * (c2d + c2d.T)  # symmetrise
    det = c2d[0, 0] * c2d[1, 1] - c2d[0, 1] * c2d[1, 0]
    trace = c2d[0, 0] + c2d[1, 1]

    # Reject singular / negative-variance / ill-conditioned matrices.
    if (det <= 0.0 or trace <= 0.0
            or c2d[0, 0] <= 0.0 or c2d[1, 1] <= 0.0
            or det < 1e-9 * (trace * 0.5) ** 2):
        return iso
    return c2d


def compute_pc_foster(miss_vector_m,
                      rel_velocity_ms,
                      cov_combined_m2,
                      hard_body_radius_m: float = None) -> float:
    """Foster 2D short-encounter collision probability.

    Parameters
    ----------
    miss_vector_m     : (3,) relative position at TCA, metres.
    rel_velocity_ms   : (3,) relative velocity at TCA, m/s (defines the plane).
    cov_combined_m2   : (3,3) combined position covariance, m^2. May be None,
                        singular, or non-finite — an isotropic fallback is used.
    hard_body_radius_m: combined keep-out radius, metres. Defaults to config
                        (hard_body_radius_m, 20 m).

    Returns
    -------
    float  Pc in [0, 1]. Deterministic.
    """
    if hard_body_radius_m is None:
        hard_body_radius_m = default_hard_body_radius()
    R = float(hard_body_radius_m)
    if R <= 0.0:
        return 0.0

    miss = np.asarray(miss_vector_m, dtype=float).reshape(3)

    # ── Encounter-plane basis and projections ────────────────────────────────
    u1, u2 = _encounter_plane_basis(rel_velocity_ms)
    P = np.vstack([u1, u2])                     # 2x3 projection

    b = P @ miss                                # 2D in-plane miss offset (m)

    # Isotropic fallback sigma: use the in-plane miss magnitude as a physical
    # scale so a degenerate covariance still yields a sane, finite Pc rather
    # than a spuriously large or zero one. Never smaller than the hard body.
    fallback_sigma = max(R, float(np.linalg.norm(b)) * 0.5, 1.0)

    cov = np.asarray(cov_combined_m2, dtype=float) if cov_combined_m2 is not None else None
    if cov is not None and cov.shape == (3, 3):
        c2d = P @ cov @ P.T
    else:
        c2d = None
    c2d = _sanitise_cov_2d(c2d, fallback_sigma)

    # ── Inverse + normalisation of the 2D Gaussian ───────────────────────────
    det = c2d[0, 0] * c2d[1, 1] - c2d[0, 1] * c2d[1, 0]
    inv = np.array([[c2d[1, 1], -c2d[0, 1]],
                    [-c2d[1, 0], c2d[0, 0]]]) / det
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det))

    # ── Polar-grid quadrature over the disk |x| <= R centred at origin ───────
    # Midpoint rule in (r, theta). weight = r * dr * dtheta.
    dr = R / _N_R
    dth = 2.0 * np.pi / _N_THETA
    r = (np.arange(_N_R) + 0.5) * dr                 # (_N_R,)
    th = (np.arange(_N_THETA) + 0.5) * dth           # (_N_THETA,)

    rr, tt = np.meshgrid(r, th, indexing="ij")       # (_N_R, _N_THETA)
    x = rr * np.cos(tt) - b[0]                        # centre Gaussian at b
    y = rr * np.sin(tt) - b[1]

    # quadratic form (x)^T inv (x)
    q = (inv[0, 0] * x * x
         + (inv[0, 1] + inv[1, 0]) * x * y
         + inv[1, 1] * y * y)
    dens = norm * np.exp(-0.5 * q)
    pc = float(np.sum(dens * rr) * dr * dth)

    if not np.isfinite(pc):
        return 0.0
    return max(0.0, min(1.0, pc))


def pc_arbiter(ml_pc: float, analytic_pc: float) -> dict:
    """Conservative arbiter between the ML Pc and the analytic Pc.

    Returns a dict describing the decision:
        pc            : max(ml_pc, analytic_pc)   — conservative choice
        ml_pc         : the machine-learning Pc as given
        analytic_pc   : the Foster analytic Pc as given
        source        : 'ml' | 'analytic' | 'tie'  — which produced pc
        disagreement  : True when the two differ by > 2 orders of magnitude
                        (a flag to surface for operator review / logging)
    """
    ml = float(ml_pc) if ml_pc is not None and np.isfinite(ml_pc) else 0.0
    an = float(analytic_pc) if analytic_pc is not None and np.isfinite(analytic_pc) else 0.0
    ml = max(0.0, min(1.0, ml))
    an = max(0.0, min(1.0, an))

    pc = max(ml, an)
    if ml == an:
        source = "tie"
    elif ml > an:
        source = "ml"
    else:
        source = "analytic"

    lo, hi = min(ml, an), max(ml, an)
    if hi <= _TINY:
        disagreement = False
    else:
        decades = np.log10(hi) - np.log10(max(lo, _TINY))
        disagreement = bool(decades > _DISAGREEMENT_DECADES)

    return {
        "pc": pc,
        "ml_pc": ml,
        "analytic_pc": an,
        "source": source,
        "disagreement": disagreement,
    }
