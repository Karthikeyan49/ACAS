/*
 * risk_scorer.h — ACAS Risk Scorer (C Port)
 * ==========================================
 * C translation of models/risk_scorer.py
 *
 * This is the FLIGHT SOFTWARE version of the risk scorer.
 * It contains zero external dependencies — only standard C math library.
 * Suitable for any satellite OBC running C (FreeRTOS, RTEMS, bare metal, Linux).
 *
 * Python original: models/risk_scorer.py
 * Logic is identical — same 6 limitations, same thresholds, same maneuver math.
 *
 * IN-SPACe context:
 *   This file proves the Atmanirbhar (indigenous) claim.
 *   The safety-critical decision layer has ZERO foreign library dependencies.
 *   It can be compiled for ARM, SPARC LEON3, AVR32 — any satellite CPU.
 */

#ifndef RISK_SCORER_H
#define RISK_SCORER_H

#include <math.h>
#include <string.h>
#include <stdio.h>

/* ── Alert levels ─────────────────────────────────────────────────────────── */
typedef enum {
    ALERT_GREEN  = 0,   /* Pc < 1e-5  — no action                            */
    ALERT_YELLOW = 1,   /* Pc < 1e-4  — monitor, downlink alert               */
    ALERT_ORANGE = 2,   /* Pc < 1e-3  — prepare maneuver                      */
    ALERT_RED    = 3    /* Pc >= 1e-3 — act now                               */
} AlertLevel;

static const char* ALERT_NAMES[] = {"GREEN", "YELLOW", "ORANGE", "RED"};

/* ── Satellite health state ───────────────────────────────────────────────── */
typedef struct {
    float fuel_pct;         /* 0-100  remaining propellant                    */
    float battery_pct;      /* 0-100  electrical power                        */
    float altitude_km;      /* current orbital altitude                       */
    int   ground_contact;   /* 1 = downlink active, 0 = no contact            */
    char  mission_phase[32];/* "nominal" | "critical" | "safe_mode"           */
    float min_altitude_km;  /* reentry floor — cannot burn below this         */
    float total_fuel_kg;    /* physical propellant mass for Tsiolkovsky eq     */
} SatState;

/* ── Conjunction data ─────────────────────────────────────────────────────── */
typedef struct {
    char  object_id[32];
    char  object_name[64];
    float miss_km;          /* miss distance at TCA                           */
    float tca_hours;        /* time to closest approach (hours)               */
    float rel_vel[3];       /* relative velocity vector (m/s)                 */
    float rel_pos[3];       /* relative position vector (km)                  */
    int   tle_stale;        /* 1 = TLE older than 48h                         */
    float tle_age_hours;    /* actual TLE age in hours                        */
} Conjunction;

/* ── Assessment output ────────────────────────────────────────────────────── */
typedef struct {
    char       object_id[32];
    float      raw_pc;              /* Pc from LightGBM (unadjusted)          */
    float      adjusted_pc;         /* Pc after all 6 limit adjustments       */
    AlertLevel alert;               /* GREEN / YELLOW / ORANGE / RED          */
    float      dv_vector[3];        /* burn direction + magnitude (m/s)       */
    float      dv_magnitude_ms;     /* scalar ΔV in m/s                       */
    float      fuel_cost_pct;       /* % fuel this burn will consume          */
    int        post_path_safe;      /* 1 = post-maneuver trajectory is clear  */
    char       decision[256];       /* human-readable decision text           */
    char       limitations[512];    /* triggered limitation descriptions      */
} Assessment;

/* ── Threshold struct ─────────────────────────────────────────────────────── */
typedef struct {
    float yellow;
    float orange;
    float red;
} Thresholds;

/* ═══════════════════════════════════════════════════════════════════════════
 * INTERNAL HELPERS
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Vector operations */
static float vec3_norm(const float v[3]) {
    return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

static void vec3_normalize(float out[3], const float v[3]) {
    float n = vec3_norm(v);
    if (n < 1e-10f) { out[0]=1.0f; out[1]=0.0f; out[2]=0.0f; return; }
    out[0] = v[0]/n; out[1] = v[1]/n; out[2] = v[2]/n;
}

static void vec3_cross(float out[3], const float a[3], const float b[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static void vec3_scale(float out[3], const float v[3], float s) {
    out[0]=v[0]*s; out[1]=v[1]*s; out[2]=v[2]*s;
}

/*
 * _fuel_thresholds()
 * Mirrors RiskScorer._fuel_thresholds() in Python exactly.
 * As fuel drops, thresholds are raised — conserve propellant for worse threats.
 */
static Thresholds _fuel_thresholds(float fuel_pct) {
    Thresholds t;
    if      (fuel_pct > 50.0f) { t.yellow=1e-5f; t.orange=1e-4f; t.red=1e-3f; }
    else if (fuel_pct > 30.0f) { t.yellow=1e-5f; t.orange=3e-4f; t.red=3e-3f; }
    else if (fuel_pct > 15.0f) { t.yellow=5e-5f; t.orange=1e-3f; t.red=8e-3f; }
    else                        { t.yellow=1e-4f; t.orange=5e-3f; t.red=5e-2f; }
    return t;
}

/*
 * _classify()
 * Maps adjusted Pc to alert level.
 */
static AlertLevel _classify(float pc, Thresholds t) {
    if      (pc >= t.red)    return ALERT_RED;
    else if (pc >= t.orange) return ALERT_ORANGE;
    else if (pc >= t.yellow) return ALERT_YELLOW;
    else                     return ALERT_GREEN;
}

/*
 * _plan_maneuver()
 * Mirrors RiskScorer._plan_maneuver() exactly.
 * Computes minimum-ΔV burn perpendicular to relative velocity.
 * Uses Tsiolkovsky rocket equation for fuel cost.
 */
static void _plan_maneuver(
    const Conjunction* conj,
    const SatState*    sat,
    int                downward_ok,
    float              dv_out[3],
    float*             dv_mag_out,
    float*             fuel_cost_out)
{
    /* No maneuver needed if miss distance already safe */
    if (conj->miss_km >= 5.0f) {
        dv_out[0]=0; dv_out[1]=0; dv_out[2]=0;
        *dv_mag_out   = 0.0f;
        *fuel_cost_out = 0.0f;
        return;
    }

    float tca = conj->tca_hours > 0.01f ? conj->tca_hours : 0.01f;

    /* Required ΔV to achieve 5km miss distance (km/s → m/s conversion) */
    float required_kms = (5.0f - conj->miss_km) / (tca * 3600.0f);

    /* Perpendicular direction to relative velocity — maximises miss gain */
    float vel_unit[3], radial[3] = {0.0f, 0.0f, 1.0f}, perp[3];
    vec3_normalize(vel_unit, conj->rel_vel);
    vec3_cross(perp, vel_unit, radial);

    float perp_mag = vec3_norm(perp);
    if (perp_mag < 1e-10f) {
        perp[0]=1.0f; perp[1]=0.0f; perp[2]=0.0f;
    } else {
        perp[0]/=perp_mag; perp[1]/=perp_mag; perp[2]/=perp_mag;
    }

    /* Altitude floor constraint — no downward Z if near reentry altitude */
    if (!downward_ok) {
        if (perp[2] < 0.0f) perp[2] = -perp[2];   /* force upward */
        float renorm = vec3_norm(perp);
        if (renorm > 1e-10f) {
            perp[0]/=renorm; perp[1]/=renorm; perp[2]/=renorm;
        }
    }

    /* Convert km/s → m/s and build ΔV vector */
    float dv_vec[3];
    vec3_scale(dv_vec, perp, required_kms * 1000.0f);

    /* Cap ΔV by fuel budget */
    float max_dv  = sat->fuel_pct * 0.5f;
    float dv_mag  = vec3_norm(dv_vec);
    if (dv_mag > max_dv) dv_mag = max_dv;
    vec3_scale(dv_out, perp, dv_mag);

    /* Tsiolkovsky rocket equation: Δm = m × (1 - e^(-ΔV / (Isp × g0))) */
    float Isp = 220.0f;    /* cold gas thruster (s)  */
    float g0  = 9.807f;    /* standard gravity (m/s²) */
    float dm  = sat->total_fuel_kg * (1.0f - expf(-dv_mag / (Isp * g0)));
    *fuel_cost_out = (dm / sat->total_fuel_kg) * 100.0f;
    *dv_mag_out    = dv_mag;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC API
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * acas_assess()
 * ==============
 * Main function. Mirrors RiskScorer.assess() exactly.
 *
 * Takes raw Pc from LightGBM inference, applies all 6 operational limits,
 * classifies alert level, plans maneuver, fills Assessment struct.
 *
 * Parameters:
 *   conj           — conjunction event (from conjunction finder)
 *   raw_pc         — output of LightGBM inference engine
 *   sat            — current satellite health state
 *   post_path_safe — 1 if post-maneuver trajectory is clear
 *   out            — Assessment struct to fill (output)
 */
static void acas_assess(
    const Conjunction* conj,
    float              raw_pc,
    const SatState*    sat,
    int                post_path_safe,
    Assessment*        out)
{
    float pc = raw_pc;
    char  lim_buf[512] = "";   /* accumulate limitation messages */

    /* ── LIMITATION 1: TLE Staleness ──────────────────────────────────────
     * Old tracking data = real position unknown = treat as more dangerous  */
    if (conj->tle_stale) {
        float age_h    = conj->tle_age_hours > 0 ? conj->tle_age_hours : 48.0f;
        float inflation = age_h / 24.0f;
        if (inflation > 5.0f) inflation = 5.0f;   /* cap at 5x             */
        pc *= (1.0f + inflation);
        char buf[128];
        snprintf(buf, sizeof(buf),
            "TLE %.0fh old -> Pc inflated %.1fx; ",
            age_h, 1.0f + inflation);
        strncat(lim_buf, buf, sizeof(lim_buf) - strlen(lim_buf) - 1);
    }

    /* ── LIMITATION 2: Fuel Level ─────────────────────────────────────────
     * Low fuel -> raise alert thresholds                                   */
    Thresholds thresholds = _fuel_thresholds(sat->fuel_pct);
    if      (sat->fuel_pct < 5.0f)
        strncat(lim_buf, "CRITICAL FUEL: threshold 50x; ",
                sizeof(lim_buf)-strlen(lim_buf)-1);
    else if (sat->fuel_pct < 15.0f)
        strncat(lim_buf, "LOW FUEL: threshold 8x; ",
                sizeof(lim_buf)-strlen(lim_buf)-1);
    else if (sat->fuel_pct < 30.0f)
        strncat(lim_buf, "MODERATE FUEL: threshold 3x; ",
                sizeof(lim_buf)-strlen(lim_buf)-1);

    /* ── LIMITATION 3: Battery / Power ───────────────────────────────────
     * Low battery = reduced thrust, boost sensitivity                     */
    if (sat->battery_pct < 20.0f) {
        pc *= 1.5f;
        strncat(lim_buf, "LOW BATTERY: sensitivity 1.5x; ",
                sizeof(lim_buf)-strlen(lim_buf)-1);
    }

    /* ── LIMITATION 4: Altitude Floor ────────────────────────────────────
     * Cannot burn downward if near reentry altitude                       */
    float altitude_margin = sat->altitude_km - sat->min_altitude_km;
    int   downward_ok     = (altitude_margin > 20.0f);
    if (!downward_ok)
        strncat(lim_buf, "ALT MARGIN LOW: downward burns disabled; ",
                sizeof(lim_buf)-strlen(lim_buf)-1);

    /* ── LIMITATION 5: Post-Maneuver Path ────────────────────────────────
     * Maneuver that creates new conjunction is twice as risky             */
    if (!post_path_safe) {
        pc *= 2.0f;
        strncat(lim_buf, "POST-MANEUVER PATH UNSAFE: Pc doubled; ",
                sizeof(lim_buf)-strlen(lim_buf)-1);
    }

    /* ── LIMITATION 6: Mission Phase ──────────────────────────────────── */
    if (strncmp(sat->mission_phase, "critical", 8) == 0) {
        pc *= 1.5f;
        strncat(lim_buf, "CRITICAL PHASE: sensitivity 1.5x; ",
                sizeof(lim_buf)-strlen(lim_buf)-1);
    } else if (strncmp(sat->mission_phase, "safe_mode", 9) == 0) {
        strncat(lim_buf, "SAFE MODE: min essential burns only; ",
                sizeof(lim_buf)-strlen(lim_buf)-1);
    }

    /* ── CLASSIFY ─────────────────────────────────────────────────────── */
    AlertLevel alert = _classify(pc, thresholds);

    /* ── PLAN MANEUVER ───────────────────────────────────────────────── */
    float dv_vec[3], dv_mag, fuel_cost;
    _plan_maneuver(conj, sat, downward_ok, dv_vec, &dv_mag, &fuel_cost);

    /* ── BUILD DECISION TEXT ─────────────────────────────────────────── */
    char dec[256];
    switch (alert) {
        case ALERT_GREEN:
            snprintf(dec, sizeof(dec),
                "No action required. Continuing nominal monitoring.");
            break;
        case ALERT_YELLOW:
            snprintf(dec, sizeof(dec),
                "Conjunction detected. Alert downlinked. Monitoring increased.");
            break;
        case ALERT_ORANGE:
            if (sat->ground_contact)
                snprintf(dec, sizeof(dec),
                    "dV=%.2f m/s fuel=%.2f%% - Awaiting ground approval.",
                    dv_mag, fuel_cost);
            else
                snprintf(dec, sizeof(dec),
                    "dV=%.2f m/s fuel=%.2f%% - No contact. "
                    "%s", dv_mag, fuel_cost,
                    conj->tca_hours < 2.0f
                        ? "TCA<2h: EXECUTING AUTONOMOUSLY."
                        : "Maneuver queued.");
            break;
        case ALERT_RED:
            if (sat->ground_contact)
                snprintf(dec, sizeof(dec),
                    "COLLISION RISK HIGH. dV=%.2f m/s. "
                    "Executing with ground confirmation.", dv_mag);
            else
                snprintf(dec, sizeof(dec),
                    "COLLISION RISK HIGH. dV=%.2f m/s. "
                    "NO GROUND CONTACT - AUTONOMOUS EXECUTION. "
                    "Logged to black box.", dv_mag);
            break;
    }

    /* ── FILL OUTPUT STRUCT ──────────────────────────────────────────── */
    strncpy(out->object_id, conj->object_id, sizeof(out->object_id)-1);
    out->raw_pc         = raw_pc;
    out->adjusted_pc    = pc;
    out->alert          = alert;
    out->dv_vector[0]   = dv_vec[0];
    out->dv_vector[1]   = dv_vec[1];
    out->dv_vector[2]   = dv_vec[2];
    out->dv_magnitude_ms = dv_mag;
    out->fuel_cost_pct  = fuel_cost;
    out->post_path_safe = post_path_safe;
    strncpy(out->decision,    dec,     sizeof(out->decision)-1);
    strncpy(out->limitations, lim_buf, sizeof(out->limitations)-1);
}

/*
 * acas_alert_name()
 * Returns string name of alert level.
 */
static const char* acas_alert_name(AlertLevel a) {
    return ALERT_NAMES[(int)a];
}

#endif /* RISK_SCORER_H */