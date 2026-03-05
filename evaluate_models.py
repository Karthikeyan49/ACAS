#!/usr/bin/env python3
# =============================================================================
# evaluate_models.py  —  ACAS Model Performance Evaluation
#
# Generates synthetic conjunction data, runs it through:
#   1. ConjunctionNet (ONNX)  — Pc prediction accuracy
#   2. RiskScorer             — Alert level classification accuracy
#   3. RL / Geometric burn    — ΔV direction quality
#
# USAGE:
#   python evaluate_models.py            # full evaluation (1000 samples)
#   python evaluate_models.py --n 5000   # more samples
#   python evaluate_models.py --quick    # 200 samples, fast
# =============================================================================

import os, sys, argparse, json
import numpy as np
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.risk_scorer import RiskScorer, SatState, Alert

def extract_features(c: dict):
    rp  = c['rel_pos'];  rv = c['rel_vel']
    md  = c['miss_km'];  tca = c['tca_hours']
    spd = np.linalg.norm(rv)
    rp_norm = np.linalg.norm(rp)
    angle   = np.dot(rp, rv) / (rp_norm * spd + 1e-10)
    danger_time = md / (spd + 1e-10)
    return np.array([rp[0],rp[1],rp[2],rv[0],rv[1],rv[2],
                     md,tca,spd,angle,float(c.get('tle_stale',0)),danger_time],
                    dtype=np.float32)

ONNX_PATH = os.path.join(ROOT, "trained_models", "conjunction_model.onnx")
RL_PATH   = os.path.join(ROOT, "trained_models", "maneuver_policy")

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────────────────
G  = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
C  = "\033[96m"; W = "\033[97m"; D = "\033[90m"; X = "\033[0m"
B  = "\033[1m";  O = "\033[38;5;208m"

def hdr(s):  print(f"\n{B}{C}{'─'*64}\n  {s}\n{'─'*64}{X}")
def ok(s):   print(f"  {G}✅  {s}{X}")
def warn(s): print(f"  {Y}⚠️   {s}{X}")
def err(s):  print(f"  {R}❌  {s}{X}")
def info(s): print(f"  {D}{s}{X}")

# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────
def load_onnx():
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
        ok(f"ONNX model loaded: {ONNX_PATH}")
        return sess
    except Exception as e:
        warn(f"ONNX not available ({e}) — will use physics fallback for Pc")
        return None

def load_rl():
    try:
        from stable_baselines3 import PPO
        agent = PPO.load(RL_PATH)
        ok(f"RL agent loaded: {RL_PATH}")
        return agent
    except Exception as e:
        warn(f"RL agent not available ({e}) — will use geometric fallback for ΔV")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# Mirrors generate_training_data() in conjunction_net.py but adds labels
# for alert level and expected action so we can measure accuracy end-to-end.
# ─────────────────────────────────────────────────────────────────────────────
def generate_test_cases(n: int, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    cases = []

    for _ in range(n):
        miss   = rng.exponential(2.5)                       # km
        tca    = rng.uniform(0.3, 72.0)                     # hours
        speed  = rng.uniform(0.5, 15.0)                     # km/s
        stale  = bool(rng.choice([False, True], p=[0.85, 0.15]))
        tle_age= rng.uniform(1, 100) if stale else rng.uniform(1, 48)

        rp_dir = rng.standard_normal(3)
        rp_dir /= np.linalg.norm(rp_dir)
        rp = rp_dir * miss

        rv_dir = rng.standard_normal(3)
        rv_dir /= np.linalg.norm(rv_dir)
        rv = rv_dir * speed

        # Physics-based TRUE Pc (same formula used to train the net)
        true_pc = min((0.01 / (miss + 1e-10))**2 * speed / 7.8, 1.0)
        if stale:
            true_pc = min(true_pc * 2.5, 1.0)

        # TRUE alert level from physics Pc
        if   true_pc >= 1e-3: true_alert = Alert.RED
        elif true_pc >= 1e-4: true_alert = Alert.ORANGE
        elif true_pc >= 1e-5: true_alert = Alert.YELLOW
        else:                 true_alert = Alert.GREEN

        # Expected action (simplified ground truth)
        if true_alert == Alert.RED:
            true_action = "BURN"
        elif true_alert == Alert.ORANGE and tca < 2.0:
            true_action = "BURN"
        elif true_alert == Alert.ORANGE:
            true_action = "QUEUE_OR_DOWNLINK"
        elif true_alert == Alert.YELLOW:
            true_action = "MONITOR"
        else:
            true_action = "NO_ACTION"

        fuel    = rng.uniform(15, 95)
        battery = rng.uniform(20, 100)
        alt     = rng.uniform(400, 900)

        cases.append({
            "conj": {
                "object_id":    f"TEST-{_:05d}",
                "object_name":  f"DEBRIS-{_:05d}",
                "object_type":  "DEBRIS",
                "miss_km":      miss,
                "tca_hours":    tca,
                "rel_pos":      rp,
                "rel_vel":      rv,
                "rel_speed_kms": speed,
                "tle_stale":    stale,
                "tle_age_hours": tle_age,
            },
            "sat": SatState(
                fuel_pct=fuel, battery_pct=battery, altitude_km=alt,
                ground_contact=bool(rng.choice([True, False], p=[0.7, 0.3])),
                mission_phase="nominal", min_altitude_km=300., total_fuel_kg=2.
            ),
            "true_pc":     true_pc,
            "true_alert":  true_alert,
            "true_action": true_action,
        })

    return cases


# ─────────────────────────────────────────────────────────────────────────────
# Pc prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict_pc(conj, onnx_sess):
    feats = extract_features(conj)
    if onnx_sess is not None:
        inp = feats.reshape(1, -1).astype(np.float32)
        out = onnx_sess.run(None, {"features": inp})
        return float(out[0][0][0]), "ONNX"
    else:
        miss, spd = float(feats[6]), float(feats[8])
        pc = min((0.01 / (miss + 1e-10))**2 * spd / 7.8, 1.0)
        return pc, "PHYSICS"


# ─────────────────────────────────────────────────────────────────────────────
# Pc → alert bucket (direct from Pc value, no satstate adjustment)
# ─────────────────────────────────────────────────────────────────────────────
def pc_to_alert(pc):
    if   pc >= 1e-3: return Alert.RED
    elif pc >= 1e-4: return Alert.ORANGE
    elif pc >= 1e-5: return Alert.YELLOW
    else:            return Alert.GREEN


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate Pc model
# ─────────────────────────────────────────────────────────────────────────────
def eval_pc_model(cases, onnx_sess):
    hdr("1 / 3 — Pc Prediction (ConjunctionNet)")

    true_pcs, pred_pcs = [], []
    abs_errs, rel_errs = [], []
    alert_true, alert_pred = [], []
    correct_alert = 0

    for tc in cases:
        tp  = tc["true_pc"]
        pp, _ = predict_pc(tc["conj"], onnx_sess)
        true_pcs.append(tp)
        pred_pcs.append(pp)

        ae = abs(pp - tp)
        re = ae / (tp + 1e-12)
        abs_errs.append(ae)
        rel_errs.append(min(re, 10.0))   # cap at 10× for display

        ta = tc["true_alert"]
        pa = pc_to_alert(pp)
        alert_true.append(ta)
        alert_pred.append(pa)
        if ta == pa:
            correct_alert += 1

    tp_arr = np.array(true_pcs)
    pp_arr = np.array(pred_pcs)
    ae_arr = np.array(abs_errs)
    re_arr = np.array(rel_errs)

    # Log-space correlation
    log_true = np.log10(tp_arr + 1e-15)
    log_pred = np.log10(pp_arr + 1e-15)
    corr = float(np.corrcoef(log_true, log_pred)[0, 1])

    # Within-order-of-magnitude accuracy
    within_1_OOM = float(np.mean(re_arr < 10.0) * 100)
    within_factor2 = float(np.mean(ae_arr < tp_arr) * 100)

    alert_acc = correct_alert / len(cases) * 100

    print(f"\n  {B}Pc Regression Metrics:{X}")
    print(f"  {'Samples':<30} {len(cases)}")
    print(f"  {'Mean Abs Error':<30} {np.mean(ae_arr):.6f}")
    print(f"  {'Median Abs Error':<30} {np.median(ae_arr):.6f}")
    print(f"  {'Mean Relative Error':<30} {np.mean(re_arr):.2f}×")
    print(f"  {'Log-space Pearson r':<30} {corr:.4f}")
    print(f"  {'Within 1 OOM (%)':<30} {within_1_OOM:.1f}%")
    print(f"  {'Within factor-of-2 (%)':<30} {within_factor2:.1f}%")

    print(f"\n  {B}Alert Level Classification (from raw Pc):{X}")
    print(f"  {'Alert accuracy':<30} {alert_acc:.1f}%")

    # Per-class breakdown
    ORDER = [Alert.GREEN, Alert.YELLOW, Alert.ORANGE, Alert.RED]
    COLS  = {Alert.GREEN:G, Alert.YELLOW:Y, Alert.ORANGE:O, Alert.RED:R}
    print(f"\n  {'Level':<10} {'True N':>8} {'Predicted':>10} {'Precision':>10} {'Recall':>10}")
    print(f"  {'─'*52}")
    for alv in ORDER:
        t_mask = np.array([a == alv for a in alert_true])
        p_mask = np.array([a == alv for a in alert_pred])
        tn = t_mask.sum()
        pn = p_mask.sum()
        tp_n = (t_mask & p_mask).sum()
        prec = tp_n / (pn + 1e-9) * 100
        rec  = tp_n / (tn + 1e-9) * 100
        col  = COLS[alv]
        print(f"  {col}{alv.value:<10}{X} {tn:>8} {pn:>10} {prec:>9.1f}% {rec:>9.1f}%")

    # Confusion matrix
    print(f"\n  {B}Confusion Matrix (rows=true, cols=pred):{X}")
    header = "           " + "".join(f"{a.value:>10}" for a in ORDER)
    print(f"  {D}{header}{X}")
    for ta in ORDER:
        row = "  " + f"{COLS[ta]}{ta.value:<10}{X}"
        for pa in ORDER:
            count = sum(1 for t,p in zip(alert_true, alert_pred) if t==ta and p==pa)
            is_diag = (ta == pa)
            cell = f"{G if is_diag else D}{count:>10}{X}"
            row += cell
        print(row)

    return {
        "alert_accuracy": alert_acc,
        "log_corr": corr,
        "mean_abs_err": float(np.mean(ae_arr)),
        "within_1_oom": within_1_OOM,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate RiskScorer end-to-end
# ─────────────────────────────────────────────────────────────────────────────
def eval_risk_scorer(cases, onnx_sess, scorer):
    hdr("2 / 3 — RiskScorer End-to-End (Pc → Assessment)")

    correct = 0
    action_correct = 0
    limitation_counts = {}
    dv_mags = []
    fp_burns = 0    # predicted BURN when true_action = NO_ACTION/MONITOR
    fn_burns = 0    # missed BURN when true_action = BURN

    for tc in cases:
        pp, _ = predict_pc(tc["conj"], onnx_sess)
        asm = scorer.assess(tc["conj"], pp, tc["sat"], post_path_safe=True)

        pred_alert  = asm.alert
        true_alert  = tc["true_alert"]
        true_action = tc["true_action"]

        if pred_alert == true_alert:
            correct += 1

        # Action correctness: did we burn when we should / not burn when we shouldn't
        pred_burns = pred_alert in (Alert.RED, Alert.ORANGE)
        true_burns = true_action == "BURN"
        if pred_burns == true_burns:
            action_correct += 1
        if pred_burns and not true_burns:
            fp_burns += 1
        if not pred_burns and true_burns:
            fn_burns += 1

        # Limitations
        for lm in asm.limitations_hit:
            limitation_counts[lm] = limitation_counts.get(lm, 0) + 1

        if asm.dv_magnitude_ms > 0:
            dv_mags.append(asm.dv_magnitude_ms)

    n = len(cases)
    alert_acc   = correct / n * 100
    action_acc  = action_correct / n * 100
    fp_rate     = fp_burns / n * 100
    fn_rate     = fn_burns / n * 100

    print(f"\n  {B}Assessment Accuracy:{X}")
    print(f"  {'Alert level match':<32} {alert_acc:.1f}%")
    print(f"  {'Burn decision match':<32} {action_acc:.1f}%")
    print(f"  {'False-positive burns':<32} {fp_rate:.1f}%  ({fp_burns} cases)")
    print(f"  {'False-negative burns':<32} {fn_rate:.1f}%  (missed {fn_burns} cases)")

    if dv_mags:
        dv_arr = np.array(dv_mags)
        print(f"\n  {B}ΔV Computed (when alert≥ORANGE):{X}")
        print(f"  {'Count':<30} {len(dv_mags)}")
        print(f"  {'Min ΔV':<30} {dv_arr.min():.4f} m/s")
        print(f"  {'Max ΔV':<30} {dv_arr.max():.4f} m/s")
        print(f"  {'Mean ΔV':<30} {dv_arr.mean():.4f} m/s")
        print(f"  {'Median ΔV':<30} {np.median(dv_arr):.4f} m/s")
        print(f"  {'P95 ΔV':<30} {np.percentile(dv_arr,95):.4f} m/s")

    if limitation_counts:
        print(f"\n  {B}Limitations Triggered:{X}")
        for k, v in sorted(limitation_counts.items(), key=lambda x: -x[1]):
            pct = v/n*100
            bar = "█" * int(pct/2)
            print(f"  {Y}  {k:<25}{X}  {v:>5}× ({pct:.1f}%)  {D}{bar}{X}")

    return {"alert_acc": alert_acc, "action_acc": action_acc,
            "fp_rate": fp_rate, "fn_rate": fn_rate}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate burn quality (RL agent vs geometric fallback)
# ─────────────────────────────────────────────────────────────────────────────
def eval_burn_quality(cases, rl_agent, scorer, onnx_sess):
    hdr("3 / 3 — Burn Quality (RL Agent vs Geometric Fallback)")

    # Only use cases where a burn is warranted (true_action == BURN)
    burn_cases = [tc for tc in cases if tc["true_action"] == "BURN"]
    if not burn_cases:
        warn("No BURN cases in this sample set — try more samples")
        return {}

    print(f"  Testing on {len(burn_cases)} cases where BURN was warranted\n")

    rl_dvs, geo_dvs = [], []
    rl_alignments, geo_alignments = [], []
    fuel_costs_rl, fuel_costs_geo = [], []

    for tc in burn_cases:
        c   = tc["conj"]
        sat = tc["sat"]

        rv = c["rel_vel"]
        rv_unit = rv / (np.linalg.norm(rv) + 1e-10)

        # Geometric fallback DV
        per = np.cross(rv_unit, [0., 0., 1.])
        if np.linalg.norm(per) < 1e-10: per = np.array([1., 0., 0.])
        else: per /= np.linalg.norm(per)
        ms = min((5. - c["miss_km"]) / (max(c["tca_hours"], .01) * 3.6) * 1000., sat.fuel_pct * .5)
        geo_dv = per * ms
        geo_dvs.append(np.linalg.norm(geo_dv))

        # Alignment score: cos(angle between DV and avoidance direction)
        # Perfect avoidance = DV perpendicular to relative velocity (angle=90° → cos=0)
        # We measure alignment with the perpendicular direction
        if np.linalg.norm(geo_dv) > 0:
            geo_unit = geo_dv / np.linalg.norm(geo_dv)
            geo_align = float(abs(np.dot(geo_unit, per)))  # 1=aligned with avoid dir
            geo_alignments.append(geo_align)

        # Fuel cost (Tsiolkovsky)
        import math
        geo_fc = (2.*(1-math.exp(-np.linalg.norm(geo_dv)/(220.*9.807)))/2.)*100.
        fuel_costs_geo.append(geo_fc)

        # RL agent DV (if available)
        if rl_agent is not None:
            obs = np.array([
                *c["rel_pos"], *c["rel_vel"],
                sat.fuel_pct, sat.battery_pct,
                c["tca_hours"], sat.altitude_km - sat.min_altitude_km
            ], dtype=np.float32)
            dv_rl, _ = rl_agent.predict(obs, deterministic=True)
            dv_rl = np.array(dv_rl)
            rl_dvs.append(np.linalg.norm(dv_rl))

            if np.linalg.norm(dv_rl) > 0:
                rl_unit = dv_rl / np.linalg.norm(dv_rl)
                rl_align = float(abs(np.dot(rl_unit, per)))
                rl_alignments.append(rl_align)

            rl_fc = (2.*(1-math.exp(-np.linalg.norm(dv_rl)/(220.*9.807)))/2.)*100.
            fuel_costs_rl.append(rl_fc)

    # Report
    if rl_agent:
        print(f"  {B}{'Metric':<30} {'RL Agent':>12} {'Geometric':>12}{X}")
        print(f"  {'─'*56}")
        print(f"  {'Mean |ΔV| (m/s)':<30} {np.mean(rl_dvs):>11.4f}  {np.mean(geo_dvs):>11.4f}")
        print(f"  {'Median |ΔV| (m/s)':<30} {np.median(rl_dvs):>11.4f}  {np.median(geo_dvs):>11.4f}")
        if rl_alignments:
            print(f"  {'Mean alignment (0–1)':<30} {np.mean(rl_alignments):>11.4f}  {np.mean(geo_alignments):>11.4f}")
        print(f"  {'Mean fuel cost (%)':<30} {np.mean(fuel_costs_rl):>11.4f}  {np.mean(fuel_costs_geo):>11.4f}")
        print(f"  {'P95 fuel cost (%)':<30} {np.percentile(fuel_costs_rl,95):>11.4f}  {np.percentile(fuel_costs_geo,95):>11.4f}")

        rl_cheaper = np.mean(np.array(fuel_costs_rl) < np.array(fuel_costs_geo)) * 100
        print(f"\n  {G if rl_cheaper>50 else Y}RL burns less fuel than geometric in {rl_cheaper:.1f}% of cases{X}")
    else:
        warn("RL agent not loaded — showing geometric fallback only")
        print(f"\n  {B}Geometric Fallback:{X}")
        print(f"  {'Mean |ΔV| (m/s)':<30} {np.mean(geo_dvs):.4f}")
        print(f"  {'Median |ΔV| (m/s)':<30} {np.median(geo_dvs):.4f}")
        if geo_alignments:
            print(f"  {'Mean alignment':<30} {np.mean(geo_alignments):.4f}")
        print(f"  {'Mean fuel cost (%)':<30} {np.mean(fuel_costs_geo):.4f}")

    return {"geo_mean_dv": float(np.mean(geo_dvs)),
            "rl_mean_dv":  float(np.mean(rl_dvs)) if rl_dvs else None}


# ─────────────────────────────────────────────────────────────────────────────
# Edge case tests — specific known scenarios
# ─────────────────────────────────────────────────────────────────────────────
def run_edge_cases(onnx_sess, scorer):
    hdr("EDGE CASES — known-answer tests")

    EDGE = [
        {"name":"Near-miss head-on fast",
         "conj":{"object_id":"E1","object_name":"EC-1","object_type":"DEBRIS",
                 "miss_km":0.1,"tca_hours":0.5,
                 "rel_pos":np.array([0.06,-0.05,0.02]),
                 "rel_vel":np.array([-14.,6.,2.5]),"rel_speed_kms":15.4,
                 "tle_stale":False,"tle_age_hours":4.},
         "expected_alert": Alert.RED},
        {"name":"Large miss slow approach",
         "conj":{"object_id":"E2","object_name":"EC-2","object_type":"DEBRIS",
                 "miss_km":5.0,"tca_hours":36.,
                 "rel_pos":np.array([3.,-2.5,0.8]),
                 "rel_vel":np.array([-0.5,0.3,0.1]),"rel_speed_kms":0.6,
                 "tle_stale":False,"tle_age_hours":6.},
         "expected_alert": Alert.GREEN},
        {"name":"Stale TLE 72h moderate miss",
         "conj":{"object_id":"E3","object_name":"EC-3","object_type":"DEBRIS",
                 "miss_km":1.5,"tca_hours":8.,
                 "rel_pos":np.array([0.9,-0.75,0.25]),
                 "rel_vel":np.array([-7.,3.,1.2]),"rel_speed_kms":7.8,
                 "tle_stale":True,"tle_age_hours":72.},
         "expected_alert": Alert.ORANGE},
        {"name":"Very close imminent collision",
         "conj":{"object_id":"E4","object_name":"EC-4","object_type":"DEBRIS",
                 "miss_km":0.05,"tca_hours":0.2,
                 "rel_pos":np.array([0.03,-0.025,0.01]),
                 "rel_vel":np.array([-12.,5.,2.]),"rel_speed_kms":13.1,
                 "tle_stale":False,"tle_age_hours":2.},
         "expected_alert": Alert.RED},
        {"name":"Borderline YELLOW/ORANGE",
         "conj":{"object_id":"E5","object_name":"EC-5","object_type":"DEBRIS",
                 "miss_km":2.0,"tca_hours":12.,
                 "rel_pos":np.array([1.2,-1.0,0.3]),
                 "rel_vel":np.array([-4.,2.,0.8]),"rel_speed_kms":4.5,
                 "tle_stale":False,"tle_age_hours":10.},
         "expected_alert": Alert.YELLOW},
    ]

    sat = SatState(fuel_pct=80., battery_pct=90., altitude_km=550.,
                   ground_contact=True, mission_phase="nominal",
                   min_altitude_km=300., total_fuel_kg=2.)

    passed = 0
    for ec in EDGE:
        pp, mth = predict_pc(ec["conj"], onnx_sess)
        asm = scorer.assess(ec["conj"], pp, sat, post_path_safe=True)
        match = asm.alert == ec["expected_alert"]
        status = f"{G}PASS{X}" if match else f"{R}FAIL{X}"
        if match: passed += 1
        print(f"  [{status}] {ec['name']}")
        print(f"         Pc={pp:.2e}  adjPc={asm.adjusted_pc:.2e}  "
              f"predicted={asm.alert.value}  expected={ec['expected_alert'].value}  "
              f"({mth})")
        if asm.limitations_hit:
            print(f"         {D}limits: {', '.join(asm.limitations_hit)}{X}")

    print(f"\n  {B}Edge case result: {passed}/{len(EDGE)} passed{X}")
    return passed, len(EDGE)


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(n, pc_res, rs_res, ec_pass, ec_total, onnx_ok, rl_ok, elapsed):
    hdr("SUMMARY REPORT")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"  {D}Generated: {ts}{X}")
    print(f"  Test samples: {n}    ONNX: {'loaded' if onnx_ok else 'fallback'}    RL: {'loaded' if rl_ok else 'fallback'}")
    print()

    rows = [
        ("Pc alert accuracy (raw Pc→bucket)",  pc_res.get("alert_accuracy",0),   85, 95),
        ("Log-space Pearson correlation",       pc_res.get("log_corr",0)*100,     70, 90),
        ("Within 1 order-of-magnitude",         pc_res.get("within_1_oom",0),     80, 95),
        ("Risk scorer alert match",             rs_res.get("alert_acc",0),        80, 95),
        ("Burn decision accuracy",              rs_res.get("action_acc",0),       85, 95),
        ("False-positive burn rate",            100-rs_res.get("fp_rate",0),      90,100),
        ("False-negative burn rate (0=bad)",    100-rs_res.get("fn_rate",0),      85,100),
        ("Edge cases passed",                   ec_pass/ec_total*100,             80,100),
    ]

    print(f"  {B}{'Metric':<42} {'Score':>8}  {'Status'}{X}")
    print(f"  {'─'*65}")
    overall_score = 0
    for label, val, warn_thr, good_thr in rows:
        if val >= good_thr:   col, sym = G, "✅ GOOD"
        elif val >= warn_thr: col, sym = Y, "⚠️  OK  "
        else:                 col, sym = R, "❌ LOW  "
        print(f"  {label:<42} {col}{val:>7.1f}%{X}  {sym}")
        overall_score += val / len(rows)

    print(f"\n  {'─'*65}")
    if   overall_score >= 90: oc,os = G, "EXCELLENT"
    elif overall_score >= 75: oc,os = Y, "GOOD"
    else:                     oc,os = R, "NEEDS IMPROVEMENT"
    print(f"  {B}Overall score: {oc}{overall_score:.1f}%  →  {os}{X}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Save JSON report
# ─────────────────────────────────────────────────────────────────────────────
def save_report(n, pc_res, rs_res, ec_pass, ec_total):
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_samples": n,
        "pc_model": pc_res,
        "risk_scorer": rs_res,
        "edge_cases": {"passed": ec_pass, "total": ec_total,
                       "pct": round(ec_pass/ec_total*100, 1)},
    }
    path = os.path.join(ROOT, "evaluation_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  {D}Full report saved → {path}{X}\n")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ACAS Model Evaluator")
    parser.add_argument("--n",     type=int, default=1000, help="Number of test samples (default 1000)")
    parser.add_argument("--seed",  type=int, default=42,   help="Random seed")
    parser.add_argument("--quick", action="store_true",    help="Fast run with 200 samples")
    args = parser.parse_args()

    if args.quick:
        args.n = 200

    import time
    t_start = time.time()

    print(f"\n{B}{C}╔══════════════════════════════════════════════════════════════╗")
    print(f"║     ACAS  MODEL  PERFORMANCE  EVALUATION                     ║")
    print(f"╚══════════════════════════════════════════════════════════════╝{X}")
    print(f"  Samples: {args.n}   Seed: {args.seed}\n")

    # Load
    hdr("Loading Models")
    onnx_sess = load_onnx()
    rl_agent  = load_rl()
    scorer    = RiskScorer()
    ok("RiskScorer initialised")

    # Generate
    hdr("Generating Synthetic Test Data")
    print(f"  Generating {args.n} synthetic conjunction scenarios...")
    cases = generate_test_cases(args.n, args.seed)

    # Alert distribution
    dist = {}
    for tc in cases:
        k = tc["true_alert"].value
        dist[k] = dist.get(k, 0) + 1
    print(f"  Distribution: " +
          "  ".join(f"{k}={v}({v/args.n*100:.0f}%)" for k, v in dist.items()))

    # Evaluate
    pc_res  = eval_pc_model(cases, onnx_sess)
    rs_res  = eval_risk_scorer(cases, onnx_sess, scorer)
    _       = eval_burn_quality(cases, rl_agent, scorer, onnx_sess)
    ec_pass, ec_total = run_edge_cases(onnx_sess, scorer)

    elapsed = time.time() - t_start
    print_summary(args.n, pc_res, rs_res, ec_pass, ec_total,
                  onnx_sess is not None, rl_agent is not None, elapsed)
    save_report(args.n, pc_res, rs_res, ec_pass, ec_total)


if __name__ == "__main__":
    main()