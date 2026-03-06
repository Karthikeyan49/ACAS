# ============================================================
# dashboard/app.py  — ACAS POWER HOUSE  v4  (complete rewrite)
#
# LAYOUT
# ┌─────────────────────┬─────────────────────────────────────────────────────┐
# │   LEFT SIDEBAR      │  MAIN AREA                                          │
# │  ─────────────────  │  [Mode badge] [10 metric cards] [Threat count]      │
# │  Model status       │  ┌──────────────────────┬──────────────────────┐   │
# │  ─────────────────  │  │  Three.js 3D Globe   │  Burn model output   │   │
# │  §1 ECI pos / vel   │  │  · Earth (stable)    │  Threat scenarios    │   │
# │     (fragment 3s)   │  │  · Satellite (60fps) │  ACAS log            │   │
# │  ─────────────────  │  │  · Debris markers    │                      │   │
# │  §2 Custom threat   │  │  · Burn exhaust      │                      │   │
# │     form (sliders)  │  └──────────────────────┴──────────────────────┘   │
# └─────────────────────┴─────────────────────────────────────────────────────┘
#
# HOW TO RUN
#   Terminal 1:  python satellite_process.py
#   Terminal 2:  streamlit run dashboard/app.py
# ============================================================

import sys, os, json, time, math
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── LightGBM integration (replaces ConjunctionNet ONNX) ──────────────────
from lgbm_inference_engine  import LGBMInferenceEngine
from models.risk_scorer     import RiskScorer, SatState, Alert

MODEL_FILE = os.path.join(ROOT, "satellite_model.json")
ONNX_PATH  = os.path.join(ROOT, "trained_models", "conjunction_model.onnx")
RL_PATH    = os.path.join(ROOT, "trained_models", "maneuver_policy")

# ============================================================
# PAGE CONFIG + GLOBAL CSS
# ============================================================
st.set_page_config(page_title="ACAS – Power House",
                   page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
html,body,.stApp{background:#02040e;color:#c8d8f0;}
.main .block-container{padding-top:.4rem;padding-bottom:.4rem;max-width:100%;}
/* ---- mode badge ---- */
.mode-badge{
  display:inline-block;padding:6px 22px;border-radius:5px;
  font-family:'Share Tech Mono',monospace;font-size:14px;
  font-weight:bold;letter-spacing:3px;border:2px solid;
  margin-bottom:8px;
}
/* ---- metric cards ---- */
.mc-row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;}
.mc{background:#07091a;border:1px solid #0d1430;border-radius:5px;
  padding:7px 12px;flex:1;min-width:90px;}
.mc-label{font-family:'Share Tech Mono',monospace;font-size:8px;
  color:#3a4a66;letter-spacing:2px;text-transform:uppercase;display:block;}
.mc-val{font-family:'Share Tech Mono',monospace;font-size:14px;
  color:#00d4ff;display:block;margin-top:1px;}
.mc-green{color:#00ff88;} .mc-yellow{color:#ffd700;}
.mc-orange{color:#ff8c00;} .mc-red{color:#ff2244;font-weight:bold;}
/* ---- sidebar vec boxes ---- */
.vg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin:4px 0 8px;}
.vc{background:#050710;border:1px solid #0d1430;border-radius:4px;
  padding:5px 4px;text-align:center;}
.va{font-family:'Share Tech Mono',monospace;font-size:8px;
  color:#3a4a66;display:block;letter-spacing:1px;}
.vn{font-family:'Share Tech Mono',monospace;font-size:11px;
  color:#00d4ff;display:block;}
/* ---- gauge ---- */
.g-wrap{margin-bottom:7px;}
.g-head{display:flex;justify-content:space-between;
  font-family:'Share Tech Mono',monospace;font-size:9px;margin-bottom:2px;}
.g-lbl{color:#3a4a66;letter-spacing:1px;} .g-val{color:#c8d8f0;}
.g-bar{height:4px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden;}
.g-fill{height:100%;border-radius:2px;transition:width .6s,background .6s;}
/* ---- pill row ---- */
.pr{display:flex;gap:5px;margin:5px 0;}
.pill{font-family:'Share Tech Mono',monospace;font-size:9px;
  padding:2px 9px;border-radius:9px;letter-spacing:1px;border:1px solid;}
.p-on{background:rgba(0,255,136,.1);color:#00ff88;border-color:rgba(0,255,136,.3);}
.p-off{background:rgba(255,34,68,.1);color:#ff2244;border-color:rgba(255,34,68,.3);}
/* ---- section title ---- */
.sec{font-family:'Share Tech Mono',monospace;font-size:8px;letter-spacing:3px;
  color:#3a4a66;text-transform:uppercase;margin:8px 0 5px;
  border-bottom:1px solid #0d1430;padding-bottom:3px;}
/* ---- right-panel cards ---- */
.rcard{background:#07091a;border:1px solid #0d1430;border-radius:6px;
  padding:10px 14px;margin-bottom:8px;}
.trow{display:flex;justify-content:space-between;align-items:baseline;margin:2px 0;}
.tkey{font-family:'Share Tech Mono',monospace;font-size:10px;color:#3a4a66;}
.tval{font-family:'Share Tech Mono',monospace;font-size:11px;color:#00d4ff;}
.tv-g{color:#00ff88;} .tv-y{color:#ffd700;}
.tv-o{color:#ff8c00;} .tv-r{color:#ff2244;font-weight:bold;}
/* ---- scenario card ---- */
.sc-card{background:#07091a;border:1px solid #0d1430;border-radius:5px;
  padding:6px 10px;margin-bottom:4px;}
/* ---- log ---- */
.mono{font-family:'Share Tech Mono',monospace;font-size:10px;line-height:1.8;}
.lg{color:#00ff88} .ly{color:#ffd700} .lo{color:#ff8c00}
.lr{color:#ff2244;font-weight:bold} .li{color:#88aaff} .lw{color:#778899}
.lb{color:#ff6600;font-weight:bold;background:#1a0800;padding:1px 5px;border-radius:2px}
/* ---- burn banner ---- */
@keyframes bp{0%,100%{box-shadow:0 0 8px #ff4400,0 0 18px #ff6600;}
              50%{box-shadow:0 0 18px #ff2200,0 0 36px #ff8800;}}
.burn-banner{animation:bp .7s infinite;border-radius:6px;padding:7px 15px;
  background:#1a0500;border:2px solid #ff4400;
  font-family:'Share Tech Mono',monospace;margin-bottom:8px;}
/* ---- pipeline ---- */
.ps{background:#080820;border-left:3px solid #2244cc;
  padding:3px 9px;margin:2px 0;border-radius:0 3px 3px 0;
  font-family:'Share Tech Mono',monospace;font-size:10px;}
</style>""", unsafe_allow_html=True)


# ============================================================
# SCENARIOS DATA
# ============================================================
SCENARIOS = [
    {"name":"SENTINEL-2 DEB","norad":"48891","alert":"YELLOW",
     "desc":"Slow fragment — large miss, 28h out",
     "miss_km":3.8,"tca_h":28.0,"rp":[2.28,-1.90,0.30],"rv":[-2.10,0.80,0.30],
     "stale":False,"tle_age":6.0},
    {"name":"SL-8 R/B","norad":"12456","alert":"YELLOW",
     "desc":"Rocket body — moderate miss, plenty of time",
     "miss_km":2.5,"tca_h":18.0,"rp":[1.50,-1.25,0.40],"rv":[-1.50,1.20,0.50],
     "stale":False,"tle_age":10.0},
    {"name":"COSMOS 2360 DEB","norad":"33401","alert":"ORANGE",
     "desc":"5h to TCA — maneuver prep required",
     "miss_km":1.2,"tca_h":5.0,"rp":[0.72,-0.60,0.20],"rv":[-6.50,3.00,1.20],
     "stale":False,"tle_age":14.0},
    {"name":"IRIDIUM 33 DEB","norad":"33738","alert":"ORANGE",
     "desc":"High speed, 3.5h TCA — RL computes optimal ΔV",
     "miss_km":0.8,"tca_h":3.5,"rp":[0.48,-0.40,0.15],"rv":[-5.00,2.50,0.80],
     "stale":False,"tle_age":20.0},
    {"name":"FENGYUN 1C DEB","norad":"28682","alert":"ORANGE",
     "desc":"No ground contact + TCA>2h → QUEUED",
     "miss_km":0.6,"tca_h":2.5,"rp":[0.36,-0.30,0.12],"rv":[-7.20,3.50,1.50],
     "stale":False,"tle_age":8.0},
    {"name":"COSMOS 954 DEB","norad":"10440","alert":"RED",
     "desc":"Critical Pc, 1.2h TCA — immediate burn",
     "miss_km":0.15,"tca_h":1.2,"rp":[0.09,-0.075,0.03],"rv":[-13.50,6.00,2.50],
     "stale":False,"tle_age":4.0},
    {"name":"USA-193 DEB","norad":"29651","alert":"RED",
     "desc":"Stale TLE 72h → Pc×4, 0.5h TCA — AUTONOMOUS",
     "miss_km":0.25,"tca_h":0.5,"rp":[0.15,-0.12,0.05],"rv":[-14.00,7.00,3.00],
     "stale":True,"tle_age":72.0},
    {"name":"NOAA-16 DEB","norad":"26536","alert":"RED",
     "desc":"Head-on approach, 1.8h TCA",
     "miss_km":0.30,"tca_h":1.8,"rp":[0.18,-0.15,0.06],"rv":[-9.00,4.50,2.00],
     "stale":False,"tle_age":5.0},
]
AC = {"GREEN":"#00ff88","YELLOW":"#ffd700","ORANGE":"#ff8c00","RED":"#ff2244"}


# ============================================================
# MODELS (cached once)
# ============================================================
@st.cache_resource
def load_models():
    r = {}
    # ── LightGBM model (replaces ONNX ConjunctionNet) ──────────────────────
    try:
        lgbm_engine = LGBMInferenceEngine()
        r['lgbm']    = lgbm_engine
        r['onnx_ok'] = lgbm_engine.is_loaded
        r['onnx_msg'] = f"✅ LightGBM loaded" if lgbm_engine.is_loaded else "⚠️ LightGBM — physics fallback"
    except Exception as e:
        r['lgbm']    = LGBMInferenceEngine()   # will use physics fallback
        r['onnx_ok'] = False
        r['onnx_msg'] = f"⚠️ LightGBM init error: {e}"
    # ── RL maneuver agent (unchanged) ─────────────────────────────────────
    try:
        from stable_baselines3 import PPO
        r['rl'] = PPO.load(RL_PATH); r['rl_ok'] = True
        r['rl_msg'] = "✅ maneuver_policy.zip loaded"
    except Exception:
        r['rl'] = None; r['rl_ok'] = False
        r['rl_msg'] = "⚠️ RL missing — geometric fallback"
    return r

M = load_models()
scorer = RiskScorer()


# ============================================================
# UTILITIES
# ============================================================
def read_model():
    for _ in range(3):
        try:
            with open(MODEL_FILE) as f:
                d = json.load(f)
            if 'eci_state' in d: return d
        except (json.JSONDecodeError, FileNotFoundError):
            time.sleep(0.04)
    return None

def add_log(msg, css="li"):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    st.session_state.log.insert(0, {'ts': ts, 'msg': msg, 'css': css})
    st.session_state.log = st.session_state.log[:100]


# ============================================================
# PIPELINE
# ============================================================
def predict_pc(conj):
    # ── LightGBM prediction (103 CDM features) ───────────────────────────
    lgbm   = M['lgbm']
    pc     = lgbm.predict_pc_from_conjunction(conj)
    mth    = "LightGBM" if lgbm.is_loaded else "PHYSICS-FALLBACK"
    # Keep feats for pipeline log display (12-element approximation)
    import numpy as _np
    miss   = conj.get('miss_km', 1.0)
    spd    = float(_np.linalg.norm(conj.get('rel_vel', [7.8,0,0])))
    danger = miss / (spd + 1e-10)
    feats  = _np.array([
        *conj.get('rel_pos',[0,0,0]),
        *conj.get('rel_vel',[0,0,0]),
        miss, conj.get('tca_hours',1.0), spd,
        0.0, float(conj.get('tle_stale',0)), danger
    ], dtype=_np.float32)
    return pc, feats, mth

def predict_burn(conj, sat):
    obs = np.array([
        *conj['rel_pos'], *conj['rel_vel'],
        sat.fuel_pct, sat.battery_pct,
        conj['tca_hours'], sat.altitude_km - sat.min_altitude_km
    ], dtype=np.float32)
    if M['rl_ok']:
        dv, _ = M['rl'].predict(obs, deterministic=True)
        mth   = "RL AGENT (maneuver_policy.zip)"
    else:
        rv  = conj['rel_vel']
        vu  = rv / (np.linalg.norm(rv)+1e-10)
        per = np.cross(vu,[0.,0.,1.])
        if np.linalg.norm(per)<1e-10: per=np.array([1.,0.,0.])
        else: per/=np.linalg.norm(per)
        ms  = min((5.-conj['miss_km'])/(max(conj['tca_hours'],.01)*3.6)*1000., sat.fuel_pct*.5)
        dv  = per * ms;  mth = "GEOMETRIC FALLBACK"
    return np.array(dv), mth

def _consume_fuel(dv):
    mag = np.linalg.norm(dv)
    dm  = 2.0*(1 - math.exp(-mag/(220.*9.807)))
    pct = (dm/2.)*100.
    cur = st.session_state.fuel_override
    if cur is None:
        sm = read_model()
        cur = sm['health']['fuel_pct'] if sm else 85.0
    st.session_state.fuel_override = max(0., float(cur)-pct)
    st.session_state.last_fuel_cost = pct

def run_pipeline(conj, sat):
    res = {'conjunction':conj,'log':[],'action':None,
           'burned':False,'dv':None,'dv_method':None,'assessment':None}
    def pl(s,m,c="li"): res['log'].append({'s':s,'m':m,'c':c})

    pc, feats, mth = predict_pc(conj)
    pl(1,f"extract_features() → {feats[:4].round(3).tolist()} …","lw")
    pl(1,f"{mth} → raw_pc={pc:.6f}","li")

    asm = scorer.assess(conj, pc, sat, post_path_safe=True)
    res['assessment'] = asm
    cm={'GREEN':'lg','YELLOW':'ly','ORANGE':'lo','RED':'lr'}
    pl(2,f"RiskScorer → {asm.alert.value} | adj_pc={asm.adjusted_pc:.2e} raw={pc:.2e}",cm[asm.alert.value])
    for lm in asm.limitations_hit: pl(2,f"  ⚡ {lm}","lo")

    a = asm.alert
    if a==Alert.GREEN:
        pl(3,"_act() → GREEN: no action.","lg"); res['action']="NO_ACTION"
    elif a==Alert.YELLOW:
        pl(3,"_act() → YELLOW: logged. Ground notified.","ly"); res['action']="YELLOW_ALERT"
    elif a==Alert.ORANGE:
        dv,rm = predict_burn(conj,sat); res['dv']=dv; res['dv_method']=rm
        pl(3,f"{rm} → ΔV=[{dv[0]:.3f},{dv[1]:.3f},{dv[2]:.3f}] m/s","lo")
        if sat.ground_contact:
            pl(3,"_act() → ORANGE+GROUND: downlink sent.","lo"); res['action']="ORANGE_DOWNLINK"
        elif conj['tca_hours']<2.:
            pl(3,f"_act() → ORANGE+TCA<2h: AUTONOMOUS BURN","lb")
            res['action']="ORANGE_AUTONOMOUS"; res['burned']=True; _consume_fuel(dv)
        else:
            pl(3,f"_act() → ORANGE: queued, auto@TCA<2h.","lo"); res['action']="ORANGE_QUEUED"
    else:
        dv,rm = predict_burn(conj,sat); res['dv']=dv; res['dv_method']=rm
        pl(3,f"{rm} → ΔV=[{dv[0]:.3f},{dv[1]:.3f},{dv[2]:.3f}] m/s","lr")
        if sat.ground_contact:
            pl(3,"_act() → RED+GROUND: burn CONFIRMED.","lr"); res['action']="RED_GROUND"
        else:
            pl(3,"_act() → RED: AUTONOMOUS BURN. Logged.","lb"); res['action']="RED_AUTONOMOUS"
        res['burned']=True; _consume_fuel(dv)
    return res

def post_orbit(pos,vel,dv,n=60):
    nv=vel+dv/1000.; R=np.linalg.norm(pos); pts=[]
    for i in range(n):
        p2=pos+nv*(i*60.); r2=np.linalg.norm(p2)
        if r2>0: p2=p2*(R/r2)
        pts.append(p2.tolist())
    return pts


# ============================================================
# SESSION STATE
# ============================================================
_DEF = dict(objects=[], log=[], cycle=0,
            maneuver_on=False, last_dv=np.zeros(3),
            post_pts=[], burn_pos=[0.,0.,0.],
            fuel_override=None, last_fuel_cost=0.,
            last_result=None)
for k,v in _DEF.items():
    if k not in st.session_state: st.session_state[k]=v


# ============================================================
# READ LIVE DATA
# ============================================================
sm = read_model()
if sm is None:
    st.error("⚠️  **satellite_model.json not found.** Run: `python satellite_process.py`")
    st.stop()

eci  = sm['eci_state'];   hlth = sm['health']
derv = sm['derived_position'];  comm = sm['communications']
env  = sm['environment'];  orb  = sm['orbital_elements']
prop = sm['propulsion'];   mis  = sm['mission']

pos = np.array([eci['pos_x_km'], eci['pos_y_km'], eci['pos_z_km']])
vel = np.array([eci['vel_x_kms'],eci['vel_y_kms'],eci['vel_z_kms']])
alt = derv['altitude_km']; spd = derv['speed_kms']
lat = derv['latitude_deg']; lon = derv['longitude_deg']

fuel_now = (st.session_state.fuel_override
            if st.session_state.fuel_override is not None
            else hlth['fuel_pct'])
bat_now  = hlth['battery_pct']
ground   = comm['ground_contact']
eclipse  = env['in_eclipse']

sat_state = SatState(fuel_pct=fuel_now, battery_pct=bat_now,
                     altitude_km=alt, ground_contact=ground,
                     mission_phase=mis['phase'], min_altitude_km=300., total_fuel_kg=2.)


# ============================================================
# RUN PIPELINE
# ============================================================
all_results = []
for obj in st.session_state.objects:
    r = run_pipeline(obj, sat_state)
    all_results.append(r)

ORDER = [Alert.GREEN,Alert.YELLOW,Alert.ORANGE,Alert.RED]
overall = (max([r['assessment'].alert for r in all_results], key=lambda x:ORDER.index(x))
           if all_results else Alert.GREEN)

# Update burn state when new burn executes
for r in all_results:
    if r['burned'] and r['dv'] is not None:
        if not np.allclose(r['dv'], st.session_state.last_dv):
            st.session_state.maneuver_on = True
            st.session_state.last_dv     = r['dv'].copy()
            st.session_state.burn_pos    = pos.tolist()
            st.session_state.post_pts    = post_orbit(pos, vel, r['dv'])
            st.session_state.last_result = r
            add_log(f"🔥 BURN EXECUTED | ΔV=[{r['dv'][0]:.3f},{r['dv'][1]:.3f},{r['dv'][2]:.3f}] m/s "
                    f"| {r['action']} | fuel_cost={st.session_state.last_fuel_cost:.3f}%","lb")

burn_active = st.session_state.maneuver_on or prop['thruster_active']
dv_ss       = st.session_state.last_dv


# ============================================================
# MISSION MODE
# ============================================================
def mission_mode():
    if burn_active or overall==Alert.RED:
        return "CRITICAL","#ff2244","rgba(255,34,68,.12)","#ff2244"
    if overall in [Alert.ORANGE,Alert.YELLOW] or bat_now<30 or fuel_now<20:
        return "SAFE MODE","#ffd700","rgba(255,215,0,.08)","#ffd700"
    return "NOMINAL","#00ff88","rgba(0,255,136,.07)","#00ff88"

mode_label, mode_col, mode_bg, mode_border = mission_mode()


# ============================================================
# THREE.JS HTML BUILDER
# ============================================================
def build_globe(sm_data, debris_list, burn_active, dv_vec, burn_pos, post_pts):
    # Satellite starts from REAL ECI position in satellite_model.json
    # then Keplerian propagation at SIM_SPD=100x matches satellite_process.py speed.
    orb_  = sm_data['orbital_elements']
    eci_  = sm_data['eci_state']
    sim_t = sm_data['sim_time_seconds']

    debris_js = json.dumps([
        {"n": d['name'], "a": d['alert'],
         "r": [round(x,3) for x in d['rel_pos']]}
        for d in debris_list])
    post_js = json.dumps(
        [[round(p[0],2),round(p[1],2),round(p[2],2)]
         for p in post_pts[::2]] if post_pts else [])
    burn_js = "true"  if burn_active else "false"
    dv_js   = json.dumps([round(x,5) for x in dv_vec])
    bp_js   = json.dumps([round(x,2) for x in burn_pos])

    # Pull real orbital elements so JS seeds from real position
    a_km   = orb_['semi_major_axis_km']
    i_deg  = orb_['inclination_deg']
    raan   = orb_['raan_deg']
    w_deg  = orb_['arg_perigee_deg']
    m0_deg = orb_['mean_anomaly_deg']
    T_s    = orb_['period_min'] * 60.0

    return (
        '<!DOCTYPE html>\n'
        '<html><head><meta charset="utf-8">\n'
        '<style>\n'
        '*{margin:0;padding:0;box-sizing:border-box;}\n'
        'body{background:#02040e;overflow:hidden;}\n'
        '#c{display:block;width:100%;height:100%;cursor:grab;}\n'
        '#c:active{cursor:grabbing;}\n'
        '#hud{position:absolute;bottom:6px;left:8px;\n'
        '  font:9px "Share Tech Mono",monospace;color:#1a2233;letter-spacing:1px;pointer-events:none;}\n'
        '#badge{position:absolute;top:7px;right:7px;\n'
        '  font:bold 10px "Share Tech Mono",monospace;\n'
        '  padding:3px 11px;border-radius:3px;border:1px solid;\n'
        '  letter-spacing:2px;display:none;pointer-events:none;}\n'
        '</style></head><body>\n'
        '<canvas id="c"></canvas>\n'
        '<div id="hud">Drag · Scroll zoom · real ECI seed</div>\n'
        '<div id="badge"></div>\n'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>\n'
        '<script>\n'
        # ── Python-injected values ──
        f'const SAT_A={a_km};\n'
        f'const SAT_I={i_deg}*Math.PI/180;\n'
        f'const SAT_RAAN={raan}*Math.PI/180;\n'
        f'const SAT_W={w_deg}*Math.PI/180;\n'
        f'const SAT_M0={m0_deg}*Math.PI/180;\n'  # real mean anomaly from file
        f'const SAT_T={T_s};\n'                   # period in seconds
        f'const SIM_T0={sim_t};\n'
        'const SIM_SPD=100;\n'                    # matches satellite_process.py speed
        f'const BURN={burn_js};\n'
        f'const DV={dv_js};\n'    # m/s ECI — RL/geometric model output
        f'const BP={bp_js};\n'    # km  ECI — where burn was executed
        f'const POST={post_js};\n'
        f'const DEBRIS={debris_js};\n'
        'const GM=398600.4418,RE=6371.0,SCALE=4.0/RE;\n'
        '\n'
        '// Kepler solver\n'
        'function kepE(M,e){let E=M;for(let i=0;i<60;i++){const d=(M-E+e*Math.sin(E))/(1-e*Math.cos(E));E+=d;if(Math.abs(d)<1e-10)break;}return E;}\n'
        'function kep2eci(a,e,i,raan,w,M){\n'
        '  const E=kepE(M,e);\n'
        '  const nu=2*Math.atan2(Math.sqrt(1+e)*Math.sin(E/2),Math.sqrt(1-e)*Math.cos(E/2));\n'
        '  const r=a*(1-e*Math.cos(E)),p_=a*(1-e*e),h=Math.sqrt(GM*p_);\n'
        '  const dn=h/r,rd=GM/h*e*Math.sin(nu);\n'
        '  const xP=r*Math.cos(nu),yP=r*Math.sin(nu);\n'
        '  const vxP=rd*Math.cos(nu)-dn*Math.sin(nu),vyP=rd*Math.sin(nu)+dn*Math.cos(nu);\n'
        '  const cR=Math.cos(raan),sR=Math.sin(raan),cI=Math.cos(i),sI=Math.sin(i),cW=Math.cos(w),sW=Math.sin(w);\n'
        '  return{px:(cR*cW-sR*sW*cI)*xP+(-cR*sW-sR*cW*cI)*yP,\n'
        '         py:(sR*cW+cR*sW*cI)*xP+(-sR*sW+cR*cW*cI)*yP,\n'
        '         pz:(sW*sI)*xP+(cW*sI)*yP,\n'
        '         vx:(cR*cW-sR*sW*cI)*vxP+(-cR*sW-sR*cW*cI)*vyP,\n'
        '         vy:(sR*cW+cR*sW*cI)*vxP+(-sR*sW+cR*cW*cI)*vyP,\n'
        '         vz:(sW*sI)*vxP+(cW*sI)*vyP};\n'
        '}\n'
        '// ECI → Three.js  (ECI Z=north → Three.js Y=up)\n'
        'function e2t(x,y,z){return new THREE.Vector3(x*SCALE,z*SCALE,y*SCALE);}\n'
        '\n'
        '// Scene\n'
        'const canvas=document.getElementById("c");\n'
        'const renderer=new THREE.WebGLRenderer({canvas,antialias:true});\n'
        'renderer.setPixelRatio(Math.min(devicePixelRatio,2));\n'
        'renderer.setClearColor(0x02040e,1);\n'
        'const scene=new THREE.Scene();\n'
        'const camera=new THREE.PerspectiveCamera(42,1,.01,4000);\n'
        'camera.position.set(10,5,10);\n'
        'function resize(){const w=canvas.clientWidth,h=canvas.clientHeight||520;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}\n'
        'resize();window.addEventListener("resize",resize);\n'
        '\n'
        '// Lighting — subdued so Earth surface is not blown out\n'
        'scene.add(new THREE.AmbientLight(0x08112a,2.5));\n'  # very dim ambient, no white
        'const sunL=new THREE.DirectionalLight(0xffeedd,1.6);\n'
        'sunL.position.set(50,15,25);scene.add(sunL);\n'
        'const fillL=new THREE.DirectionalLight(0x060e20,0.4);\n'  # barely visible fill
        'fillL.position.set(-30,-10,-20);scene.add(fillL);\n'
        '\n'
        '// Stars\n'
        '(()=>{const N=3000,p=new Float32Array(N*3);for(let i=0;i<N;i++){const r=600+Math.random()*300,t=Math.random()*Math.PI*2,ph=Math.acos(2*Math.random()-1);p[i*3]=r*Math.sin(ph)*Math.cos(t);p[i*3+1]=r*Math.cos(ph);p[i*3+2]=r*Math.sin(ph)*Math.sin(t);}const g=new THREE.BufferGeometry();g.setAttribute("position",new THREE.BufferAttribute(p,3));scene.add(new THREE.Points(g,new THREE.PointsMaterial({size:.5,color:0x6677aa,sizeAttenuation:false})));})();\n'
        '\n'
        '// Earth — dark ocean, NO emissive blow-out\n'
        'const earthMat=new THREE.MeshPhongMaterial({\n'
        '  color:0x08213f,       // dark navy ocean\n'
        '  emissive:0x010508,    // nearly black self-glow\n'
        '  specular:0x0a1828,\n'
        '  shininess:12\n'
        '});\n'
        'const earth=new THREE.Mesh(new THREE.SphereGeometry(4.0,64,64),earthMat);\n'
        'scene.add(earth);\n'
        '// Thin atmosphere haze\n'
        'scene.add(new THREE.Mesh(new THREE.SphereGeometry(4.07,32,32),\n'
        '  new THREE.MeshPhongMaterial({color:0x0a2288,transparent:true,opacity:.04,side:THREE.FrontSide})));\n'
        '// Grid\n'
        'scene.add(new THREE.Mesh(new THREE.SphereGeometry(4.01,36,18),\n'
        '  new THREE.MeshBasicMaterial({color:0x0b2850,wireframe:true,transparent:true,opacity:.07})));\n'
        '// Equatorial ring\n'
        '(()=>{const pts=[];for(let i=0;i<=360;i++){const a=i*Math.PI/180;pts.push(new THREE.Vector3(4.02*Math.cos(a),0,4.02*Math.sin(a)));}scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0x112a55,transparent:true,opacity:.4})));})();\n'
        '\n'
        '// Orbit track\n'
        '(()=>{const pts=[];for(let i=0;i<=200;i++){const M_=i/200*2*Math.PI;const s=kep2eci(SAT_A,.0001,SAT_I,SAT_RAAN,SAT_W,M_);pts.push(e2t(s.px,s.py,s.pz));}scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0x005577,transparent:true,opacity:.40})));})();\n'
        '\n'
        '// Satellite — GOLD color, compact size\n'
        'const satG=new THREE.Group();\n'
        'const coreMat=new THREE.MeshPhongMaterial({color:0xffc840,emissive:0x301800,specular:0xffe88a,shininess:100});\n'
        'const core=new THREE.Mesh(new THREE.SphereGeometry(.07,14,14),coreMat);\n'
        'satG.add(core);\n'
        'satG.add(new THREE.Mesh(new THREE.BoxGeometry(.16,.04,.04),new THREE.MeshPhongMaterial({color:0xc0ccd8,specular:0x5577aa,shininess:60})));\n'
        '[-1,1].forEach(s=>{const p=new THREE.Mesh(new THREE.BoxGeometry(.34,.003,.10),new THREE.MeshPhongMaterial({color:0x071640,emissive:0x030a20,shininess:30}));p.position.x=s*.25;satG.add(p);});\n'
        'const glowMat=new THREE.MeshBasicMaterial({color:0xffc840,transparent:true,opacity:.14});\n'
        'satG.add(new THREE.Mesh(new THREE.SphereGeometry(.14,10,10),glowMat));\n'
        'scene.add(satG);\n'
        '\n'
        '// Velocity arrow\n'
        'const vArr=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]),new THREE.LineBasicMaterial({color:0x003366,transparent:true,opacity:.5}));\n'
        'scene.add(vArr);\n'
        '\n'
        '// DV arrow (green = direction model says satellite should accelerate)\n'
        'const dvArr=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]),new THREE.LineBasicMaterial({color:0x00ff44,linewidth:2}));\n'
        'dvArr.visible=false;scene.add(dvArr);\n'
        '\n'
        '// ── Burn exhaust particles ─────────────────────────────────────────\n'
        '// DV in ECI frame (m/s) → Three.js direction (swap y↔z)\n'
        '// Satellite accelerates in dvDir3 direction\n'
        '// Exhaust fires OPPOSITE = exhDir\n'
        'const dvDir3=new THREE.Vector3(DV[0],DV[2],DV[1]).normalize();\n'
        'const exhDir=dvDir3.clone().negate();\n'
        'const NP=260;\n'
        'const pPos=new Float32Array(NP*3),pCol=new Float32Array(NP*3);\n'
        'const pVel=Array.from({length:NP},()=>new THREE.Vector3());\n'
        'const pLife=new Float32Array(NP),pMax=new Float32Array(NP);\n'
        'const pGeo=new THREE.BufferGeometry();\n'
        'pGeo.setAttribute("position",new THREE.BufferAttribute(pPos,3));\n'
        'pGeo.setAttribute("color",new THREE.BufferAttribute(pCol,3));\n'
        'const parts=new THREE.Points(pGeo,new THREE.PointsMaterial({size:.028,vertexColors:true,transparent:true,opacity:.92,depthWrite:false,sizeAttenuation:true}));\n'
        'parts.visible=false;scene.add(parts);\n'
        '\n'
        'function spawnP(i,origin){\n'
        '  const j=i*3;\n'
        '  // place at nozzle (slightly behind sat in exhaust direction)\n'
        '  pPos[j  ]=origin.x+exhDir.x*.03;\n'
        '  pPos[j+1]=origin.y+exhDir.y*.03;\n'
        '  pPos[j+2]=origin.z+exhDir.z*.03;\n'
        '  // cone spread around exhDir\n'
        '  const sp=.18,spd=.05+Math.random()*.07;\n'
        '  const right=new THREE.Vector3().crossVectors(exhDir,new THREE.Vector3(0,1,.1)).normalize();\n'
        '  const up2=new THREE.Vector3().crossVectors(exhDir,right);\n'
        '  const ang=Math.random()*Math.PI*2,rad=Math.random()*sp;\n'
        '  pVel[i].copy(exhDir).add(right.clone().multiplyScalar(Math.cos(ang)*rad)).add(up2.clone().multiplyScalar(Math.sin(ang)*rad)).normalize().multiplyScalar(spd);\n'
        '  // color: white→orange→red\n'
        '  const t=Math.random();\n'
        '  pCol[j]=1.0;pCol[j+1]=t<.4?.85+t*.3:t<.7?.4+t*.2:.1+t*.1;pCol[j+2]=t<.3?.5:0;\n'
        '  pMax[i]=.45+Math.random()*.6;pLife[i]=0;\n'
        '}\n'
        '\n'
        '// Burn point marker\n'
        'const bMkr=new THREE.Mesh(new THREE.SphereGeometry(.04,8,8),new THREE.MeshBasicMaterial({color:0xff2200,transparent:true,opacity:.55}));\n'
        'bMkr.visible=false;scene.add(bMkr);\n'
        'if(BURN&&(BP[0]||BP[1]||BP[2])){bMkr.position.copy(e2t(BP[0],BP[1],BP[2]));bMkr.visible=true;}\n'
        '\n'
        '// Post-maneuver trajectory (green dashed line)\n'
        'if(POST.length>1){\n'
        '  const pts=POST.map(p=>e2t(p[0],p[1],p[2]));\n'
        '  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0x00ff44,transparent:true,opacity:.6})));\n'
        '  const ep=new THREE.Mesh(new THREE.SphereGeometry(.05,8,8),new THREE.MeshBasicMaterial({color:0x00ff44,transparent:true,opacity:.7}));\n'
        '  ep.position.copy(pts[pts.length-1]);scene.add(ep);\n'
        '}\n'
        '\n'
        '// Debris objects\n'
        'const AC3={GREEN:0x00ff88,YELLOW:0xffd700,ORANGE:0xff8c00,RED:0xff2244};\n'
        'const debObjs=DEBRIS.map(d=>{\n'
        '  const col=AC3[d.a]||0xaaaaaa;\n'
        '  const mesh=new THREE.Mesh(new THREE.SphereGeometry(.055,10,10),new THREE.MeshPhongMaterial({color:col,emissive:col,emissiveIntensity:.6}));\n'
        '  scene.add(mesh);\n'
        '  const lGeo=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]);\n'
        '  const line=new THREE.Line(lGeo,new THREE.LineBasicMaterial({color:col,transparent:true,opacity:.55}));\n'
        '  scene.add(line);\n'
        '  return{mesh,line,rp:d.r};\n'
        '});\n'
        '\n'
        '// Alert badge\n'
        '(()=>{\n'
        '  if(!DEBRIS.length)return;\n'
        '  const ord={GREEN:0,YELLOW:1,ORANGE:2,RED:3};\n'
        '  const w=DEBRIS.reduce((a,b)=>ord[b.a]>ord[a.a]?b:a,DEBRIS[0]);\n'
        '  if(w.a==="GREEN")return;\n'
        '  const cc={YELLOW:"#ffd700",ORANGE:"#ff8c00",RED:"#ff2244"};\n'
        '  const bg={YELLOW:"#0f0d00",ORANGE:"#0f0500",RED:"#0f0005"};\n'
        '  const el=document.getElementById("badge");\n'
        '  el.style.color=cc[w.a]||"#fff";el.style.borderColor=cc[w.a]||"#fff";el.style.background=bg[w.a]||"#000";\n'
        '  el.textContent="⚠ "+w.a+" — "+w.n;el.style.display="block";\n'
        '})();\n'
        '\n'
        '// Camera\n'
        'let drag=false,prev={x:0,y:0},cam={th:.7,ph:1.0,r:13};\n'
        'canvas.addEventListener("mousedown",e=>{drag=true;prev={x:e.clientX,y:e.clientY};});\n'
        'canvas.addEventListener("mouseup",()=>drag=false);\n'
        'canvas.addEventListener("mouseleave",()=>drag=false);\n'
        'canvas.addEventListener("mousemove",e=>{if(!drag)return;cam.th-=(e.clientX-prev.x)*.008;cam.ph=Math.max(.08,Math.min(Math.PI-.08,cam.ph+(e.clientY-prev.y)*.008));prev={x:e.clientX,y:e.clientY};});\n'
        'canvas.addEventListener("wheel",e=>{cam.r=Math.max(4.5,Math.min(80,cam.r+e.deltaY*.03));},{passive:true});\n'
        'canvas.addEventListener("touchstart",e=>{drag=true;prev={x:e.touches[0].clientX,y:e.touches[0].clientY};});\n'
        'canvas.addEventListener("touchend",()=>drag=false);\n'
        'canvas.addEventListener("touchmove",e=>{cam.th-=(e.touches[0].clientX-prev.x)*.008;cam.ph=Math.max(.08,Math.min(Math.PI-.08,cam.ph+(e.touches[0].clientY-prev.y)*.008));prev={x:e.touches[0].clientX,y:e.touches[0].clientY};});\n'
        '\n'
        '// Animation\n'
        'const t0=performance.now()/1000;\n'
        'const n0=2*Math.PI/SAT_T;\n'
        'let pInited=false;\n'
        '\n'
        'function animate(){\n'
        '  requestAnimationFrame(animate);\n'
        '  const tR=performance.now()/1000-t0;\n'
        '  // Real mean anomaly from file + propagation at 100x\n'
        '  const M_now=(SAT_M0+n0*(tR*SIM_SPD))%(2*Math.PI);\n'
        '  const st=kep2eci(SAT_A,.0001,SAT_I,SAT_RAAN,SAT_W,M_now);\n'
        '  const satPos=e2t(st.px,st.py,st.pz);\n'
        '  satG.position.copy(satPos);\n'
        '  // Orient: prograde=X, nadir=Y\n'
        '  const vT=new THREE.Vector3(st.vx,st.vz,st.vy).normalize();\n'
        '  const rT=satPos.clone().negate().normalize();\n'
        '  const sT=new THREE.Vector3().crossVectors(vT,rT).normalize();\n'
        '  satG.setRotationFromMatrix(new THREE.Matrix4().makeBasis(vT,rT,sT));\n'
        '  // Satellite appearance\n'
        '  if(BURN&&dvDir3.length()>.001){\n'
        '    coreMat.color.setHex(0xff5500);\n'
        '    coreMat.emissive.setHex(0x330e00);\n'
        '    glowMat.color.setHex(0xff6600);\n'
        '    glowMat.opacity=.28+.12*Math.sin(tR*14);\n'
        '  }else{\n'
        '    coreMat.color.setHex(0xffc840);\n'
        '    coreMat.emissive.setHex(0x221200);\n'
        '    glowMat.color.setHex(0xffc840);\n'
        '    glowMat.opacity=.10+.03*Math.sin(tR*1.5);\n'
        '  }\n'
        '  // Velocity arrow (dim blue)\n'
        '  const vEnd=satPos.clone().add(vT.clone().multiplyScalar(.6));\n'
        '  {const p=vArr.geometry.attributes.position;p.setXYZ(0,satPos.x,satPos.y,satPos.z);p.setXYZ(1,vEnd.x,vEnd.y,vEnd.z);p.needsUpdate=true;}\n'
        '  // Debris\n'
        '  debObjs.forEach(d=>{\n'
        '    const dPos=e2t(st.px+d.rp[0],st.py+d.rp[1],st.pz+d.rp[2]);\n'
        '    d.mesh.position.copy(dPos);\n'
        '    d.mesh.material.emissiveIntensity=.4+.3*Math.sin(tR*4);\n'
        '    const lp=d.line.geometry.attributes.position;\n'
        '    lp.setXYZ(0,satPos.x,satPos.y,satPos.z);\n'
        '    lp.setXYZ(1,dPos.x,dPos.y,dPos.z);\n'
        '    lp.needsUpdate=true;\n'
        '  });\n'
        '  // Burn exhaust — fires OPPOSITE to ΔV (exhaust = -ΔV direction)\n'
        '  if(BURN&&dvDir3.length()>.001){\n'
        '    parts.visible=true;\n'
        '    if(!pInited){for(let i=0;i<NP;i++)spawnP(i,satPos);pInited=true;}\n'
        '    for(let i=0;i<NP;i++){\n'
        '      const j=i*3;\n'
        '      pLife[i]+=.016;\n'
        '      if(pLife[i]>pMax[i]){spawnP(i,satPos);}\n'
        '      else{pPos[j]+=pVel[i].x;pPos[j+1]+=pVel[i].y;pPos[j+2]+=pVel[i].z;pCol[j]*=.995;pCol[j+1]*=.97;pCol[j+2]*=.95;}\n'
        '    }\n'
        '    pGeo.attributes.position.needsUpdate=true;\n'
        '    pGeo.attributes.color.needsUpdate=true;\n'
        '    // DV arrow = direction model says satellite moves\n'
        '    dvArr.visible=true;\n'
        '    const dvEnd=satPos.clone().add(dvDir3.clone().multiplyScalar(.9));\n'
        '    {const dp=dvArr.geometry.attributes.position;dp.setXYZ(0,satPos.x,satPos.y,satPos.z);dp.setXYZ(1,dvEnd.x,dvEnd.y,dvEnd.z);dp.needsUpdate=true;}\n'
        '  }else{parts.visible=false;dvArr.visible=false;}\n'
        '  // Earth slow rotation\n'
        '  earth.rotation.y+=.000028;\n'
        '  // Camera\n'
        '  const cx=cam.r*Math.sin(cam.ph)*Math.cos(cam.th);\n'
        '  const cy=cam.r*Math.cos(cam.ph);\n'
        '  const cz=cam.r*Math.sin(cam.ph)*Math.sin(cam.th);\n'
        '  camera.position.lerp(new THREE.Vector3(cx,cy,cz),.07);\n'
        '  camera.lookAt(0,0,0);\n'
        '  renderer.render(scene,camera);\n'
        '}\n'
        'animate();\n'
        '</script></body></html>'
    )


# ============================================================
# RENDER: MODE BADGE + METRICS ROW + THREAT COUNT
# ============================================================
r1,r2,r3 = st.columns([2,8,1])
with r1:
    st.markdown(
        f'<div class="mode-badge" style="color:{mode_col};background:{mode_bg};'
        f'border-color:{mode_border}">{mode_label}</div>',
        unsafe_allow_html=True)

fuel_c  = "mc-red" if fuel_now<20 else "mc-yellow" if fuel_now<40 else "mc-green"
bat_c   = "mc-red" if bat_now<25  else "mc-yellow" if bat_now<50  else ""
ecl_c   = "mc-red" if eclipse else "mc-green"
gnd_c   = "mc-green" if ground else "mc-red"

with r2:
    st.markdown(
        f'<div class="mc-row">'
        f'<div class="mc"><span class="mc-label">Altitude</span><span class="mc-val">{alt:.2f} km</span></div>'
        f'<div class="mc"><span class="mc-label">Latitude</span><span class="mc-val">{lat:.3f}°</span></div>'
        f'<div class="mc"><span class="mc-label">Longitude</span><span class="mc-val">{lon:.3f}°</span></div>'
        f'<div class="mc"><span class="mc-label">Speed</span><span class="mc-val">{spd:.4f} km/s</span></div>'
        f'<div class="mc"><span class="mc-label">Battery</span><span class="mc-val {bat_c}">{bat_now:.1f}%</span></div>'
        f'<div class="mc"><span class="mc-label">Fuel</span><span class="mc-val {fuel_c}">{fuel_now:.2f}%</span></div>'
        f'<div class="mc"><span class="mc-label">Eclipse</span><span class="mc-val {ecl_c}">{"YES" if eclipse else "NO"}</span></div>'
        f'<div class="mc"><span class="mc-label">Ground</span><span class="mc-val {gnd_c}">{"LINK" if ground else "LOSS"}</span></div>'
        f'<div class="mc"><span class="mc-label">Orbit #</span><span class="mc-val">{sm["orbit_number"]}</span></div>'
        f'</div>', unsafe_allow_html=True)

with r3:
    n_thr = len(st.session_state.objects)
    tc = "mc-red" if n_thr>0 and overall==Alert.RED else "mc-orange" if overall==Alert.ORANGE else "mc-yellow" if overall==Alert.YELLOW else ""
    st.markdown(
        f'<div class="mc" style="text-align:center">'
        f'<span class="mc-label">THREATS</span>'
        f'<span class="mc-val {tc}" style="font-size:24px">{n_thr}</span>'
        f'</div>', unsafe_allow_html=True)

# Burn banner
if burn_active and np.linalg.norm(dv_ss)>0:
    bp=st.session_state.burn_pos
    st.markdown(
        f'<div class="burn-banner">🔥 <b style="color:#ff4400">THRUSTERS FIRING</b>'
        f' &nbsp;|&nbsp; <span style="color:#ff8800">ΔV = [{dv_ss[0]:.3f}, {dv_ss[1]:.3f}, {dv_ss[2]:.3f}] m/s'
        f'  |ΔV|={np.linalg.norm(dv_ss):.3f} m/s</span>'
        f' &nbsp;|&nbsp; <span style="color:#ffd700">Burn ECI = [{bp[0]:.1f}, {bp[1]:.1f}, {bp[2]:.1f}] km</span>'
        f' &nbsp;|&nbsp; <span style="color:#aabbff">{"AUTONOMOUS" if not ground else "GND CONFIRMED"}</span>'
        f'</div>', unsafe_allow_html=True)


# ============================================================
# MAIN COLUMNS: 3D GLOBE (left) | RIGHT PANEL
# ============================================================
c3d, crp = st.columns([11, 9])

# ── 3D GLOBE ─────────────────────────────────────────────────
with c3d:
    debris_for_js = []
    for r in all_results:
        c = r['conjunction']
        debris_for_js.append({
            'name':    c['object_name'],
            'alert':   r['assessment'].alert.value,
            'rel_pos': c['rel_pos'].tolist(),
        })

    html_globe = build_globe(
        sm_data    = sm,
        debris_list= debris_for_js,
        burn_active= bool(burn_active),
        dv_vec     = dv_ss.tolist(),
        burn_pos   = st.session_state.burn_pos,
        post_pts   = st.session_state.post_pts,
    )
    components.html(html_globe, height=520, scrolling=False)


# ── RIGHT PANEL ───────────────────────────────────────────────
with crp:

    # §2.3 — BURN / MODEL OUTPUT (shown when burn is active)
    if burn_active and np.linalg.norm(dv_ss)>0:
        lr = st.session_state.last_result
        asm_lr = lr['assessment'] if lr else None
        dv2 = dv_ss; dvm = np.linalg.norm(dv2)
        dv_unit = dv2/(dvm+1e-10)

        st.markdown('<div class="sec">🔥 MODEL BURN OUTPUT</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="rcard" style="border-color:#ff4400">'
            f'<div class="trow"><span class="tkey">Alert Level</span>'
            f'<span class="tval tv-r">{asm_lr.alert.value if asm_lr else "RED"}</span></div>'
            f'<div class="trow"><span class="tkey">Action</span>'
            f'<span class="tval tv-r">{lr["action"] if lr else "BURN"}</span></div>'
            f'<div class="trow"><span class="tkey">Method</span>'
            f'<span class="tval">{lr["dv_method"] if lr else "—"}</span></div>'
            f'<div class="trow"><span class="tkey">ΔV Vector (m/s)</span>'
            f'<span class="tval tv-o">[{dv2[0]:.4f}, {dv2[1]:.4f}, {dv2[2]:.4f}]</span></div>'
            f'<div class="trow"><span class="tkey">|ΔV| Magnitude</span>'
            f'<span class="tval tv-o">{dvm:.4f} m/s</span></div>'
            f'<div class="trow"><span class="tkey">Burn Direction</span>'
            f'<span class="tval">[{dv_unit[0]:.4f}, {dv_unit[1]:.4f}, {dv_unit[2]:.4f}]</span></div>'
            f'<div class="trow"><span class="tkey">Burn ECI Position</span>'
            f'<span class="tval">[{st.session_state.burn_pos[0]:.1f},'
            f' {st.session_state.burn_pos[1]:.1f},'
            f' {st.session_state.burn_pos[2]:.1f}] km</span></div>'
            f'<div class="trow"><span class="tkey">Adjusted Pc</span>'
            f'<span class="tval tv-r">{(f"{asm_lr.adjusted_pc:.3e}") if asm_lr else "—"}</span></div>'
            f'<div class="trow"><span class="tkey">Fuel Cost</span>'
            f'<span class="tval tv-o">{st.session_state.last_fuel_cost:.4f} %</span></div>'
            f'<div class="trow"><span class="tkey">Fuel Remaining</span>'
            f'<span class="tval {fuel_c}">{fuel_now:.3f} %</span></div>'
            f'</div>', unsafe_allow_html=True)

        # Pipeline trace for the latest result
        if lr:
            with st.expander("🔬 Pipeline Trace", expanded=False):
                html_t='<div class="mono">'
                for pe in lr['log']:
                    html_t+=(f'<div class="ps"><span style="color:#2244cc">S{pe["s"]}</span> '
                             f'<span class="{pe["c"]}">{pe["m"]}</span></div>')
                html_t+='</div>'
                st.markdown(html_t, unsafe_allow_html=True)

    elif all_results:
        # Pipeline results without burn
        for r in all_results:
            c=r['conjunction']; a=r['assessment']
            BC=AC[a.alert.value]
            IC={"GREEN":"✅","YELLOW":"⚠️","ORANGE":"🟠","RED":"🔴"}[a.alert.value]
            mc2="tv-r" if c['miss_km']<1 else "tv-o" if c['miss_km']<2 else "tval"
            st.markdown(
                f'<div class="rcard" style="border-color:{BC}">'
                f'<b style="color:{BC};font-family:Share Tech Mono,monospace">'
                f'{IC} {c["object_name"]}</b>'
                f'<br><div class="trow"><span class="tkey">Miss Distance</span>'
                f'<span class="tval {mc2}">{c["miss_km"]:.3f} km</span></div>'
                f'<div class="trow"><span class="tkey">TCA</span>'
                f'<span class="tval">{c["tca_hours"]:.2f} h</span></div>'
                f'<div class="trow"><span class="tkey">Adjusted Pc</span>'
                f'<span class="tval">{a.adjusted_pc:.3e}</span></div>'
                f'<div class="trow"><span class="tkey">Action</span>'
                f'<span class="tval">{r["action"]}</span></div>'
                f'</div>', unsafe_allow_html=True)
            with st.expander("Pipeline trace", expanded=False):
                html_t='<div class="mono">'
                for pe in r['log']:
                    html_t+=(f'<div class="ps"><span style="color:#2244cc">S{pe["s"]}</span> '
                             f'<span class="{pe["c"]}">{pe["m"]}</span></div>')
                html_t+='</div>'
                st.markdown(html_t, unsafe_allow_html=True)

    # §2.3 — THREAT SCENARIOS
    st.markdown('<div class="sec">🎯 Threat Scenarios</div>', unsafe_allow_html=True)
    for sc in SCENARIOS:
        col_a=AC[sc['alert']]
        c1,c2=st.columns([4,1])
        with c1:
            spd_sc=np.linalg.norm(sc['rv'])
            st.markdown(
                f'<div class="sc-card">'
                f'<span style="color:{col_a};font-family:Share Tech Mono,monospace;'
                f'font-size:10px;font-weight:bold">{sc["alert"]}</span>'
                f'&nbsp;<span style="color:#c8d8f0;font-family:Share Tech Mono,monospace;'
                f'font-size:11px">{sc["name"]}</span>'
                f'<br><span style="color:#3a4a66;font-family:Share Tech Mono,monospace;font-size:9px">'
                f'miss={sc["miss_km"]:.2f}km &nbsp;TCA={sc["tca_h"]:.1f}h &nbsp;'
                f'spd={spd_sc:.2f}km/s{"  [STALE]" if sc["stale"] else ""}'
                f'</span>'
                f'<br><span style="color:#263040;font-family:Share Tech Mono,monospace;'
                f'font-size:9px">{sc["desc"]}</span>'
                f'</div>', unsafe_allow_html=True)
        with c2:
            if st.button("Inject", key=f"sc_{sc['norad']}", use_container_width=True):
                rpa=np.array(sc['rp']); rva=np.array(sc['rv'])
                st.session_state.objects.append({
                    "object_id":sc['norad'],"object_name":sc['name'],
                    "object_type":"DEBRIS","miss_km":sc['miss_km'],
                    "tca_hours":sc['tca_h'],"rel_pos":rpa,"rel_vel":rva,
                    "rel_speed_kms":float(np.linalg.norm(rva)),
                    "tle_stale":sc['stale'],"tle_age_hours":sc['tle_age'],
                })
                add_log(f"📥 {sc['name']} | {sc['alert']} | miss={sc['miss_km']:.2f}km TCA={sc['tca_h']:.1f}h","lw")
                st.session_state.cycle+=1
                st.rerun()

    # §2.4 — ACAS LOG
    st.markdown('<div class="sec">▶ ACAS Log</div>', unsafe_allow_html=True)
    if not st.session_state.log:
        st.caption("No log entries yet.")
    else:
        h='<div class="mono">'
        for e in st.session_state.log[:40]:
            h+=(f'<div><span style="color:#1a2233">[{e["ts"]}]</span> '
                f'<span class="{e["css"]}">{e["msg"]}</span></div>')
        h+='</div>'
        st.markdown(h, unsafe_allow_html=True)

    if st.session_state.objects:
        if st.button("🗑️ Clear all threats", use_container_width=True):
            st.session_state.objects=[]; st.session_state.maneuver_on=False
            st.session_state.last_dv=np.zeros(3); st.session_state.post_pts=[]
            st.session_state.fuel_override=None; st.session_state.last_result=None
            add_log("🗑️ All threats cleared.","lw"); st.rerun()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:6px 0">'
        '<span style="font-size:24px">🛰️</span><br>'
        '<b style="font-size:14px;color:#00d4ff;letter-spacing:3px">POWER HOUSE</b><br>'
        '<span style="font-size:9px;color:#3a4a66;letter-spacing:2px">IN-SPACe · ACAS DASHBOARD</span>'
        '</div>', unsafe_allow_html=True)
    st.divider()

    # Model status
    if M['onnx_ok']: st.success(M['onnx_msg'], icon="✅")
    else:            st.warning(M['onnx_msg'],  icon="⚠️")
    if M['rl_ok']:   st.success(M['rl_msg'],    icon="✅")
    else:            st.warning(M['rl_msg'],     icon="⚠️")
    st.divider()

    # §1 — Live ECI position & velocity (fragment, 3s update)
    # Only this block re-runs every 3 seconds — everything else is stable
    @st.fragment(run_every=3)
    def sidebar_live():
        sm2 = read_model()
        if not sm2: return
        e2 = sm2['eci_state']; h2 = sm2['health']
        fuel2=(st.session_state.fuel_override
               if st.session_state.fuel_override is not None
               else h2['fuel_pct'])
        bat2=h2['battery_pct']
        ecl2=sm2['environment']['in_eclipse']
        gnd2=sm2['communications']['ground_contact']

        st.markdown('<div class="sec">§1 — Satellite ECI State</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-family:Share Tech Mono,monospace;font-size:9px;'
            f'color:#3a4a66;letter-spacing:1px;margin-bottom:3px">POSITION (km)</div>'
            f'<div class="vg">'
            f'<div class="vc"><span class="va">X</span><span class="vn">{e2["pos_x_km"]:.1f}</span></div>'
            f'<div class="vc"><span class="va">Y</span><span class="vn">{e2["pos_y_km"]:.1f}</span></div>'
            f'<div class="vc"><span class="va">Z</span><span class="vn">{e2["pos_z_km"]:.1f}</span></div>'
            f'</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-family:Share Tech Mono,monospace;font-size:9px;'
            f'color:#3a4a66;letter-spacing:1px;margin-bottom:3px">VELOCITY (km/s)</div>'
            f'<div class="vg">'
            f'<div class="vc"><span class="va">Vx</span><span class="vn">{e2["vel_x_kms"]:.4f}</span></div>'
            f'<div class="vc"><span class="va">Vy</span><span class="vn">{e2["vel_y_kms"]:.4f}</span></div>'
            f'<div class="vc"><span class="va">Vz</span><span class="vn">{e2["vel_z_kms"]:.4f}</span></div>'
            f'</div>', unsafe_allow_html=True)

        # Fuel gauge
        fc="#ff2244" if fuel2<20 else "#ffd700" if fuel2<40 else "#00ff88"
        st.markdown(
            f'<div class="g-wrap"><div class="g-head">'
            f'<span class="g-lbl">FUEL</span><span class="g-val">{fuel2:.2f}%</span>'
            f'</div><div class="g-bar">'
            f'<div class="g-fill" style="width:{fuel2:.1f}%;background:{fc}"></div>'
            f'</div></div>', unsafe_allow_html=True)

        # Battery gauge
        bc2="#ff2244" if bat2<25 else "#ffd700" if bat2<50 else "#00d4ff"
        st.markdown(
            f'<div class="g-wrap"><div class="g-head">'
            f'<span class="g-lbl">BATTERY</span><span class="g-val">{bat2:.1f}%</span>'
            f'</div><div class="g-bar">'
            f'<div class="g-fill" style="width:{bat2:.1f}%;background:{bc2}"></div>'
            f'</div></div>', unsafe_allow_html=True)

        # Status pills
        st.markdown(
            f'<div class="pr">'
            f'<span class="pill {"p-on" if gnd2 else "p-off"}">'
            f'{"GND ✓" if gnd2 else "BLACKOUT"}</span>'
            f'<span class="pill {"p-off" if ecl2 else "p-on"}">'
            f'{"🌑 ECLIPSE" if ecl2 else "☀️ SUNLIT"}</span>'
            f'</div>', unsafe_allow_html=True)

    sidebar_live()
    st.divider()

    # §2 — Custom threat injection form
    st.markdown('<div class="sec">§2 — Custom Threat Injection</div>', unsafe_allow_html=True)
    with st.form("threat_form", clear_on_submit=False):
        name = st.text_input("Object Name", "CUSTOM DEB-1")
        nid  = st.text_input("NORAD ID",    "99001")
        c1,c2 = st.columns(2)
        miss_km   = c1.slider("Miss (km)",  0.01, 5.0, 1.5, 0.01)
        tca_hours = c2.slider("TCA (h)",    0.1, 48.0, 3.0, 0.1)
        st.markdown("**Relative Position (km)**")
        rpx = st.slider("X", -10.,10.,  0.9,0.01,key="rpx")
        rpy = st.slider("Y", -10.,10., -0.75,0.01,key="rpy")
        rpz = st.slider("Z", -10.,10.,  0.2, 0.01,key="rpz")
        st.markdown("**Relative Velocity (km/s)**")
        rvx = st.slider("Vx",-15.,15.,-6.6, 0.01,key="rvx")
        rvy = st.slider("Vy",-15.,15., 3.0, 0.01,key="rvy")
        rvz = st.slider("Vz",-15.,15., 1.65,0.01,key="rvz")
        stale = st.checkbox("TLE Stale (>48h)")
        t_age = st.slider("TLE Age (h)",0.,200.,72. if stale else 8.,1.)
        sub   = st.form_submit_button("🚀 INJECT THREAT", type="primary", use_container_width=True)

    if sub:
        rpa=np.array([rpx,rpy,rpz]); rva=np.array([rvx,rvy,rvz])
        st.session_state.objects.append({
            "object_id":nid,"object_name":name,
            "object_type":"DEBRIS","miss_km":miss_km,
            "tca_hours":tca_hours,"rel_pos":rpa,"rel_vel":rva,
            "rel_speed_kms":float(np.linalg.norm(rva)),
            "tle_stale":stale,"tle_age_hours":float(t_age),
        })
        add_log(f"📥 CUSTOM: {name} | miss={miss_km:.2f}km TCA={tca_hours:.1f}h "
                f"rp=[{rpx:.2f},{rpy:.2f},{rpz:.2f}] "
                f"rv=[{rvx:.2f},{rvy:.2f},{rvz:.2f}]"
                f"{'  [STALE]' if stale else ''}","lw")
        st.session_state.cycle+=1
        st.rerun()

    # Clear all threats button always visible in sidebar
    if st.session_state.objects:
        if st.button("🗑️ Clear All Threats", use_container_width=True, key="sb_clear"):
            st.session_state.objects      = []
            st.session_state.maneuver_on  = False
            st.session_state.last_dv      = np.zeros(3)
            st.session_state.post_pts     = []
            st.session_state.fuel_override = None
            st.session_state.last_result  = None
            add_log("🗑️ All threats cleared.", "lw")
            st.rerun()

    st.caption(f"Cycle #{st.session_state.cycle}  ·  {sm['timestamp'][11:19]} UTC")