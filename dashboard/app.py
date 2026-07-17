"""
dashboard/app.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    Streamlit web dashboard. Renders the 3D globe, live telemetry, threat
    table, ACAS pipeline log, and burn visualisation. Allows manual threat
    injection via sliders or built-in SCENARIOS.

CALLED FROM
    Terminal:  streamlit run dashboard/app.py
    pyproject.toml entry point: "acas-dashboard"

CALLS INTO
    model/lgbm_engine.py     LGBMInferenceEngine (loaded once, cached)
    core/risk_scorer.py      RiskScorer, SatState, Alert
    stable_baselines3        PPO.load() for RL manoeuvre agent
    data_files/satellite_model.json  live telemetry, read on every refresh

WHAT IT RENDERS
    Left panel
        Satellite telemetry card   fuel, battery, altitude, ground contact
        Conjunction table          miss_km, TCA, Pc, alert badge
        ACAS pipeline log          step-by-step trace of last prediction
    Right panel
        3D globe (Three.js via st.components)
        Burn output card           alert, Pc, ΔV, fuel cost, decision text
    Sidebar
        Model status banner        LightGBM active / physics fallback
        8 built-in SCENARIOS
        Custom threat form         sliders for miss_km, tca_h, rel_pos, rel_vel

PREDICTION FLOW (each threat injection)
    load_models()                     @st.cache_resource, runs once
    predict_pc(conj)                  LGBMInferenceEngine.predict_pc_from_conjunction
    RiskScorer.assess()               → Assessment
    predict_burn()                    PPO RL agent or geometric fallback
    render alert card + pipeline log

IMPORT CHANGES FROM ORIGINAL (post-patch dashboard/app.py)
    from lgbm_inference_engine import  →  from model.lgbm_engine import
    from models.risk_scorer    import  →  from core.risk_scorer  import
══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, json, time, math
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── LightGBM integration (replaces ConjunctionNet ONNX) ──────────────────
from model.lgbm_engine  import LGBMInferenceEngine
from core.risk_scorer   import RiskScorer, SatState, Alert

MODEL_FILE = os.path.join(ROOT, "data_files", "satellite_model.json")
ONNX_PATH  = os.path.join(ROOT, "trained_models", "conjunction_model.onnx")

# Bundled locally so the globe renders with no outbound network call
# (ground segment networks are often firewalled/air-gapped).
with open(os.path.join(ROOT, "dashboard", "static", "three.min.js")) as _f:
    THREE_JS_SRC = _f.read()
RL_PATH    = os.path.join(ROOT, "trained_models", "rl", "maneuver_policy")

# ============================================================
# PAGE CONFIG + GLOBAL CSS
# ============================================================
st.set_page_config(page_title="ACAS – Power House",
                   page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ============ DESIGN TOKENS ============
   bg base   #0a0e17   panel #121826   raised #161d2e
   border    #232d44   subtle #1a2234
   text      #dfe6f2   secondary #9aa7c0   muted #6b7a99
   accent    #4d9fff   data-cyan #38bdf8
   nominal   #34d399   watch #fbbf24   caution #fb923c   critical #f43f5e
*/
:root{
  --font-ui:'Inter',system-ui,-apple-system,sans-serif;
  --font-mono:'IBM Plex Mono',ui-monospace,'SF Mono',monospace;
}
html,body,.stApp{background:#0a0e17;color:#dfe6f2;font-family:var(--font-ui);}
.main .block-container{padding-top:1.1rem;padding-bottom:1rem;max-width:100%;}
/* kill the default white Streamlit header / toolbar strip */
[data-testid="stHeader"]{background:transparent;height:0;}
[data-testid="stToolbar"]{display:none;}
[data-testid="stDecoration"]{display:none;}
#MainMenu,footer{visibility:hidden;}
h1,h2,h3,h4,h5,h6,p,span,div,label{font-family:var(--font-ui);}

/* ---- SIDEBAR (was unstyled/white) ---- */
[data-testid="stSidebar"]{background:#0c111c;border-right:1px solid #1a2234;}
[data-testid="stSidebar"] .block-container{padding-top:1.4rem;}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p{color:#9aa7c0;}
[data-testid="stSidebar"] [data-testid="stForm"]{
  background:#111726;border:1px solid #1e2740;border-radius:10px;padding:14px;}

/* ---- native Streamlit buttons → professional ---- */
.stButton>button{
  font-family:var(--font-ui);font-weight:600;font-size:12px;
  border-radius:8px;border:1px solid #26324e;background:#141b2b;
  color:#c4d0e6;transition:all .15s ease;letter-spacing:.2px;}
.stButton>button:hover{border-color:#4d9fff;color:#eaf1ff;background:#182238;}
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg,#f43f5e,#c81e3f);border:none;color:#fff;
  font-weight:700;letter-spacing:.4px;box-shadow:0 2px 12px rgba(244,63,94,.35);}
.stButton>button[kind="primary"]:hover{filter:brightness(1.08);color:#fff;}

/* ---- expander / divider polish ---- */
[data-testid="stExpander"]{border:1px solid #1e2740;border-radius:8px;background:#111726;}
hr{border-color:#1a2234 !important;margin:.9rem 0 !important;}

/* ---- mode badge (mission state) ---- */
.mode-badge{
  display:inline-flex;align-items:center;gap:8px;padding:8px 20px;border-radius:8px;
  font-family:var(--font-ui);font-size:13px;font-weight:700;letter-spacing:1.5px;
  border:1px solid;margin-bottom:2px;text-transform:uppercase;}
.mode-badge::before{content:"";width:8px;height:8px;border-radius:50%;
  background:currentColor;box-shadow:0 0 8px currentColor;}
/* ---- metric cards (top telemetry row) ---- */
.mc-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:2px;}
.mc{background:#121826;border:1px solid #1e2740;border-radius:9px;
  padding:9px 14px;flex:1;min-width:96px;}
.mc-label{font-family:var(--font-ui);font-size:9px;font-weight:600;
  color:#7382a0;letter-spacing:1.2px;text-transform:uppercase;display:block;}
.mc-val{font-family:var(--font-mono);font-size:16px;font-weight:500;
  color:#eaf1ff;display:block;margin-top:3px;}
.mc-green{color:#34d399;} .mc-yellow{color:#fbbf24;}
.mc-orange{color:#fb923c;} .mc-red{color:#f43f5e;font-weight:600;}
/* ---- sidebar vec boxes ---- */
.vg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin:5px 0 10px;}
.vc{background:#0e1420;border:1px solid #1e2740;border-radius:7px;
  padding:7px 4px;text-align:center;}
.va{font-family:var(--font-ui);font-size:9px;font-weight:600;
  color:#7382a0;display:block;letter-spacing:.5px;}
.vn{font-family:var(--font-mono);font-size:12px;font-weight:500;
  color:#7dd3fc;display:block;margin-top:2px;}
/* ---- gauge ---- */
.g-wrap{margin-bottom:11px;}
.g-head{display:flex;justify-content:space-between;
  font-family:var(--font-ui);font-size:10px;font-weight:600;margin-bottom:4px;}
.g-lbl{color:#8492ab;letter-spacing:.5px;text-transform:uppercase;}
.g-val{color:#eaf1ff;font-family:var(--font-mono);}
.g-bar{height:6px;background:#1a2234;border-radius:4px;overflow:hidden;}
.g-fill{height:100%;border-radius:4px;transition:width .6s,background .6s;}
/* ---- pill row ---- */
.pr{display:flex;gap:6px;margin:7px 0;}
.pill{font-family:var(--font-ui);font-size:10px;font-weight:600;
  padding:3px 11px;border-radius:6px;letter-spacing:.4px;border:1px solid;}
.p-on{background:rgba(52,211,153,.12);color:#34d399;border-color:rgba(52,211,153,.35);}
.p-off{background:rgba(244,63,94,.12);color:#f43f5e;border-color:rgba(244,63,94,.35);}
/* ---- section title ---- */
.sec{font-family:var(--font-ui);font-size:10px;font-weight:700;letter-spacing:1.5px;
  color:#8492ab;text-transform:uppercase;margin:14px 0 8px;
  border-bottom:1px solid #1e2740;padding-bottom:5px;}
/* ---- right-panel cards ---- */
.rcard{background:#121826;border:1px solid #1e2740;border-radius:10px;
  padding:13px 16px;margin-bottom:10px;}
.trow{display:flex;justify-content:space-between;align-items:baseline;
  margin:4px 0;gap:10px;}
.tkey{font-family:var(--font-ui);font-size:11px;color:#8492ab;font-weight:500;}
.tval{font-family:var(--font-mono);font-size:12px;color:#eaf1ff;text-align:right;}
.tv-g{color:#34d399;} .tv-y{color:#fbbf24;}
.tv-o{color:#fb923c;} .tv-r{color:#f43f5e;font-weight:600;}
/* ---- scenario card ---- */
.sc-card{background:#121826;border:1px solid #1e2740;border-radius:9px;
  padding:10px 13px;margin-bottom:7px;}
/* ---- log ---- */
.mono{font-family:var(--font-mono);font-size:11px;line-height:1.75;}
.lg{color:#34d399} .ly{color:#fbbf24} .lo{color:#fb923c}
.lr{color:#f43f5e;font-weight:600} .li{color:#7dd3fc} .lw{color:#8492ab}
.lb{color:#fb923c;font-weight:600;background:#231108;padding:1px 6px;border-radius:4px}
/* ---- burn banner ---- */
@keyframes bp{0%,100%{box-shadow:0 0 0 1px rgba(244,63,94,.5),0 0 16px rgba(244,63,94,.25);}
              50%{box-shadow:0 0 0 1px rgba(244,63,94,.8),0 0 28px rgba(244,63,94,.5);}}
.burn-banner{animation:bp 1.4s ease-in-out infinite;border-radius:10px;padding:11px 18px;
  background:linear-gradient(135deg,#1c0a10,#160810);border:1px solid #f43f5e;
  font-family:var(--font-mono);font-size:13px;margin-bottom:10px;color:#ffd7de;}
/* ---- pipeline trace ---- */
.ps{background:#0e1524;border-left:3px solid #4d9fff;
  padding:5px 11px;margin:3px 0;border-radius:0 5px 5px 0;
  font-family:var(--font-mono);font-size:11px;}
/* ---- mission story banner ---- */
.story{border-radius:11px;padding:15px 20px;margin-bottom:12px;
  font-size:15px;line-height:1.6;font-family:var(--font-ui);
  background:linear-gradient(135deg,#111726,#0e1420);
  border:1px solid #1e2740;border-left:3px solid #4d9fff;color:#dfe6f2;}
.story b{color:#7dd3fc;font-weight:600}
.story .hl-r{color:#f43f5e;font-weight:700}
.story .hl-o{color:#fb923c;font-weight:700}
.story .hl-g{color:#34d399;font-weight:700}
/* ---- ACAS pipeline stepper ---- */
.stepper{display:flex;gap:8px;margin-bottom:14px;align-items:stretch;}
.step{flex:1;background:#121826;border:1px solid #1e2740;border-radius:10px;
  padding:11px 14px;min-width:0;position:relative;}
.step .st-n{font-family:var(--font-ui);font-size:9px;font-weight:700;
  letter-spacing:1px;color:#7382a0;display:block;text-transform:uppercase}
.step .st-t{font-family:var(--font-ui);font-size:11.5px;
  color:#c4d0e6;display:block;margin-top:5px;line-height:1.4}
.step-ok{border-color:rgba(52,211,153,.4)}.step-ok .st-n{color:#34d399}
.step-hot{border-color:#f43f5e;background:linear-gradient(135deg,#1c0a10,#141826);
  box-shadow:0 0 16px rgba(244,63,94,.2)}
.step-hot .st-n{color:#f43f5e}
.step-warn{border-color:rgba(251,146,60,.45)}.step-warn .st-n{color:#fb923c}
/* ---- outcome KPI tiles ---- */
.kpi-row{display:flex;gap:8px;margin-bottom:14px}
.kpi{flex:1;background:#121826;border:1px solid #1e2740;border-radius:10px;
  padding:13px 15px;text-align:center}
.kpi .k-l{font-family:var(--font-ui);font-size:9px;font-weight:600;
  letter-spacing:1px;color:#7382a0;text-transform:uppercase;display:block}
.kpi .k-v{font-family:var(--font-mono);font-size:22px;font-weight:600;
  color:#7dd3fc;display:block;margin-top:5px;letter-spacing:-.5px}
.kpi .k-s{font-family:var(--font-ui);font-size:10px;
  color:#8492ab;display:block;margin-top:3px}
.k-good{color:#34d399 !important}.k-bad{color:#f43f5e !important}
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
AC = {"GREEN":"#34d399","YELLOW":"#fbbf24","ORANGE":"#fb923c","RED":"#f43f5e"}


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
        return "CRITICAL","#f43f5e","rgba(244,63,94,.10)","rgba(244,63,94,.5)"
    if overall in [Alert.ORANGE,Alert.YELLOW] or bat_now<30 or fuel_now<20:
        return "ELEVATED","#fbbf24","rgba(251,191,36,.08)","rgba(251,191,36,.45)"
    return "NOMINAL","#34d399","rgba(52,211,153,.07)","rgba(52,211,153,.4)"

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
        '#burnbadge{position:absolute;top:7px;left:8px;\n'
        '  font:bold 10px "Share Tech Mono",monospace;color:#ff7733;\n'
        '  background:#180500;border:1px solid #ff5511;border-radius:3px;\n'
        '  padding:3px 11px;letter-spacing:2px;display:none;pointer-events:none;\n'
        '  animation:bpulse 0.8s ease-in-out infinite alternate;}\n'
        '@keyframes bpulse{from{opacity:.65}to{opacity:1}}\n'
        '#legend{position:absolute;bottom:20px;right:8px;\n'
        '  font:9px "Share Tech Mono",monospace;color:#3a4a66;\n'
        '  background:rgba(2,4,14,.72);border:1px solid #0d1830;border-radius:4px;\n'
        '  padding:6px 9px;line-height:1.7;letter-spacing:1px;pointer-events:none;}\n'
        '#legend .sw{display:inline-block;width:8px;height:8px;border-radius:50%;\n'
        '  margin-right:6px;vertical-align:-1px;}\n'
        '</style></head><body>\n'
        '<canvas id="c"></canvas>\n'
        '<div id="hud">Drag · Scroll zoom · real ECI seed</div>\n'
        '<div id="badge"></div>\n'
        '<div id="burnbadge">🔥 BURN IN PROGRESS</div>\n'
        '<div id="legend">'
        '<span class="sw" style="background:#ffc233"></span>Power House (satellite)<br>'
        '<span class="sw" style="background:#fbbf24"></span>Threat · Watch<br>'
        '<span class="sw" style="background:#fb923c"></span>Threat · Caution<br>'
        '<span class="sw" style="background:#f43f5e"></span>Threat · Critical<br>'
        '<span class="sw" style="background:#34d399;border-radius:0;height:2px;vertical-align:2px"></span>Post-burn trajectory'
        '</div>\n'
        f'<script>{THREE_JS_SRC}</script>\n'
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
        '// Atmosphere rim glow (back-side shell reads as a blue limb halo)\n'
        'scene.add(new THREE.Mesh(new THREE.SphereGeometry(4.16,48,48),\n'
        '  new THREE.MeshBasicMaterial({color:0x1a4dbb,transparent:true,opacity:.10,side:THREE.BackSide,depthWrite:false,blending:THREE.AdditiveBlending})));\n'
        'scene.add(new THREE.Mesh(new THREE.SphereGeometry(4.30,48,48),\n'
        '  new THREE.MeshBasicMaterial({color:0x0d2a77,transparent:true,opacity:.05,side:THREE.BackSide,depthWrite:false,blending:THREE.AdditiveBlending})));\n'
        '// Grid\n'
        'scene.add(new THREE.Mesh(new THREE.SphereGeometry(4.01,36,18),\n'
        '  new THREE.MeshBasicMaterial({color:0x0b2850,wireframe:true,transparent:true,opacity:.07})));\n'
        '// Equatorial ring\n'
        '(()=>{const pts=[];for(let i=0;i<=360;i++){const a=i*Math.PI/180;pts.push(new THREE.Vector3(4.02*Math.cos(a),0,4.02*Math.sin(a)));}scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0x112a55,transparent:true,opacity:.4})));})();\n'
        '\n'
        '// Orbit track\n'
        '(()=>{const pts=[];for(let i=0;i<=200;i++){const M_=i/200*2*Math.PI;const s=kep2eci(SAT_A,.0001,SAT_I,SAT_RAAN,SAT_W,M_);pts.push(e2t(s.px,s.py,s.pz));}scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0x005577,transparent:true,opacity:.40})));})();\n'
        '\n'
        '// Text label sprite helper (canvas texture — no external fonts)\n'
        'function makeLabel(text,color,scale){\n'
        '  const cv=document.createElement("canvas");cv.width=256;cv.height=64;\n'
        '  const ctx=cv.getContext("2d");\n'
        '  ctx.font="bold 26px monospace";ctx.textAlign="center";ctx.textBaseline="middle";\n'
        '  ctx.shadowColor="#000";ctx.shadowBlur=6;\n'
        '  ctx.fillStyle=color;ctx.fillText(text,128,32);\n'
        '  const tex=new THREE.CanvasTexture(cv);\n'
        '  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthWrite:false}));\n'
        '  sp.scale.set(1.5*scale,.38*scale,1);\n'
        '  return sp;\n'
        '}\n'
        '\n'
        '// Satellite — GOLD, clearly visible, labeled\n'
        'const satG=new THREE.Group();\n'
        'const coreMat=new THREE.MeshPhongMaterial({color:0xffc233,emissive:0x301800,specular:0xffe88a,shininess:100});\n'
        'const core=new THREE.Mesh(new THREE.SphereGeometry(.10,16,16),coreMat);\n'
        'satG.add(core);\n'
        'satG.add(new THREE.Mesh(new THREE.BoxGeometry(.22,.06,.06),new THREE.MeshPhongMaterial({color:0xc0ccd8,specular:0x5577aa,shininess:60})));\n'
        '[-1,1].forEach(s=>{const p=new THREE.Mesh(new THREE.BoxGeometry(.46,.004,.14),new THREE.MeshPhongMaterial({color:0x0a1f5c,emissive:0x04102e,specular:0x3355aa,shininess:80}));p.position.x=s*.34;satG.add(p);});\n'
        'const glowMat=new THREE.MeshBasicMaterial({color:0xffc233,transparent:true,opacity:.14});\n'
        'satG.add(new THREE.Mesh(new THREE.SphereGeometry(.19,12,12),glowMat));\n'
        'scene.add(satG);\n'
        'const satLabel=makeLabel("POWER HOUSE","#ffc233",1.0);\n'
        'scene.add(satLabel);\n'
        '\n'
        '// Fading orbit trail behind the satellite\n'
        'const TRAIL_N=90;\n'
        'const trailPos=new Float32Array(TRAIL_N*3);\n'
        'const trailGeo=new THREE.BufferGeometry();\n'
        'trailGeo.setAttribute("position",new THREE.BufferAttribute(trailPos,3));\n'
        'const trail=new THREE.Line(trailGeo,new THREE.LineBasicMaterial({color:0xffc233,transparent:true,opacity:.5}));\n'
        'scene.add(trail);\n'
        'let trailInit=false;\n'
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
        'const NP=560;\n'
        'const pPos=new Float32Array(NP*3),pCol=new Float32Array(NP*3);\n'
        'const pVel=Array.from({length:NP},()=>new THREE.Vector3());\n'
        'const pLife=new Float32Array(NP),pMax=new Float32Array(NP);\n'
        'const pGeo=new THREE.BufferGeometry();\n'
        'pGeo.setAttribute("position",new THREE.BufferAttribute(pPos,3));\n'
        'pGeo.setAttribute("color",new THREE.BufferAttribute(pCol,3));\n'
        'const parts=new THREE.Points(pGeo,new THREE.PointsMaterial({size:.042,vertexColors:true,transparent:true,opacity:.95,depthWrite:false,sizeAttenuation:true,blending:THREE.AdditiveBlending}));\n'
        'parts.visible=false;scene.add(parts);\n'
        '// Flickering nozzle light + ignition shockwave ring\n'
        'const burnLight=new THREE.PointLight(0xff7722,0,3.5);\n'
        'scene.add(burnLight);\n'
        'const shock=new THREE.Mesh(new THREE.RingGeometry(.02,.05,40),\n'
        '  new THREE.MeshBasicMaterial({color:0xffaa55,transparent:true,opacity:0,side:THREE.DoubleSide,depthWrite:false,blending:THREE.AdditiveBlending}));\n'
        'scene.add(shock);\n'
        'let shockT=-1;\n'
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
        '  // color ramp: white-hot core → orange → deep red tail\n'
        '  const t=Math.random();\n'
        '  if(t<.25){pCol[j]=1;pCol[j+1]=1;pCol[j+2]=.92;}\n'
        '  else if(t<.6){pCol[j]=1;pCol[j+1]=.55+t*.3;pCol[j+2]=.08;}\n'
        '  else{pCol[j]=1;pCol[j+1]=.18+t*.15;pCol[j+2]=0;}\n'
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
        '// Debris objects — alert-coloured, labelled, RED ones get a pulsing ring\n'
        'const AC3={GREEN:0x34d399,YELLOW:0xfbbf24,ORANGE:0xfb923c,RED:0xf43f5e};\n'
        'const ACH={GREEN:"#34d399",YELLOW:"#fbbf24",ORANGE:"#fb923c",RED:"#f43f5e"};\n'
        'const debObjs=DEBRIS.map(d=>{\n'
        '  const col=AC3[d.a]||0xaaaaaa;\n'
        '  const mesh=new THREE.Mesh(new THREE.OctahedronGeometry(.085,0),new THREE.MeshPhongMaterial({color:col,emissive:col,emissiveIntensity:.7,flatShading:true}));\n'
        '  scene.add(mesh);\n'
        '  // dashed approach line satellite↔debris\n'
        '  const lGeo=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]);\n'
        '  const line=new THREE.Line(lGeo,new THREE.LineDashedMaterial({color:col,transparent:true,opacity:.55,dashSize:.12,gapSize:.08}));\n'
        '  scene.add(line);\n'
        '  // a bright bead that slides ALONG the line toward the satellite → shows closing motion\n'
        '  const bead=new THREE.Mesh(new THREE.SphereGeometry(.035,10,10),\n'
        '    new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:.95}));\n'
        '  scene.add(bead);\n'
        '  const label=makeLabel(d.n,ACH[d.a]||"#ccc",.66);\n'
        '  scene.add(label);\n'
        '  let ring=null;\n'
        '  if(d.a==="RED"||d.a==="ORANGE"){\n'
        '    ring=new THREE.Mesh(new THREE.RingGeometry(.13,.16,32),\n'
        '      new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:.8,side:THREE.DoubleSide,depthWrite:false}));\n'
        '    scene.add(ring);\n'
        '  }\n'
        '  return{mesh,line,bead,label,ring,rp:d.r,alert:d.a};\n'
        '});\n'
        '\n'
        '// Alert badge\n'
        '(()=>{\n'
        '  if(!DEBRIS.length)return;\n'
        '  const ord={GREEN:0,YELLOW:1,ORANGE:2,RED:3};\n'
        '  const w=DEBRIS.reduce((a,b)=>ord[b.a]>ord[a.a]?b:a,DEBRIS[0]);\n'
        '  if(w.a==="GREEN")return;\n'
        '  const cc={YELLOW:"#fbbf24",ORANGE:"#fb923c",RED:"#f43f5e"};\n'
        '  const bg={YELLOW:"#1a1405",ORANGE:"#1a0e05",RED:"#1a0510"};\n'
        '  const el=document.getElementById("badge");\n'
        '  el.style.color=cc[w.a]||"#fff";el.style.borderColor=cc[w.a]||"#fff";el.style.background=bg[w.a]||"#000";\n'
        '  el.textContent="⚠ "+w.a+" — "+w.n;el.style.display="block";\n'
        '})();\n'
        '\n'
        '// Camera\n'
        'let drag=false,userMoved=false,prev={x:0,y:0},cam={th:.7,ph:1.0,r:13};\n'
        'canvas.addEventListener("mousedown",e=>{drag=true;userMoved=true;prev={x:e.clientX,y:e.clientY};});\n'
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
        '  satLabel.position.copy(satPos).add(new THREE.Vector3(0,.34,0));\n'
        '  // Orbit trail: seed once, then shift and append current position\n'
        '  if(!trailInit){for(let i=0;i<TRAIL_N;i++){trailPos[i*3]=satPos.x;trailPos[i*3+1]=satPos.y;trailPos[i*3+2]=satPos.z;}trailInit=true;}\n'
        '  else{for(let i=0;i<TRAIL_N-1;i++){trailPos[i*3]=trailPos[(i+1)*3];trailPos[i*3+1]=trailPos[(i+1)*3+1];trailPos[i*3+2]=trailPos[(i+1)*3+2];}\n'
        '    trailPos[(TRAIL_N-1)*3]=satPos.x;trailPos[(TRAIL_N-1)*3+1]=satPos.y;trailPos[(TRAIL_N-1)*3+2]=satPos.z;}\n'
        '  trailGeo.attributes.position.needsUpdate=true;\n'
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
        '    coreMat.color.setHex(0xffc233);\n'
        '    coreMat.emissive.setHex(0x221200);\n'
        '    glowMat.color.setHex(0xffc233);\n'
        '    glowMat.opacity=.10+.03*Math.sin(tR*1.5);\n'
        '  }\n'
        '  // Velocity arrow (dim blue)\n'
        '  const vEnd=satPos.clone().add(vT.clone().multiplyScalar(.6));\n'
        '  {const p=vArr.geometry.attributes.position;p.setXYZ(0,satPos.x,satPos.y,satPos.z);p.setXYZ(1,vEnd.x,vEnd.y,vEnd.z);p.needsUpdate=true;}\n'
        '  // Debris\n'
        '  debObjs.forEach(d=>{\n'
        '    const dPos=e2t(st.px+d.rp[0],st.py+d.rp[1],st.pz+d.rp[2]);\n'
        '    d.mesh.position.copy(dPos);\n'
        '    d.mesh.rotation.y=tR*1.4;d.mesh.rotation.x=tR*.9;\n'
        '    d.mesh.material.emissiveIntensity=.5+.3*Math.sin(tR*4);\n'
        '    d.label.position.copy(dPos).add(new THREE.Vector3(0,.24,0));\n'
        '    if(d.ring){\n'
        '      d.ring.position.copy(dPos);d.ring.lookAt(camera.position);\n'
        '      const pu=d.alert==="RED"?1+.45*Math.abs(Math.sin(tR*5)):1+.2*Math.abs(Math.sin(tR*2.5));\n'
        '      d.ring.scale.set(pu,pu,pu);\n'
        '      d.ring.material.opacity=d.alert==="RED"?.5+.4*Math.abs(Math.sin(tR*5)):.35+.2*Math.abs(Math.sin(tR*2.5));\n'
        '    }\n'
        '    const lp=d.line.geometry.attributes.position;\n'
        '    lp.setXYZ(0,dPos.x,dPos.y,dPos.z);\n'
        '    lp.setXYZ(1,satPos.x,satPos.y,satPos.z);\n'
        '    lp.needsUpdate=true;\n'
        '    d.line.computeLineDistances();\n'
        '    // bead slides debris→satellite on a loop, conveying the closing approach\n'
        '    const bt=(tR*(d.alert==="RED"?.55:.3))%1;\n'
        '    d.bead.position.lerpVectors(dPos,satPos,bt);\n'
        '    d.bead.material.opacity=.95*(1-bt*.7);\n'
        '  });\n'
        '  // Burn exhaust — fires OPPOSITE to ΔV (exhaust = -ΔV direction)\n'
        '  if(BURN&&dvDir3.length()>.001){\n'
        '    parts.visible=true;\n'
        '    document.getElementById("burnbadge").style.display="block";\n'
        '    if(!pInited){for(let i=0;i<NP;i++)spawnP(i,satPos);pInited=true;shockT=tR;}\n'
        '    // flickering nozzle light\n'
        '    burnLight.position.copy(satPos).add(exhDir.clone().multiplyScalar(.12));\n'
        '    burnLight.intensity=2.2+Math.random()*1.6;\n'
        '    // ignition shockwave: expand + fade over ~1.2s\n'
        '    if(shockT>=0){\n'
        '      const sAge=tR-shockT;\n'
        '      if(sAge<1.2){\n'
        '        shock.position.copy(satPos);shock.lookAt(camera.position);\n'
        '        const ss=1+sAge*9;shock.scale.set(ss,ss,ss);\n'
        '        shock.material.opacity=.85*(1-sAge/1.2);\n'
        '      }else{shock.material.opacity=0;}\n'
        '    }\n'
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
        '  }else{parts.visible=false;dvArr.visible=false;burnLight.intensity=0;shock.material.opacity=0;document.getElementById("burnbadge").style.display="none";}\n'
        '  // Earth slow rotation\n'
        '  earth.rotation.y+=.000028;\n'
        '  // Camera — gentle auto-orbit until the user drags\n'
        '  if(!drag&&!userMoved)cam.th+=.0009;\n'
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
        f'<div class="burn-banner">'
        f'<b style="color:#f43f5e;letter-spacing:.8px">● THRUSTERS FIRING</b>'
        f' &nbsp;&nbsp;<span style="color:#fca5b4">ΔV [{dv_ss[0]:.3f}, {dv_ss[1]:.3f}, {dv_ss[2]:.3f}] m/s'
        f' · |ΔV| {np.linalg.norm(dv_ss):.3f} m/s</span>'
        f' &nbsp;&nbsp;<span style="color:#8492ab">ECI [{bp[0]:.0f}, {bp[1]:.0f}, {bp[2]:.0f}] km</span>'
        f' &nbsp;&nbsp;<span style="color:{"#fbbf24" if not ground else "#34d399"};font-weight:600">'
        f'{"AUTONOMOUS" if not ground else "GROUND-CONFIRMED"}</span>'
        f'</div>', unsafe_allow_html=True)


# ============================================================
# MISSION STORY + LIVE PIPELINE STEPPER + OUTCOME KPIs
# (plain-language layer: a reviewer reads this row and understands
#  what the AI just did without decoding any telemetry)
# ============================================================
def pc_to_odds(pc):
    """8.5e-3 → '1 in 118' — collision probability in human terms."""
    if pc is None or pc <= 0: return "—"
    odds = 1.0 / max(pc, 1e-12)
    if odds >= 1e9: return f"1 in {odds/1e9:.1f} billion"
    if odds >= 1e6: return f"1 in {odds/1e6:.1f} million"
    return f"1 in {odds:,.0f}"

ACTION_WORDS = {
    "NO_ACTION":        "no action needed",
    "YELLOW_ALERT":     "ground notified — monitoring",
    "ORANGE_DOWNLINK":  "manoeuvre plan downlinked to ground",
    "ORANGE_QUEUED":    "burn queued (auto-executes if TCA < 2 h)",
    "ORANGE_AUTONOMOUS":"autonomous avoidance burn",
    "RED_GROUND":       "ground-confirmed avoidance burn",
    "RED_AUTONOMOUS":   "autonomous avoidance burn (no ground link)",
}

worst = (max(all_results, key=lambda r: ORDER.index(r['assessment'].alert))
         if all_results else None)

# ── one-sentence mission narrative ──
if burn_active and np.linalg.norm(dv_ss) > 0 and st.session_state.last_result:
    lr_  = st.session_state.last_result
    c_   = lr_['conjunction']; a_ = lr_['assessment']
    dvm_ = float(np.linalg.norm(dv_ss))
    _shift = dvm_ * c_['tca_hours'] * 3600.0
    proj_miss = min(5.0, c_['miss_km'] + _shift / 1000.0)
    _proj_txt = f'{proj_miss:.1f} km' if proj_miss >= 1 else f'{proj_miss*1000:.0f} m'
    story = (f'🔥 <span class="hl-r">{c_["object_name"]}</span> was predicted to pass '
             f'<b>{c_["miss_km"]*1000:.0f} m</b> from the satellite '
             f'(collision odds <span class="hl-r">{pc_to_odds(a_.adjusted_pc)}</span>). '
             f'ACAS executed a <b>{dvm_:.3f} m/s</b> '
             f'{"<b>autonomous</b> burn — no ground link needed" if not ground else "ground-confirmed burn"}. '
             f'Projected separation at closest approach: <span class="hl-g">{_proj_txt}</span>.')
elif worst and worst['assessment'].alert in (Alert.RED, Alert.ORANGE):
    c_, a_ = worst['conjunction'], worst['assessment']
    hl = "hl-r" if a_.alert == Alert.RED else "hl-o"
    story = (f'⚠ <span class="{hl}">{c_["object_name"]}</span> will pass within '
             f'<b>{c_["miss_km"]*1000:.0f} m</b> in <b>{c_["tca_hours"]:.1f} h</b> — '
             f'AI predicts collision odds of <span class="{hl}">{pc_to_odds(a_.adjusted_pc)}</span> '
             f'→ <span class="{hl}">{a_.alert.value}</span>. '
             f'Decision: <b>{ACTION_WORDS.get(worst["action"], worst["action"])}</b>.')
elif worst and worst['assessment'].alert == Alert.YELLOW:
    c_ = worst['conjunction']
    story = (f'<span style="color:#fbbf24;font-weight:600">{c_["object_name"]}</span> is being tracked '
             f'(closest pass {c_["miss_km"]:.1f} km in {c_["tca_hours"]:.0f} h) — '
             f'below manoeuvre threshold. <b>Ground notified, monitoring every 60 s.</b>')
else:
    story = ('<span class="hl-g">✓ All clear.</span> ACAS screens the debris catalogue '
             'every <b>60 seconds</b> — no object inside the 5 km corridor exceeds '
             'the 1-in-100,000 alert threshold. The AI decides in <b>&lt;100 ms</b> per threat.')
st.markdown(f'<div class="story">{story}</div>', unsafe_allow_html=True)

# ── live pipeline stepper: the architecture, visible ──
def _step(name, text, cls=""):
    return (f'<div class="step {cls}"><span class="st-n">{name}</span>'
            f'<span class="st-t">{text}</span></div>')

if worst:
    c_, a_ = worst['conjunction'], worst['assessment']
    is_hot  = a_.alert in (Alert.RED, Alert.ORANGE)
    alert_cls = "step-hot" if a_.alert == Alert.RED else "step-warn" if a_.alert == Alert.ORANGE else "step-ok"
    burned  = burn_active and np.linalg.norm(dv_ss) > 0
    stepper = (
        _step("1 · Detect", f'{len(all_results)} object(s) inside 5 km screening corridor', "step-ok") +
        _step("2 · Predict", f'LightGBM + physics: Pc {a_.raw_pc:.1e} ({pc_to_odds(a_.raw_pc)})', "step-ok") +
        _step("3 · Assess", f'6 operational limits → <b>{a_.alert.value}</b>', alert_cls) +
        _step("4 · Decide", ACTION_WORDS.get(worst["action"], worst["action"]),
              alert_cls if is_hot else "") +
        _step("5 · Act", ("🔥 burn executed — trajectory shifted" if burned else
                          "standing by" if not is_hot else "awaiting execution window"),
              "step-hot" if burned else "")
    )
else:
    stepper = (
        _step("1 · Detect", "screening 5 km corridor — clear", "step-ok") +
        _step("2 · Predict", "LightGBM + Foster physics cross-check idle") +
        _step("3 · Assess", "6 operational limits · thresholds from config") +
        _step("4 · Decide", "governance gate: ground veto / autonomy rules") +
        _step("5 · Act", "thrusters standing by")
    )
st.markdown(f'<div class="stepper">{stepper}</div>', unsafe_allow_html=True)

# ── outcome KPIs after a burn: the quantifiable-improvement evidence ──
if burn_active and np.linalg.norm(dv_ss) > 0 and st.session_state.last_result:
    lr_ = st.session_state.last_result
    c_  = lr_['conjunction']
    dvm_ = float(np.linalg.norm(dv_ss))
    shift_m   = dvm_ * c_['tca_hours'] * 3600.0            # along-track drift by TCA
    proj_miss = min(5.0, c_['miss_km'] + shift_m / 1000.0)
    proj_txt  = f'{proj_miss:.1f} km' if proj_miss >= 1 else f'{proj_miss*1000:.0f} m'
    fuel_g = st.session_state.last_fuel_cost / 100.0 * 2000.0   # 2 kg tank
    fuel_txt = f'{fuel_g:.1f} g' if fuel_g >= 1 else f'{fuel_g*1000:.0f} mg'
    st.markdown(
        f'<div class="kpi-row">'
        f'<div class="kpi"><span class="k-l">Closest pass · before</span>'
        f'<span class="k-v k-bad">{c_["miss_km"]*1000:.0f} m</span>'
        f'<span class="k-s">{c_["object_name"]}</span></div>'
        f'<div class="kpi"><span class="k-l">Projected · after burn</span>'
        f'<span class="k-v k-good">{proj_txt}</span>'
        f'<span class="k-s">+{shift_m:.0f} m displacement by closest approach</span></div>'
        f'<div class="kpi"><span class="k-l">AI decision time</span>'
        f'<span class="k-v">&lt;100 ms</span>'
        f'<span class="k-s">vs hours of manual screening</span></div>'
        f'<div class="kpi"><span class="k-l">Fuel spent</span>'
        f'<span class="k-v">{fuel_txt}</span>'
        f'<span class="k-s">{st.session_state.last_fuel_cost:.4f}% of tank</span></div>'
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

        st.markdown('<div class="sec">Model Burn Output</div>', unsafe_allow_html=True)
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
            f'<div class="trow"><span class="tkey">Collision odds</span>'
            f'<span class="tval tv-r">{pc_to_odds(asm_lr.adjusted_pc) if asm_lr else "—"}'
            f' <span style="color:#44526b">({(f"{asm_lr.adjusted_pc:.2e}") if asm_lr else "—"})</span></span></div>'
            f'<div class="trow"><span class="tkey">Fuel Cost</span>'
            f'<span class="tval tv-o">{st.session_state.last_fuel_cost:.4f} %</span></div>'
            f'<div class="trow"><span class="tkey">Fuel Remaining</span>'
            f'<span class="tval {fuel_c}">{fuel_now:.3f} %</span></div>'
            f'</div>', unsafe_allow_html=True)

        # Pipeline trace for the latest result
        if lr:
            with st.expander("Pipeline trace — feature extraction → Pc → decision", expanded=False):
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
            mc2="tv-r" if c['miss_km']<1 else "tv-o" if c['miss_km']<2 else "tval"
            st.markdown(
                f'<div class="rcard" style="border-left:3px solid {BC}">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
                f'<span style="width:9px;height:9px;border-radius:50%;background:{BC};'
                f'box-shadow:0 0 7px {BC};display:inline-block"></span>'
                f'<b style="color:{BC};font-size:13px">{c["object_name"]}</b>'
                f'<span style="margin-left:auto;font-size:10px;font-weight:700;'
                f'letter-spacing:.5px;color:{BC}">{a.alert.value}</span></div>'
                f'<br><div class="trow"><span class="tkey">Miss Distance</span>'
                f'<span class="tval {mc2}">{c["miss_km"]:.3f} km</span></div>'
                f'<div class="trow"><span class="tkey">TCA</span>'
                f'<span class="tval">{c["tca_hours"]:.2f} h</span></div>'
                f'<div class="trow"><span class="tkey">Collision odds</span>'
                f'<span class="tval">{pc_to_odds(a.adjusted_pc)}'
                f' <span style="color:#44526b">({a.adjusted_pc:.2e})</span></span></div>'
                f'<div class="trow"><span class="tkey">Decision</span>'
                f'<span class="tval">{ACTION_WORDS.get(r["action"], r["action"])}</span></div>'
                f'</div>', unsafe_allow_html=True)
            with st.expander("Pipeline trace", expanded=False):
                html_t='<div class="mono">'
                for pe in r['log']:
                    html_t+=(f'<div class="ps"><span style="color:#2244cc">S{pe["s"]}</span> '
                             f'<span class="{pe["c"]}">{pe["m"]}</span></div>')
                html_t+='</div>'
                st.markdown(html_t, unsafe_allow_html=True)

    # §2.3 — DEMO SCENARIOS
    def _inject_scenario(sc):
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

    st.markdown('<div class="sec">Demo · Inject a Simulated Threat</div>',
                unsafe_allow_html=True)
    st.caption("Each button feeds one realistic conjunction into the live pipeline "
               "above: detection → AI prediction → risk assessment → decision → burn.")
    if st.button("▶  Run Full Demo — Critical Debris → Autonomous Burn",
                 type="primary", use_container_width=True):
        red_sc = next((s for s in SCENARIOS if s['alert']=="RED"), SCENARIOS[-1])
        _inject_scenario(red_sc)
        st.rerun()

    for sc in SCENARIOS:
        col_a=AC[sc['alert']]
        c1,c2=st.columns([4,1])
        with c1:
            spd_sc=np.linalg.norm(sc['rv'])
            miss_txt=(f'{sc["miss_km"]*1000:.0f} m' if sc["miss_km"]<1
                      else f'{sc["miss_km"]:.1f} km')
            st.markdown(
                f'<div class="sc-card" style="border-left:3px solid {col_a}">'
                f'<div style="display:flex;align-items:center;gap:7px">'
                f'<span style="width:8px;height:8px;border-radius:50%;background:{col_a};'
                f'display:inline-block"></span>'
                f'<span style="color:#eaf1ff;font-weight:600;font-size:12.5px">{sc["name"]}</span>'
                f'<span style="margin-left:auto;color:{col_a};font-size:9px;'
                f'font-weight:700;letter-spacing:.5px">{sc["alert"]}</span></div>'
                f'<div style="color:#9aa7c0;font-size:11px;margin-top:4px">'
                f'Passes within <b style="color:{col_a}">{miss_txt}</b> in {sc["tca_h"]:.1f} h, '
                f'closing at {spd_sc:.1f} km/s'
                f'{" · stale tracking data" if sc["stale"] else ""}</div>'
                f'<div style="color:#5c6a85;font-size:10px;margin-top:2px">{sc["desc"]}</div>'
                f'</div>', unsafe_allow_html=True)
        with c2:
            if st.button("Inject", key=f"sc_{sc['norad']}", use_container_width=True):
                _inject_scenario(sc)
                st.rerun()

    # ACAS event log
    st.markdown('<div class="sec">Event Log</div>', unsafe_allow_html=True)
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
        '<div style="padding:2px 0 10px">'
        '<div style="font-size:19px;font-weight:700;color:#eaf1ff;letter-spacing:.5px">'
        'POWER&nbsp;HOUSE</div>'
        '<div style="font-size:10px;font-weight:600;color:#6b7a99;letter-spacing:1.5px;'
        'text-transform:uppercase;margin-top:2px">IN-SPACe · ACAS</div>'
        '<div style="font-size:11px;color:#8492ab;margin-top:6px">'
        'Autonomous Collision Avoidance System</div>'
        '</div>', unsafe_allow_html=True)

    # Onboard model status — compact professional readout
    def _status_chip(ok, label, detail):
        c   = "#34d399" if ok else "#fb923c"
        dot = "●" if ok else "▲"
        state = "ONLINE" if ok else "FALLBACK"
        return (f'<div style="display:flex;align-items:center;gap:9px;'
                f'background:#111726;border:1px solid #1e2740;border-radius:8px;'
                f'padding:8px 12px;margin-bottom:6px">'
                f'<span style="color:{c};font-size:11px">{dot}</span>'
                f'<div style="flex:1"><div style="font-size:11px;font-weight:600;'
                f'color:#c4d0e6">{label}</div>'
                f'<div style="font-size:9.5px;color:#6b7a99">{detail}</div></div>'
                f'<span style="font-size:9px;font-weight:700;letter-spacing:.5px;'
                f'color:{c}">{state}</span></div>')
    st.markdown('<div class="sec">Onboard AI Stack</div>', unsafe_allow_html=True)
    st.markdown(
        _status_chip(M['onnx_ok'], "LightGBM Pc engine",
                     "collision-probability model" if M['onnx_ok'] else "physics fallback active") +
        _status_chip(M['rl_ok'], "RL manoeuvre policy",
                     "PPO agent loaded" if M['rl_ok'] else "geometric fallback active"),
        unsafe_allow_html=True)

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

        st.markdown('<div class="sec">Satellite State · ECI</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-family:var(--font-ui);font-size:9px;'
            f'color:#7382a0;letter-spacing:.8px;font-weight:600;margin-bottom:4px">POSITION (km)</div>'
            f'<div class="vg">'
            f'<div class="vc"><span class="va">X</span><span class="vn">{e2["pos_x_km"]:.1f}</span></div>'
            f'<div class="vc"><span class="va">Y</span><span class="vn">{e2["pos_y_km"]:.1f}</span></div>'
            f'<div class="vc"><span class="va">Z</span><span class="vn">{e2["pos_z_km"]:.1f}</span></div>'
            f'</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-family:var(--font-ui);font-size:9px;'
            f'color:#7382a0;letter-spacing:.8px;font-weight:600;margin-bottom:4px">VELOCITY (km/s)</div>'
            f'<div class="vg">'
            f'<div class="vc"><span class="va">Vx</span><span class="vn">{e2["vel_x_kms"]:.4f}</span></div>'
            f'<div class="vc"><span class="va">Vy</span><span class="vn">{e2["vel_y_kms"]:.4f}</span></div>'
            f'<div class="vc"><span class="va">Vz</span><span class="vn">{e2["vel_z_kms"]:.4f}</span></div>'
            f'</div>', unsafe_allow_html=True)

        # Fuel gauge
        fc="#f43f5e" if fuel2<20 else "#fbbf24" if fuel2<40 else "#34d399"
        st.markdown(
            f'<div class="g-wrap"><div class="g-head">'
            f'<span class="g-lbl">FUEL</span><span class="g-val">{fuel2:.2f}%</span>'
            f'</div><div class="g-bar">'
            f'<div class="g-fill" style="width:{fuel2:.1f}%;background:{fc}"></div>'
            f'</div></div>', unsafe_allow_html=True)

        # Battery gauge
        bc2="#f43f5e" if bat2<25 else "#fbbf24" if bat2<50 else "#38bdf8"
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
            f'{"GROUND LINK" if gnd2 else "NO CONTACT"}</span>'
            f'<span class="pill {"p-off" if ecl2 else "p-on"}">'
            f'{"ECLIPSE" if ecl2 else "SUNLIT"}</span>'
            f'</div>', unsafe_allow_html=True)

    sidebar_live()
    st.divider()

    # Custom threat injection form
    st.markdown('<div class="sec">Custom Threat Injection</div>', unsafe_allow_html=True)
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