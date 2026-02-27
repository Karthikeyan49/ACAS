# ─────────────────────────────────────────────────────────────────────────────
# dashboard/app.py  —  LIVE CONNECTED DASHBOARD
#
# Fixes from original:
#   ✅ Real SGP4 position: ISS tracked live, altitude/speed computed from TLE
#   ✅ Thruster animation: orange exhaust plume + green ΔV arrow on the globe
#   ✅ Live downlog: timestamped ECI position, velocity, altitude every cycle
#   ✅ Inject any threat via sidebar → real Pc calculated → real RED/ORANGE
#   ✅ Shares same RiskScorer as acas_controller.py
#
# HOW TO RUN:
#   streamlit run dashboard/app.py
# ─────────────────────────────────────────────────────────────────────────────

import sys, os, time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.risk_scorer import RiskScorer, SatState, Alert

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ACAS Live", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.main{background:#06060f;color:#e0e0ff}
.downlog{font-family:monospace;font-size:12px;line-height:1.8}
.lg{color:#00ff88}.ly{color:#ffdd00}.lo{color:#ff8800}
.lr{color:#ff3300;font-weight:bold}.li{color:#88bbff}
.lb{color:#ff6600;font-weight:bold;background:#220a00;
    padding:1px 6px;border-radius:3px}
</style>""", unsafe_allow_html=True)

# ── Our satellite: ISS — real tracked object ──────────────────────────────────
# Replace these two lines with any real satellite TLE from space-track.org
MY_TLE1 = "1 25544U 98067A   24001.50000000  .00005764  00000-0  10780-3 0  9993"
MY_TLE2 = "2 25544  51.6416 290.0015 0002627  55.4917 344.9690 15.49960988432698"

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init():
    defaults = dict(
        fuel=75.0, battery=88.0, ground=True, phase='nominal',
        downlog=[], maneuver_on=False, last_dv=np.zeros(3),
        cycle=0, threats=[], overall=Alert.GREEN,
        post_x=[], post_y=[], post_z=[]
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init()

# ─────────────────────────────────────────────────────────────────────────────
# REAL SGP4 — computes ISS position right now + 90-min orbit
# ─────────────────────────────────────────────────────────────────────────────
def sgp4_now():
    sat = Satrec.twoline2rv(MY_TLE1, MY_TLE2)
    now = datetime.utcnow()
    jd, fr = jday(now.year, now.month, now.day,
                  now.hour, now.minute, now.second + now.microsecond/1e6)
    _, r, v = sat.sgp4(jd, fr)
    pos = np.array(r); vel = np.array(v)
    alt = np.linalg.norm(pos) - 6371
    spd = np.linalg.norm(vel)
    # Lat/lon via simplified ECEF conversion
    gmst   = (280.46061837 + 360.98564736629*(jd+fr-2451545.0)) % 360
    th     = np.radians(gmst)
    xe     = pos[0]*np.cos(th)+pos[1]*np.sin(th)
    ye     = -pos[0]*np.sin(th)+pos[1]*np.cos(th)
    ze     = pos[2]
    lat    = np.degrees(np.arcsin(ze / np.linalg.norm(pos)))
    lon    = np.degrees(np.arctan2(ye, xe))
    # Full 90-min orbit
    ox,oy,oz = [],[],[]
    for i in range(90):
        t2 = now+timedelta(minutes=i)
        jd2,fr2 = jday(t2.year,t2.month,t2.day,t2.hour,t2.minute,t2.second)
        e2,r2,_ = sat.sgp4(jd2,fr2)
        if e2==0: ox.append(r2[0]);oy.append(r2[1]);oz.append(r2[2])
    return pos,vel,alt,spd,lat,lon,ox,oy,oz

def post_maneuver_path(pos,vel,dv):
    nv = vel + dv/1000.0
    px,py,pz = [],[],[]
    R = np.linalg.norm(pos)
    for i in range(45):
        np_ = pos + nv*(i*60.0)
        r2  = np.linalg.norm(np_)
        if r2>0: np_ = np_*(R/r2)
        px.append(np_[0]);py.append(np_[1]);pz.append(np_[2])
    return px,py,pz

# ─────────────────────────────────────────────────────────────────────────────
# LOG HELPER
# ─────────────────────────────────────────────────────────────────────────────
def log(msg, css="li", alert=None):
    if alert == Alert.GREEN:  css="lg"
    elif alert == Alert.YELLOW: css="ly"
    elif alert == Alert.ORANGE: css="lo"
    elif alert == Alert.RED:    css="lr"
    ts = datetime.utcnow().strftime("%H:%M:%S")
    st.session_state.downlog.insert(0, {'ts':ts,'msg':msg,'css':css})
    st.session_state.downlog = st.session_state.downlog[:120]

# ─────────────────────────────────────────────────────────────────────────────
# RISK ASSESSMENT — same scorer as acas_controller
# ─────────────────────────────────────────────────────────────────────────────
def assess(threats, sat):
    scorer = RiskScorer()
    out = []
    for c in threats:
        miss  = c['miss_km']; spd2 = c['rel_speed_kms']
        pc    = min((0.01/(miss+1e-10))**2 * spd2/7.8, 1.0)
        if c.get('tle_stale'): pc = min(pc*2.5, 1.0)
        a = scorer.assess(c, pc, sat)
        out.append((c, a))
    return out

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛰️ ACAS")
    st.caption("Autonomous Collision Avoidance System")
    st.divider()

    st.subheader("📡 Satellite Controls")
    st.session_state.fuel     = float(st.slider("Fuel (%)",0,100,int(st.session_state.fuel)))
    st.session_state.battery  = float(st.slider("Battery (%)",0,100,int(st.session_state.battery)))
    st.session_state.ground   = st.toggle("Ground Contact", value=st.session_state.ground)
    st.session_state.phase    = st.selectbox("Mission Phase",
                                             ["nominal","critical","safe_mode"])

    st.divider()
    st.subheader("☄️ Inject Threat")
    st.caption("Creates a debris conjunction → AI calculates real Pc → real alert")
    t_miss  = st.slider("Miss Distance (km)", 0.1, 4.9, 2.0, 0.1)
    t_tca   = st.slider("TCA (hours)",         0.5, 48.0, 4.0, 0.5)
    t_spd   = st.slider("Rel Speed (km/s)",    1.0, 15.0, 7.5, 0.5)
    t_stale = st.checkbox("TLE Stale (72h old)")

    if st.button("💥 Inject Threat", type="primary"):
        rp = np.array([t_miss*0.6*np.random.choice([-1,1]),
                       t_miss*0.5*np.random.choice([-1,1]),
                       t_miss*0.2])
        rv = np.array([-t_spd*0.88, t_spd*0.40, t_spd*0.22])
        c = {"object_id":   f"DEB-{np.random.randint(10000,99999)}",
             "object_name": f"DEBRIS {np.random.randint(1000,9999)}",
             "object_type": "DEBRIS",
             "miss_km": t_miss, "tca_hours": t_tca,
             "rel_pos": rp, "rel_vel": rv,
             "rel_speed_kms": t_spd,
             "tle_stale": t_stale,
             "tle_age_hours": 72.0 if t_stale else 8.0}
        st.session_state.threats.append(c)
        log(f"🎯 INJECTED: miss={t_miss:.2f}km TCA={t_tca:.1f}h "
            f"spd={t_spd:.1f}km/s {'[STALE]' if t_stale else ''}","lo")
        st.rerun()

    if st.button("🗑️ Clear Threats"):
        st.session_state.threats = []
        st.session_state.maneuver_on = False
        st.session_state.last_dv = np.zeros(3)
        st.session_state.post_x = []
        st.session_state.post_y = []
        st.session_state.post_z = []
        log("✅ All threats cleared","lg")
        st.rerun()

    st.divider()
    auto = st.checkbox("🔄 Auto-refresh (10s)")
    if st.button("▶ Run One Cycle", type="secondary"):
        st.session_state.cycle += 1
        st.rerun()

    st.caption(f"Cycle #{st.session_state.cycle}")
    st.caption(datetime.utcnow().strftime("%H:%M:%S UTC"))

# ─────────────────────────────────────────────────────────────────────────────
# GET LIVE SATELLITE DATA
# ─────────────────────────────────────────────────────────────────────────────
pos, vel, alt, spd, lat, lon, ox, oy, oz = sgp4_now()

sat = SatState(
    fuel_pct=st.session_state.fuel,
    battery_pct=st.session_state.battery,
    altitude_km=alt,
    ground_contact=st.session_state.ground,
    mission_phase=st.session_state.phase,
    min_altitude_km=300.0, total_fuel_kg=2.0
)

# ─────────────────────────────────────────────────────────────────────────────
# RUN ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
assessments = assess(st.session_state.threats, sat) if st.session_state.threats else []
order = [Alert.GREEN, Alert.YELLOW, Alert.ORANGE, Alert.RED]
overall = max([a.alert for _,a in assessments],
              key=lambda x: order.index(x)) if assessments else Alert.GREEN

if st.session_state.cycle > 0:
    if not assessments:
        log("✅ All clear. No conjunctions within 5km.","lg")
    else:
        for c,a in assessments:
            log(f"[{a.alert.value}] {c['object_name']} | "
                f"miss={c['miss_km']:.3f}km TCA={c['tca_hours']:.2f}h "
                f"Pc(raw)={a.raw_pc:.2e}→Pc(adj)={a.adjusted_pc:.2e}",
                alert=a.alert)
            for lim in a.limitations_hit:
                log(f"  ⚡ {lim}","lo")

# Execute maneuver if RED or ORANGE
if assessments:
    top_c, top_a = max(assessments, key=lambda x: order.index(x[1].alert))
    if top_a.alert in [Alert.RED, Alert.ORANGE] and np.linalg.norm(top_a.dv_vector)>0:
        dv = top_a.dv_vector
        dv_mag = np.linalg.norm(dv)
        st.session_state.maneuver_on = True
        st.session_state.last_dv = dv
        fc = dv_mag * 0.12
        st.session_state.fuel = max(0, st.session_state.fuel - fc)
        px,py,pz = post_maneuver_path(pos, vel, dv)
        st.session_state.post_x = px
        st.session_state.post_y = py
        st.session_state.post_z = pz
        mode = "AUTONOMOUS" if not sat.ground_contact else "GROUND-CONFIRMED"
        log(f"🔥 BURN [{mode}] ΔV=[{dv[0]:.2f},{dv[1]:.2f},{dv[2]:.2f}] m/s "
            f"mag={dv_mag:.2f}m/s fuel_cost={fc:.3f}% "
            f"remaining={st.session_state.fuel:.1f}%","lb")
    elif top_a.alert == Alert.GREEN:
        st.session_state.maneuver_on = False

# Telemetry log
if st.session_state.cycle > 0:
    log(f"TELEMETRY | pos=[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}]km | "
        f"vel=[{vel[0]:.3f},{vel[1]:.3f},{vel[2]:.3f}]km/s | "
        f"alt={alt:.1f}km spd={spd:.3f}km/s | "
        f"lat={lat:.2f}° lon={lon:.2f}° | "
        f"fuel={st.session_state.fuel:.1f}% bat={st.session_state.battery:.1f}%","li")

# ─────────────────────────────────────────────────────────────────────────────
# ALERT BANNER
# ─────────────────────────────────────────────────────────────────────────────
AB = {
    Alert.GREEN:  ("✅ ALL CLEAR — NOMINAL OPERATIONS", "#00ff88", "#0a1f0a"),
    Alert.YELLOW: ("⚠️ YELLOW — CONJUNCTION DETECTED",  "#ffdd00", "#1f1a00"),
    Alert.ORANGE: ("🟠 ORANGE — MANEUVER COMPUTED",     "#ff8800", "#1f0f00"),
    Alert.RED:    ("🔴 RED — EXECUTING AVOIDANCE",      "#ff2200", "#1f0000"),
}
lbl,bc,bg = AB[overall]
st.markdown(
    f'<div style="background:{bg};border:2px solid {bc};border-radius:8px;'
    f'padding:12px 20px;margin-bottom:10px;">'
    f'<h2 style="color:{bc};margin:0">{lbl}</h2>'
    f'<p style="color:#aaa;margin:4px 0 0;font-size:13px">'
    f'ISS (NORAD 25544) &nbsp;|&nbsp; Alt: {alt:.1f} km &nbsp;|&nbsp;'
    f' Spd: {spd:.3f} km/s &nbsp;|&nbsp; Lat: {lat:.2f}° Lon: {lon:.2f}°'
    f' &nbsp;|&nbsp; Threats: {len(st.session_state.threats)}</p></div>',
    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("🛢️ Fuel",      f"{st.session_state.fuel:.1f}%",
          delta="⚠️LOW" if st.session_state.fuel<15 else None,
          delta_color="inverse" if st.session_state.fuel<15 else "normal")
c2.metric("⚡ Battery",   f"{st.session_state.battery:.1f}%")
c3.metric("📡 Altitude",  f"{alt:.1f} km")
c4.metric("🚀 Speed",     f"{spd:.3f} km/s")
c5.metric("🌍 Lat/Lon",   f"{lat:.1f}°/{lon:.1f}°")
c6.metric("📶 Ground",    "YES" if sat.ground_contact else "BLACKOUT")
c7.metric("🎯 Threats",   str(len(st.session_state.threats)))

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 3D GLOBE + DOWNLOG
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([3,2])

with left:
    st.subheader("🌍 Live Orbital View")

    # Earth
    u = np.linspace(0,2*np.pi,80); v2 = np.linspace(0,np.pi,40)
    Re = 6371.0
    xe = Re*np.outer(np.cos(u),np.sin(v2))
    ye = Re*np.outer(np.sin(u),np.sin(v2))
    ze = Re*np.outer(np.ones(len(u)),np.cos(v2))

    fig = go.Figure()
    fig.add_trace(go.Surface(x=xe,y=ye,z=ze,
        colorscale=[[0,'#000820'],[0.5,'#002080'],[1,'#0040cc']],
        opacity=0.8,showscale=False,hoverinfo='skip',name="Earth"))

    # Real 90-min orbit path
    if ox:
        fig.add_trace(go.Scatter3d(x=ox,y=oy,z=oz,mode='lines',
            line=dict(color='cyan',width=2),name='Orbit Path (90 min)'))

    # Satellite marker
    sc = '#ff4400' if st.session_state.maneuver_on else '#00ffff'
    ss = 14 if st.session_state.maneuver_on else 10
    fig.add_trace(go.Scatter3d(
        x=[pos[0]],y=[pos[1]],z=[pos[2]],mode='markers',
        marker=dict(color=sc,size=ss,symbol='diamond' if st.session_state.maneuver_on else 'circle',
                    line=dict(color='white',width=2)),
        name=f'ISS {"🔥BURN" if st.session_state.maneuver_on else "✅OK"}',
        hovertemplate=f'ISS<br>Alt:{alt:.1f}km<br>Spd:{spd:.3f}km/s<br>Lat:{lat:.2f}° Lon:{lon:.2f}°'))

    # Velocity arrow
    vu = vel/(np.linalg.norm(vel)+1e-10)
    ae = pos+vu*500
    fig.add_trace(go.Scatter3d(x=[pos[0],ae[0]],y=[pos[1],ae[1]],z=[pos[2],ae[2]],
        mode='lines',line=dict(color='cyan',width=3),name='Velocity',hoverinfo='skip'))

    # ── THRUSTER EXHAUST + ΔV ARROW ──────────────────────────────────────
    dv = st.session_state.last_dv
    if st.session_state.maneuver_on and np.linalg.norm(dv)>0:
        dvu   = dv/(np.linalg.norm(dv)+1e-10)
        thrust_dir = -dvu   # exhaust goes opposite to movement

        # Exhaust plume: 3 rings of orange lines spreading behind satellite
        p1 = np.cross(dvu, np.array([0,0,1.0]))
        if np.linalg.norm(p1)<1e-10: p1 = np.cross(dvu,np.array([1,0,0.0]))
        p1 /= np.linalg.norm(p1)
        p2 = np.cross(dvu,p1)

        for ri,(spread,length,col_name) in enumerate(
            [(0.10,600,'#ff6600'),(0.20,400,'#ffaa00'),(0.30,250,'#ffff00')]):
            for ang in [0,90,180,270]:
                ar = np.radians(ang)
                pd = thrust_dir + spread*(np.cos(ar)*p1+np.sin(ar)*p2)
                pd /= (np.linalg.norm(pd)+1e-10)
                ep  = pos+pd*length
                fig.add_trace(go.Scatter3d(
                    x=[pos[0],ep[0]],y=[pos[1],ep[1]],z=[pos[2],ep[2]],
                    mode='lines',line=dict(color=col_name,width=max(1,4-ri)),
                    showlegend=(ri==0 and ang==0),
                    name='Thruster Exhaust' if (ri==0 and ang==0) else None,
                    hoverinfo='skip'))

        # ΔV arrow (lime green — direction satellite moves)
        dv_end = pos+dvu*700
        fig.add_trace(go.Scatter3d(
            x=[pos[0],dv_end[0]],y=[pos[1],dv_end[1]],z=[pos[2],dv_end[2]],
            mode='lines',line=dict(color='lime',width=5),
            name=f'ΔV {np.linalg.norm(dv):.1f}m/s',
            hovertemplate=f'ΔV=[{dv[0]:.2f},{dv[1]:.2f},{dv[2]:.2f}]m/s'))

    # Post-maneuver trajectory
    if st.session_state.post_x and st.session_state.maneuver_on:
        fig.add_trace(go.Scatter3d(
            x=st.session_state.post_x,y=st.session_state.post_y,z=st.session_state.post_z,
            mode='lines',line=dict(color='lime',width=3,dash='dash'),
            name='Post-Maneuver Path'))

    # Threat objects
    TC = {Alert.GREEN:'#00ff88',Alert.YELLOW:'#ffdd00',
          Alert.ORANGE:'#ff8800',Alert.RED:'#ff2200'}
    for i,(c,a) in enumerate(assessments):
        rp = c['rel_pos']
        sc2 = max(150, c['miss_km']*200)
        ru  = rp/(np.linalg.norm(rp)+1e-10)
        tp  = pos+ru*sc2
        fig.add_trace(go.Scatter3d(x=[tp[0]],y=[tp[1]],z=[tp[2]],mode='markers',
            marker=dict(color=TC[a.alert],size=12,symbol='x',
                        line=dict(color='white',width=1)),
            name=f"{c['object_name'][:14]} [{a.alert.value}]",
            hovertemplate=(f"{c['object_name']}<br>Miss:{c['miss_km']:.3f}km<br>"
                           f"TCA:{c['tca_hours']:.2f}h<br>Pc:{a.adjusted_pc:.2e}")))
        fig.add_trace(go.Scatter3d(
            x=[tp[0],pos[0]],y=[tp[1],pos[1]],z=[tp[2],pos[2]],
            mode='lines',line=dict(color=TC[a.alert],width=1,dash='dot'),
            showlegend=False,hoverinfo='skip'))

    fig.update_layout(height=560,paper_bgcolor='#05050f',plot_bgcolor='#05050f',
        scene=dict(bgcolor='#05050f',
                   xaxis=dict(visible=False),yaxis=dict(visible=False),
                   zaxis=dict(visible=False),aspectmode='data',
                   camera=dict(eye=dict(x=1.4,y=1.4,z=0.7))),
        legend=dict(bgcolor='rgba(10,10,30,0.9)',font=dict(color='white',size=10),
                    x=0,y=1),margin=dict(l=0,r=0,t=0,b=0))

    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.maneuver_on:
        dv = st.session_state.last_dv
        st.error(f"🔥 **THRUSTER FIRING** | "
                 f"ΔV=[{dv[0]:.2f},{dv[1]:.2f},{dv[2]:.2f}] m/s | "
                 f"Mag={np.linalg.norm(dv):.2f} m/s | "
                 f"{'AUTONOMOUS — no ground contact' if not sat.ground_contact else 'Ground confirmed'}")
    else:
        st.success("🟢 Thrusters NOMINAL — No burn active")

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT — THREAT CARDS + DOWNLOG
# ─────────────────────────────────────────────────────────────────────────────
with right:

    if not assessments:
        st.success("✅ No active threats")
        st.info(f"**Satellite: ISS (NORAD 25544)**\n\n"
                f"Position (ECI): [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] km\n\n"
                f"Velocity (ECI): [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}] km/s\n\n"
                f"Altitude: {alt:.1f} km\n\nSpeed: {spd:.3f} km/s\n\n"
                f"Latitude: {lat:.2f}°  |  Longitude: {lon:.2f}°\n\n"
                f"Use **Inject Threat** in sidebar to trigger a real alert.")
    else:
        for c,a in assessments:
            bc2 = {Alert.GREEN:"#00ff88",Alert.YELLOW:"#ffdd00",
                   Alert.ORANGE:"#ff8800",Alert.RED:"#ff2200"}[a.alert]
            ic  = {Alert.GREEN:"✅",Alert.YELLOW:"⚠️",
                   Alert.ORANGE:"🟠",Alert.RED:"🔴"}[a.alert]
            st.markdown(
                f'<div style="border:1px solid {bc2};border-radius:8px;'
                f'padding:10px;margin-bottom:8px;background:#0a0a1a;">'
                f'<b style="color:{bc2}">{ic} {c["object_name"]}</b>'
                f'<span style="color:#666;font-size:11px"> — {c["object_id"]}</span>'
                f'</div>', unsafe_allow_html=True)

            m1,m2 = st.columns(2)
            m1.metric("Miss Dist", f"{c['miss_km']:.3f} km")
            m2.metric("TCA",       f"{c['tca_hours']:.2f} h")
            m1.metric("Rel Speed", f"{c['rel_speed_kms']:.2f} km/s")
            m2.metric("Pc (adj)",  f"{a.adjusted_pc:.2e}")

            if a.dv_magnitude_ms > 0:
                dv2 = a.dv_vector
                st.info(f"**Burn** ΔV=[{dv2[0]:.2f},{dv2[1]:.2f},{dv2[2]:.2f}] m/s  \n"
                        f"Mag: {a.dv_magnitude_ms:.2f} m/s | Fuel: {a.fuel_cost_pct:.3f}%")

            if a.limitations_hit:
                with st.expander(f"⚡ {len(a.limitations_hit)} limitations"):
                    for lm in a.limitations_hit: st.warning(lm)

            {Alert.GREEN:st.success,Alert.YELLOW:st.info,
             Alert.ORANGE:st.warning,Alert.RED:st.error}[a.alert](
                f"**{a.decision}**")
            st.divider()

    # DOWNLOG
    st.subheader("📋 Live Downlink Log")
    if not st.session_state.downlog:
        st.caption("Press ▶ Run One Cycle to start. Log appears here.")
    else:
        html = '<div class="downlog">'
        for e in st.session_state.downlog[:50]:
            html += (f'<div><span style="color:#444">[{e["ts"]}]</span> '
                     f'<span class="{e["css"]}">{e["msg"]}</span></div>')
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────────────────────────────────────
if auto:
    time.sleep(10)
    st.session_state.cycle += 1
    st.rerun()

st.caption(f"ACAS | ISS NORAD 25544 | Cycle #{st.session_state.cycle} | "
           f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")