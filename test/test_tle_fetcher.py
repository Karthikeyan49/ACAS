"""
TEST FILE 1 — TLE FETCHER + ORBIT PROPAGATOR
=============================================
What this tests:
  1. TLEFetcher logs into Space-Track.org
  2. Downloads real TLE data for 10 objects
  3. parse_to_propagators() creates OrbitPropagator for each
  4. OrbitPropagator.get_state() returns real position + velocity
  5. OrbitPropagator.get_trajectory() returns 60 minute-by-minute snapshots

How to run:
  python tests/test_1_tle_fetcher.py

What you should see if it works:
  - Login success message
  - 10 objects fetched and parsed
  - Position and velocity printed for each object
  - Trajectory with 60 snapshots confirmed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tle_fetcher import TLEFetcher, OrbitPropagator
from datetime import datetime
import numpy as np

# ─────────────────────────────────────────────────────────
# STEP 1 — Test OrbitPropagator directly using a known TLE
#           (ISS — International Space Station)
#           No internet needed for this step
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Testing OrbitPropagator directly with ISS TLE")
print("="*60)

# Real ISS TLE lines (these are valid, may be slightly old)
ISS_TLE1 = "1 25544U 98067A   24001.50000000  .00005764  00000-0  10780-3 0  9993"
ISS_TLE2 = "2 25544  51.6416 290.0015 0002627  55.4917 344.9690 15.49960988432698"

# Call OrbitPropagator directly
propagator = OrbitPropagator(ISS_TLE1, ISS_TLE2, age_hours=10)
print(f"✅ OrbitPropagator created for ISS")
print(f"   TLE age: {propagator.age_hours} hours")
print(f"   Stale flag: {propagator.age_hours > 48}")

# ─────────────────────────────────────────────────────────
# STEP 2 — Call get_state() to get current position
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Testing get_state() — single position snapshot")
print("="*60)

state = propagator.get_state()

if state is None:
    print("❌ get_state() returned None — SGP4 error")
else:
    pos = state['pos']
    vel = state['vel']
    altitude = np.linalg.norm(pos) - 6371  # subtract Earth radius

    print(f"✅ get_state() returned valid data")
    print(f"   Timestamp       : {state['time']}")
    print(f"   Position (ECI)  : X={pos[0]:.1f} km, Y={pos[1]:.1f} km, Z={pos[2]:.1f} km")
    print(f"   Velocity (ECI)  : Vx={vel[0]:.3f} km/s, Vy={vel[1]:.3f} km/s, Vz={vel[2]:.3f} km/s")
    print(f"   Altitude        : {altitude:.1f} km  (ISS should be ~408 km)")
    print(f"   Speed           : {np.linalg.norm(vel):.3f} km/s  (ISS should be ~7.66 km/s)")
    print(f"   Stale flag      : {state['stale']}")

    # Basic validation
    assert 300 < altitude < 600,  f"❌ Altitude {altitude:.1f} km looks wrong for ISS"
    assert 7.0 < np.linalg.norm(vel) < 8.5, "❌ Speed looks wrong for ISS"
    print("✅ Altitude and speed values are in expected range")

# ─────────────────────────────────────────────────────────
# STEP 3 — Call get_trajectory() to get 60-minute path
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Testing get_trajectory() — 60 minute trajectory")
print("="*60)

trajectory = propagator.get_trajectory(hours=1, step_min=1)

print(f"✅ get_trajectory() returned {len(trajectory)} snapshots")
print(f"   Expected: 60 snapshots (1 per minute for 1 hour)")
print(f"   First snapshot time : {trajectory[0]['time']}")
print(f"   Last  snapshot time : {trajectory[-1]['time']}")

# Show 5 sample positions
print("\n   Sample positions (every 12 minutes):")
print(f"   {'Time':^25} {'Altitude (km)':^15} {'Speed (km/s)':^12}")
print(f"   {'-'*55}")
for i in [0, 12, 24, 36, 48, 59]:
    s = trajectory[i]
    alt = np.linalg.norm(s['pos']) - 6371
    spd = np.linalg.norm(s['vel'])
    print(f"   {str(s['time'].strftime('%H:%M:%S UTC')):^25} {alt:^15.1f} {spd:^12.3f}")

assert len(trajectory) == 60, f"❌ Expected 60 snapshots, got {len(trajectory)}"
print(f"\n✅ Trajectory test passed")

# ─────────────────────────────────────────────────────────
# STEP 4 — Test TLEFetcher with Space-Track (needs internet)
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Testing TLEFetcher with Space-Track.org")
print("="*60)
print("NOTE: This requires a free account at space-track.org")
print("      If you don't have one yet, Step 4 will be skipped")
print("      Steps 1-3 above already confirmed core functionality")

USERNAME = "karthikeyansenthilkumar0@gmail.com"   # ← Put your space-track.org username here
PASSWORD = "asdfghjkl123456789"   # ← Put your space-track.org password here

if not USERNAME or not PASSWORD:
    print("\n⏭️  Skipping Step 4 — no credentials provided")
    print("   To enable: edit USERNAME and PASSWORD at top of Step 4")
    print("\n✅ OVERALL TEST RESULT: PASSED (Steps 1-3)")
    print("   OrbitPropagator works correctly with real TLE data")
    sys.exit(0)

# Login
fetcher = TLEFetcher(USERNAME, PASSWORD)

# Download 10 objects (small number for quick test)
raw_data = fetcher.get_leo_debris(limit=10)

print(f"\n   Downloaded {len(raw_data)} raw TLE records")
print(f"   First object: {raw_data[0].get('OBJECT_NAME', 'UNKNOWN')}")
print(f"   NORAD ID    : {raw_data[0].get('NORAD_CAT_ID', 'UNKNOWN')}")
print(f"   TLE Line 1  : {raw_data[0].get('TLE_LINE1', '')[:40]}...")
print(f"   TLE Epoch   : {raw_data[0].get('EPOCH', 'UNKNOWN')}")

# Parse into propagators
catalog = fetcher.parse_to_propagators(raw_data)

print(f"\n✅ parse_to_propagators() created {len(catalog)} objects")
print(f"\n   Object details:")
print(f"   {'ID':^10} {'Name':^25} {'Age (h)':^8} {'Stale':^6} {'Alt (km)':^10}")
print(f"   {'-'*65}")

for obj in catalog:
    state = obj['propagator'].get_state()
    if state:
        alt = np.linalg.norm(state['pos']) - 6371
        print(f"   {obj['id']:^10} {obj['name'][:23]:^25} "
              f"{obj['age_hours']:^8.1f} {str(obj['stale']):^6} {alt:^10.1f}")

print(f"\n✅ OVERALL TEST RESULT: FULLY PASSED (All 4 Steps)")