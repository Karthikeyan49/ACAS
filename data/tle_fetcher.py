# ─────────────────────────────────────────────────────────────────────────────
# data/tle_fetcher.py
#
# PURPOSE:
#   1. TLEFetcher  — logs into Space-Track.org and downloads real TLE data
#   2. OrbitPropagator — takes a TLE and computes position/velocity at any time
#
# CALLED BY:
#   onboard/acas_controller.py  — at startup to build the catalog
# ─────────────────────────────────────────────────────────────────────────────

import requests
import numpy as np
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday


# ─────────────────────────────────────────────────────────────────────────────
# OrbitPropagator
# Takes two TLE lines and uses the SGP4 algorithm to predict where
# the object will be at any future time.
# ─────────────────────────────────────────────────────────────────────────────
class OrbitPropagator:

    def __init__(self, tle1: str, tle2: str, age_hours: float = 0.0):
        """
        tle1, tle2    : The two lines of the TLE string
        age_hours     : How old this TLE is (hours since epoch)
                        Used to flag stale data as a limitation
        """
        self.sat       = Satrec.twoline2rv(tle1, tle2)
        self.age_hours = age_hours
        self.tle1      = tle1
        self.tle2      = tle2

    def get_state(self, dt: datetime = None) -> dict:
        """
        Compute position and velocity at a specific datetime.

        Returns dict with:
          pos   : np.array([x, y, z]) in km  — Earth-Centred Inertial frame
          vel   : np.array([vx, vy, vz]) in km/s
          time  : the datetime used
          stale : True if TLE is older than 48 hours (tracking uncertainty)

        Returns None if SGP4 encounters an error (rare, happens for
        very old or corrupted TLE data).
        """
        if dt is None:
            dt = datetime.utcnow()

        # Convert Python datetime → Julian Date (what SGP4 needs)
        jd, fr = jday(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute, dt.second
        )

        # Run SGP4 physics engine
        error, r, v = self.sat.sgp4(jd, fr)

        if error != 0:
            return None

        return {
            'pos':   np.array(r),          # km
            'vel':   np.array(v),          # km/s
            'time':  dt,
            'stale': self.age_hours > 48   # LIMITATION: old TLE = uncertainty
        }

    def get_trajectory(self, hours: float = 24, step_min: int = 1) -> list:
        """
        Propagate the full trajectory for `hours` hours,
        one snapshot every `step_min` minutes.

        For 24h at 1 min intervals → returns 1,440 state dicts.
        This list is what ConjunctionFinder compares against your satellite.
        """
        trajectory = []
        now        = datetime.utcnow()
        steps      = int(hours * 60 / step_min)

        for i in range(steps):
            t     = now + timedelta(minutes=i * step_min)
            state = self.get_state(t)
            if state is not None:
                trajectory.append(state)

        return trajectory


# ─────────────────────────────────────────────────────────────────────────────
# TLEFetcher
# Authenticates with Space-Track.org and downloads TLE records.
# Space-Track is the US Space Force's official satellite catalog —
# it tracks all 27,000+ objects in orbit.
# Free account: https://www.space-track.org
# ─────────────────────────────────────────────────────────────────────────────
class TLEFetcher:

    BASE_URL = "https://www.space-track.org"

    def __init__(self, username: str, password: str):
        """
        Creates a persistent HTTP session and logs in.
        The session cookie is reused for all subsequent API calls.
        """
        self.session = requests.Session()
        resp = self.session.post(
            f"{self.BASE_URL}/ajaxauth/login",
            data={'identity': username, 'password': password}
        )
        if resp.status_code == 200:
            print("✅ Space-Track login successful")
        else:
            print(f"❌ Login failed — HTTP {resp.status_code}")

    def get_leo_debris(self, limit: int = 200) -> list:
        """
        Downloads TLE data for objects in Low Earth Orbit (LEO).

        Filter criteria:
          MEAN_MOTION > 11.25 rev/day  → orbital period < 128 min (LEO)
          ECCENTRICITY < 0.25          → nearly circular orbits only

        Returns raw JSON list. Each item is one space object.
        """
        url = (
            f"{self.BASE_URL}/basicspacedata/query/class/gp/"
            f"MEAN_MOTION/>11.25/ECCENTRICITY/<0.25/"
            f"orderby/NORAD_CAT_ID/limit/{limit}/format/json"
        )
        resp = self.session.get(url)
        data = resp.json()
        print(f"✅ Downloaded {len(data)} TLE records from Space-Track")
        return data

    def parse_to_propagators(self, tle_data: list) -> list:
        """
        Converts raw JSON TLE records into usable catalog objects.

        For each TLE record:
          1. Parse the epoch to compute TLE age in hours
          2. Create an OrbitPropagator for that object
          3. Set stale=True if age > 48h  (LIMITATION: old data = uncertainty)

        Returns a list of dicts — each dict is one catalog object
        with a live .propagator that can compute positions.
        """
        catalog = []

        for item in tle_data:
            try:
                # Compute age of this TLE
                epoch_str  = item.get('EPOCH', '')
                epoch      = datetime.strptime(epoch_str[:19], '%Y-%m-%dT%H:%M:%S')
                age_hours  = (datetime.utcnow() - epoch).total_seconds() / 3600

                catalog.append({
                    'id':         item['NORAD_CAT_ID'],
                    'name':       item.get('OBJECT_NAME', 'UNKNOWN'),
                    'type':       item.get('OBJECT_TYPE', 'UNKNOWN'),
                    'line1':      item['TLE_LINE1'],
                    'line2':      item['TLE_LINE2'],
                    'age_hours':  age_hours,
                    'stale':      age_hours > 48,
                    # OrbitPropagator created here — one per object
                    'propagator': OrbitPropagator(
                        item['TLE_LINE1'],
                        item['TLE_LINE2'],
                        age_hours
                    )
                })
            except Exception:
                # Skip malformed TLE records silently
                continue

        print(f"✅ Parsed {len(catalog)} orbit propagators")
        return catalog

    def refresh_catalog(self, catalog: list) -> list:
        """
        Re-downloads TLEs for all objects in an existing catalog.
        Called every 90 minutes (one orbit) to keep data fresh.

        LIMITATION SOLVER: prevents TLE staleness from building up.
        """
        ids   = [obj['id'] for obj in catalog]
        id_str = ','.join(ids[:200])
        url   = (
            f"{self.BASE_URL}/basicspacedata/query/class/gp/"
            f"NORAD_CAT_ID/{id_str}/format/json"
        )
        fresh_data = self.session.get(url).json()
        print(f"✅ Refreshed {len(fresh_data)} TLE records")
        return self.parse_to_propagators(fresh_data)