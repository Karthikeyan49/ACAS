"""
data/conjunction_finder.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    Scans your satellite's 24-hour trajectory against every catalog object.
    Finds the Time of Closest Approach (TCA) and miss distance for each pair.
    Returns all events with miss_km < 5 km, sorted soonest first.

CALLED FROM
    core/controller.py   ConjunctionFinder().find_all(my_traj, catalog)

CALLS INTO
    numpy
    Nothing from this project.

WHAT IT PROVIDES
    ConjunctionFinder
        SCREEN_KM = 5.0
        find_all(your_traj, catalog) → list[dict]
            your_traj   1,440 state dicts from OrbitPropagator.get_trajectory()
            catalog     list from TLEFetcher.parse_to_propagators()
            returns     conjunction dicts sorted by tca_hours (soonest first)

CONJUNCTION DICT FORMAT (one per event — consumed by model/lgbm_engine.py)
    object_id      str         NORAD catalog ID
    object_name    str         e.g. "COSMOS 2251 DEB"
    object_type    str         "DEBRIS" | "PAYLOAD" | "ROCKET_BODY"
    miss_km        float       minimum separation at TCA (km)
    tca_hours      float       hours until Time of Closest Approach
    tca_time       datetime    UTC time of TCA
    rel_pos        np.array    relative position at TCA (km)
    rel_vel        np.array    relative velocity at TCA (km/s)
    rel_speed_kms  float       scalar closing speed (km/s)
    tle_stale      bool        True if TLE age > 48h
    tle_age_hours  float       hours since TLE epoch

ALGORITHM
    Brute-force minimum distance over 1,440 time steps.
    Same method as NASA CARA initial screening pass.

MIGRATED FROM
    satellite-acas/data/conjunction_finder.py  — no import changes
══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from datetime import datetime


class ConjunctionFinder:

    # Industry standard screening distance.
    # Anything coming within 5km is investigated further.
    SCREEN_KM = 5.0

    def find_all(self, your_traj: list, catalog: list) -> list:
        """
        Main method. Scans all catalog objects for close approaches.

        your_traj : list of state dicts from your satellite's propagator
                    (1,440 snapshots for 24 hours at 1/min)
        catalog   : list of catalog objects from TLEFetcher.parse_to_propagators()

        Returns:
          List of conjunction dicts, sorted by tca_hours (soonest first).
          Each dict has everything the AI model needs.
        """
        conjunctions = []
        now          = datetime.utcnow()

        for obj in catalog:
            try:
                # Get this object's 24-hour trajectory
                obj_traj = obj['propagator'].get_trajectory(
                    hours=24, step_min=1
                )

                # Find the moment of closest approach
                result = self._closest_approach(your_traj, obj_traj)

                # Only keep objects that pass within screening distance
                if result and result['miss_km'] < self.SCREEN_KM:
                    tca_hours = (
                        result['tca_time'] - now
                    ).total_seconds() / 3600

                    conjunctions.append({
                        'object_id':     obj['id'],
                        'object_name':   obj['name'],
                        'object_type':   obj['type'],
                        'miss_km':       result['miss_km'],
                        'tca_hours':     tca_hours,
                        'tca_time':      result['tca_time'],
                        'rel_pos':       result['rel_pos'],
                        'rel_vel':       result['rel_vel'],
                        'rel_speed_kms': np.linalg.norm(result['rel_vel']),
                        'tle_stale':     obj['stale'],
                        'tle_age_hours': obj['age_hours']
                    })

            except Exception:
                continue

        # Sort by urgency — soonest TCA first
        return sorted(conjunctions, key=lambda x: x['tca_hours'])

    def _closest_approach(self, traj1: list, traj2: list) -> dict:
        """
        Compares two trajectories snapshot by snapshot.
        At each minute, computes the distance between the two objects.
        Returns the snapshot where the distance was smallest.

        Algorithm: Brute Force Minimum Distance
          For each of 1,440 time steps:
            dist = |your_pos[i] - their_pos[i]|
          Return the time step with the minimum dist.

        This is the same method used by NASA's CARA (Conjunction
        Assessment Risk Analysis) for initial screening.
        """
        min_dist = float('inf')
        best     = None
        n        = min(len(traj1), len(traj2))

        for i in range(n):
            s1      = traj1[i]
            s2      = traj2[i]
            rel_pos = s1['pos'] - s2['pos']
            dist    = np.linalg.norm(rel_pos)

            if dist < min_dist:
                min_dist = dist
                best = {
                    'miss_km':  dist,
                    'tca_time': s1['time'],
                    'rel_pos':  rel_pos,
                    'rel_vel':  s1['vel'] - s2['vel']
                }

        return best