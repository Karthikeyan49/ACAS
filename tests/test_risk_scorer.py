"""
tests/test_risk_scorer.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    Unit tests for core/risk_scorer.py. Tests every alert threshold,
    every fuel scaling step, every operational limitation, and the
    manoeuvre planning geometry. No model loading needed.

RUN WITH
    pytest tests/test_risk_scorer.py -v

CALLS INTO
    core/risk_scorer.py   RiskScorer, SatState, Alert
    numpy

TESTS INCLUDED
    Alert thresholds
        test_green_threshold    Pc = 5e-6  → GREEN
        test_yellow_threshold   Pc = 5e-5  → YELLOW
        test_orange_threshold   Pc = 5e-4  → ORANGE
        test_red_threshold      Pc = 5e-3  → RED

    Fuel scaling
        test_fuel_50pct_standard      no change
        test_fuel_30pct_raised_3x
        test_fuel_15pct_raised_8x
        test_fuel_5pct_raised_50x

    Limitations
        test_tle_staleness_inflation    stale=True, age=72h
        test_battery_inflation          battery=15%
        test_post_path_unsafe           post_path_safe=False
        test_critical_phase             mission_phase=critical
        test_altitude_floor_block       alt margin 10km

    Manoeuvre planning
        test_no_burn_if_safe            miss=10km → dv=0
        test_burn_direction_perp        dv · rel_vel ≈ 0
        test_fuel_cost_positive
        test_tsiolkovsky_reasonable     < 5% fuel for 1 m/s burn

    Decision text
        test_green_decision_text
        test_red_autonomous_text
══════════════════════════════════════════════════════════════════════════════
"""
