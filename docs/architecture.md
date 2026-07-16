# ACAS — Architecture

## Data Flow

```
Space-Track.org TLE catalog
        ↓
data/tle_fetcher.py          fetch, parse, SGP4 propagate
data/conjunction_finder.py   TCA scan, produce conjunction dicts
        ↓ conjunction dict
model/lgbm_engine.py         103 CDM features → LightGBM → model_pc
   ├─ core/pc_analytic.py     Foster 2D analytic Pc (physics cross-check)
   ├─ pc_arbiter              conservative max(model_pc, analytic_pc)
   └─ model/calibration.py    isotonic calibration (identity if no artifact)
        ↓ raw_pc
core/risk_scorer.py          6 operational limits → adjusted_pc + Alert
                             (thresholds from config/thresholds.yaml)
        ↓ Assessment
core/decision_fsm.py         governance gate:
                             NOMINAL→ALERT→PLAN→AWAIT_GROUND_VETO
                                    →EXECUTE→VERIFY→RECOVER
        ↓ decision
core/controller.py           act: GREEN|YELLOW|ORANGE|RED
core/maneuver_planner.py     ΔV vector via PPO or geometric fallback
        ↓ BurnCommand (core/schemas.py)
data_files/burn_command.json ── command bus ──▶ simulator/orbital.py
                                                 applies ΔV to live orbit
data_files/burn_ack.json    ◀── ack ── verify_burn() confirms achieved ΔV
```

The Pc arbiter and the closed command-bus loop are the two safety-critical
additions: no single learned model can veto physics (analytic Pc floors the
estimate), and no burn is trusted until the simulator returns a matching ack.

## Module Dependency Map

```
simulator/orbital.py         → (nothing — reads burn_command.json off disk)
data/tle_fetcher.py          → (nothing)
data/conjunction_finder.py   → (nothing)
data/data_pipeline.py        → model/config.py
model/config.py              → (nothing)
model/lgbm_model.py          → (nothing)
model/lgbm_engine.py         → model/config, data/data_pipeline, model/lgbm_model,
                               core/pc_analytic, model/calibration
model/inference.py           → model/config, data/data_pipeline, model/lgbm_model
model/calibration.py         → (sklearn — ground software only)
model/evaluate.py            → model/config
model/train.py               → all model/ + data/data_pipeline
core/config_loader.py        → (stdlib only; PyYAML if present)
core/pc_analytic.py          → core/config_loader
core/schemas.py              → (stdlib only — dataclasses, no pydantic)
core/risk_scorer.py          → core/config_loader
core/decision_fsm.py         → core/config_loader
core/maneuver_planner.py     → (nothing)
core/controller.py           → data/, core/risk_scorer, core/decision_fsm,
                               core/schemas, model/lgbm_engine
pipeline/tle_processor.py    → (nothing)
pipeline/model_bridge.py     → pipeline/tle_processor, model/, data/data_pipeline
dashboard/adapter.py         → model/lgbm_engine, core/risk_scorer
dashboard/app.py             → model/lgbm_engine, core/risk_scorer
api/endpoint.py              → pipeline/model_bridge
```

**Flight-software constraint:** everything under `core/` and `simulator/`
imports only numpy + stdlib (and `core/config_loader`), never sklearn/pandas
/lightgbm — so the onboard safety path stays dependency-light. `core/schemas.py`
uses plain dataclasses (not pydantic) for the same reason.

## Import Changes When Migrating Source Files

See each file's docstring — the last section always lists the exact
import lines that need updating when copying from the original projects.
