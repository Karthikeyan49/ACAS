# ACAS — Architecture

## Data Flow

```
Space-Track.org TLE catalog
        ↓
data/tle_fetcher.py          fetch, parse, SGP4 propagate
data/conjunction_finder.py   TCA scan, produce conjunction dicts
        ↓ conjunction dict
model/lgbm_engine.py         103 CDM features → LightGBM → raw_pc
        ↓ raw_pc
core/risk_scorer.py          6 operational limits → adjusted_pc + Alert
        ↓ Assessment
core/controller.py           act: GREEN|YELLOW|ORANGE|RED
core/maneuver_planner.py     ΔV vector via PPO or geometric fallback
```

## Module Dependency Map

```
simulator/orbital.py         → (nothing)
data/tle_fetcher.py          → (nothing)
data/conjunction_finder.py   → (nothing)
data/data_pipeline.py        → model/config.py
model/config.py              → (nothing)
model/lgbm_model.py          → (nothing)
model/lgbm_engine.py         → model/config, data/data_pipeline, model/lgbm_model
model/inference.py           → model/config, data/data_pipeline, model/lgbm_model
model/evaluate.py            → model/config
model/train.py               → all model/ + data/data_pipeline
core/risk_scorer.py          → (nothing)
core/maneuver_planner.py     → (nothing)
core/controller.py           → data/, core/risk_scorer, model/lgbm_engine
pipeline/tle_processor.py    → (nothing)
pipeline/model_bridge.py     → pipeline/tle_processor, model/, data/data_pipeline
dashboard/adapter.py         → model/lgbm_engine, core/risk_scorer
dashboard/app.py             → model/lgbm_engine, core/risk_scorer
api/endpoint.py              → pipeline/model_bridge
```

## Import Changes When Migrating Source Files

See each file's docstring — the last section always lists the exact
import lines that need updating when copying from the original projects.
