# ACAS — API Reference

## REST  `api/endpoint.py`

```
GET  /api/health
POST /api/predict   body: {satellite_tle, object_tle, object_type}
GET  /api/thresholds
```

## Python Modules

```python
# TLE pair → full prediction
from pipeline.model_bridge import predict_collision_risk
result = predict_collision_risk(satellite_tle, object_tle, object_type="DEBRIS")

# CDM feature dict → prediction
from model.inference import SatelliteCollisionPredictor
result = SatelliteCollisionPredictor().predict_single(feature_dict)

# Conjunction dict → raw_pc
from model.lgbm_engine import LGBMInferenceEngine
raw_pc = LGBMInferenceEngine().predict_pc_from_conjunction(conj)

# raw_pc → Assessment
from core.risk_scorer import RiskScorer, SatState
assessment = RiskScorer().assess(conj, raw_pc, sat_state)
```

See each file's docstring for full parameter and return type details.
