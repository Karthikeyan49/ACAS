# Changelog

## [2.0.0] — 2026-03-06

### Changed
- Replaced ConjunctionNet ONNX (16.2% accuracy) with LightGBM
- Restructured two separate projects into one coherent codebase

### Added
- model/lgbm_engine.py, model/lgbm_model.py, model/train.py, model/evaluate.py
- data/data_pipeline.py, pipeline/tle_processor.py, pipeline/model_bridge.py
- api/endpoint.py, dashboard/adapter.py
- config/thresholds.yaml, docs/, tests/, pyproject.toml, CONTRIBUTING.md

### Removed
- models/conjunction_net.py (broken neural net)
- trained_models/conjunction_model.onnx + .onnx.data + .pt
- evaluate_models.py, patch_dashboard.py

## [1.0.0] — 2026-01-15

Initial prototype: ConjunctionNet, RiskScorer, PPO agent, Streamlit dashboard.
