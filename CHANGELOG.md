# Changelog

## [2.2.0] — 2026-07-19

### Added
- Dynamic per-encounter hard-body radius: the Space-Track RCS_SIZE class
  (SMALL/MEDIUM/LARGE) now flows tle_fetcher → conjunction dict → analytic Pc,
  giving R = own_radius + object_radius per threat (config: hard_body.*).
  Unknown size falls back to the conservative fixed 20 m — never smaller.
- dashboard/mission_console.html — self-contained reviewer-facing mission
  console: animated orbit view, conjunction board, FSM/veto-window display,
  live command-bus (BurnCommand/BurnAck) panel and decision log, replaying a
  scripted end-to-end conjunction using the live pipeline field schema.

## [2.1.0] — 2026-07-16

### Added
- core/pc_analytic.py — Foster 2D analytic collision probability + a
  conservative arbiter that takes max(model Pc, analytic Pc)
- model/calibration.py — isotonic Pc calibration with reliability diagrams
  (graceful identity fallback when no calibrator artifact is present)
- core/config_loader.py — shared loader making config/thresholds.yaml the
  real source of truth for alert bands, fuel scaling, and the decision gate
- core/decision_fsm.py — autonomy governance state machine with a
  ground-veto window and a max-autonomous-ΔV budget
- core/schemas.py — typed dataclass contracts (ConjunctionEvent,
  BurnCommand, BurnAck) at module boundaries
- Closed hardware-in-the-loop: controller writes burn_command.json, the
  simulator applies the ΔV to its live orbit and returns burn_ack.json

### Changed
- core/risk_scorer.py reads thresholds from config/thresholds.yaml
- model/lgbm_engine.py Pc now passes through the analytic arbiter + calibrator
- dashboard 3D scene: labelled satellite + orbit trail, alert-coloured debris
  markers with pulsing halos, and a high-clarity thruster-fire effect
- simulator/orbital.py verify path replaces the previous np.random burn stub

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
