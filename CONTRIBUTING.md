# Contributing

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install black isort pytest pytest-cov
```

## Folder rules
| Folder | Responsibility |
|---|---|
| `core/` | Runtime only — no heavy ML imports |
| `data/` | Data acquisition and preprocessing |
| `model/` | LightGBM training, inference, config |
| `pipeline/` | End-to-end TLE → prediction workflows |
| `simulator/` | Physics engine only — standalone |
| `dashboard/` | Streamlit UI only |
| `api/` | FastAPI endpoints only |
| `tests/` | Accuracy evaluation and unit tests |

**Dependency rule**: `core/` never imports from `model/`.
Edit thresholds in `config/thresholds.yaml`, not source code.

## Before committing
```bash
black . && isort .
pytest
```
