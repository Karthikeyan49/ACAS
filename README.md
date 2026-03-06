# ACAS — Autonomous Collision Avoidance System

AI-powered satellite safety: real-time conjunction detection,
LightGBM risk assessment, autonomous manoeuvre planning.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env            # add Space-Track credentials
python model/train.py --tune    # train LightGBM model (~30 min)

python simulator/orbital.py     # Terminal 1 — physics engine
python core/controller.py       # Terminal 2 — 60s ACAS loop
streamlit run dashboard/app.py  # Terminal 3 — web dashboard
```

## REST API

```bash
uvicorn api.endpoint:app --reload --port 8000
# POST http://localhost:8000/api/predict
```

## Docs
- `docs/architecture.md` — system design, module dependency map
- `docs/api.md`          — Python and REST API reference
