"""
api/endpoint.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    FastAPI REST backend. Exposes the full TLE → LightGBM → prediction
    pipeline over HTTP for integration with external tools.

CALLED FROM
    Terminal:  uvicorn api.endpoint:app --reload --port 8000
    pyproject.toml entry point: "acas-api"

CALLS INTO
    pipeline/model_bridge.py   CollisionRiskPredictor, predict_collision_risk
    fastapi, uvicorn, pydantic

WHAT IT PROVIDES (HTTP endpoints)
    GET  /api/health
        Model load status, version, regressor/classifier details.

    POST /api/predict
        Body:     { satellite_tle, object_tle, object_type }
        Response: full prediction dict from pipeline/model_bridge.py
        HTTP 422  on TLE parse error.

    GET  /api/thresholds
        Alert Pc threshold values from config/thresholds.yaml.

CORS
    allow_origins=["*"] — restrict to dashboard URL in production.

IMPORT CHANGE FROM ORIGINAL (satellite_lgbm/api_endpoint.py)
    from model_bridge import  →  from pipeline.model_bridge import
══════════════════════════════════════════════════════════════════════════════
"""
"""
api_endpoint.py
───────────────
FastAPI backend endpoint that connects your dashboard to the LightGBM model.
This replaces your old backend model call.

Install: pip install fastapi uvicorn pydantic
Run    : uvicorn api_endpoint:app --reload --port 8000

Dashboard calls:
    POST http://localhost:8000/api/predict
    GET  http://localhost:8000/api/health
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict
import logging
import uvicorn

from pipeline.model_bridge import CollisionRiskPredictor, predict_collision_risk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Satellite Collision Avoidance API",
    description="LightGBM-powered collision risk prediction from TLE data",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # restrict to your dashboard URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model ONCE at startup (singleton)
predictor = CollisionRiskPredictor()


# ── Request / Response schemas ────────────────────────────────────────────────

class TLEInput(BaseModel):
    """
    Input from dashboard — TLE pair injected manually by operator.
    """
    satellite_tle: str = Field(
        ...,
        description="3-line TLE of your satellite",
        example="ISS (ZARYA)\n1 25544U 98067A...\n2 25544..."
    )
    object_tle: str = Field(
        ...,
        description="3-line TLE of threat object / debris",
        example="DEBRIS\n1 33442U 93036ABH...\n2 33442..."
    )
    object_type: Optional[str] = Field(
        default="UNKNOWN",
        description="Object type: PAYLOAD | ROCKET BODY | DEBRIS | UNKNOWN"
    )
    satellite_rcs_m2: Optional[float] = Field(
        default=None,
        description="Radar cross-section of satellite (m²)"
    )
    object_rcs_m2: Optional[float] = Field(
        default=None,
        description="Radar cross-section of object (m²)"
    )
    space_weather: Optional[Dict] = Field(
        default=None,
        description="Space weather indices: {F10, F3M, SSN, AP}"
    )
    mission_id: Optional[int] = Field(default=1)
    event_id:   Optional[int] = Field(default=0)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    """Health check — dashboard polls this to confirm model is loaded."""
    return predictor.health_check()


@app.post("/api/predict")
def predict(data: TLEInput):
    """
    ╔══════════════════════════════════════════════════════════╗
    ║  Main prediction endpoint                                ║
    ║                                                          ║
    ║  Dashboard sends TLE pair → returns full risk assessment ║
    ╚══════════════════════════════════════════════════════════╝

    Returns complete risk assessment including:
      - risk score and collision probability
      - alert level (NOMINAL / LOW / ELEVATED / HIGH / CRITICAL)
      - conjunction geometry (miss distance, TCA time, RIC vectors)
      - manoeuvre recommendation (delta-V if needed)
      - satellite and object info
      - uncertainty estimates
    """
    result = predictor.predict(
        satellite_tle_str = data.satellite_tle,
        object_tle_str    = data.object_tle,
        object_type       = data.object_type or "UNKNOWN",
        satellite_rcs_m2  = data.satellite_rcs_m2,
        object_rcs_m2     = data.object_rcs_m2,
        space_weather     = data.space_weather,
        mission_id        = data.mission_id or 1,
        event_id          = data.event_id or 0,
    )

    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@app.get("/api/thresholds")
def get_thresholds():
    """Return operational threshold definitions for dashboard legend."""
    return {
        "CRITICAL":         {"risk_score": -4.0, "probability": "1e-4",
                             "action": "Emergency manoeuvre required"},
        "HIGH":             {"risk_score": -5.0, "probability": "1e-5",
                             "action": "Manoeuvre recommended"},
        "ELEVATED":         {"risk_score": -6.0, "probability": "1e-6",
                             "action": "Enhanced monitoring"},
        "LOW":              {"risk_score": -8.0, "probability": "1e-8",
                             "action": "Log and watch"},
        "NOMINAL":          {"risk_score": -99,  "probability": "<1e-8",
                             "action": "No action required"},
    }


# ── Run server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api_endpoint:app", host="0.0.0.0", port=8000, reload=True)