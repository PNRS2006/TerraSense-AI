"""
CERIP — FastAPI Main Application (M7)
=======================================
Entry point: uvicorn src.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Project root ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


# =============================================================
# Application state (loaded once at startup)
# =============================================================

class AppState:
    heat_suite    = None
    aqi_suite     = None
    grid_df       = None
    aqi_df        = None
    heat_shap     = None
    aqi_shap      = None
    unified_index = None
    feature_cols_heat = []
    feature_cols_aqi  = []
    X_heat_df     = None
    X_aqi_df      = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data once at startup."""
    log.info("=== CERIP API startup — loading models … ===")
    try:
        from src.models.heat_model import HeatRiskModelSuite
        from src.models.aqi_forecast import AQIForecastSuite
        from src.models.unified_index import UnifiedIndex
        from src.data.synthetic_generator import generate_grid_features, generate_aqi_timeseries
        from src.features.heat_features import build_heat_feature_matrix
        from src.features.aqi_features import build_aqi_feature_matrix, split_targets

        models_dir = ROOT / "models_saved"

        # ── Load or train heat models ─────────────────────────────
        reg_path = models_dir / "heat_regressor_best.joblib"
        if reg_path.exists():
            log.info("Loading pre-trained heat models …")
            state.heat_suite = HeatRiskModelSuite.load(models_dir)
        else:
            log.info("No saved heat models found — training on synthetic data …")
            grid_df = generate_grid_features()
            X_df, X_scaled, y_reg, y_clf, scaler, feat_cols = \
                build_heat_feature_matrix(grid_df)
            state.heat_suite = HeatRiskModelSuite()
            state.heat_suite.train(X_df, X_scaled, y_reg, y_clf, feat_cols, scaler)
            state.heat_suite.save(models_dir)

        # ── Load or train AQI models ──────────────────────────────
        gb_path = models_dir / "aqi_gb_24h.joblib"
        if gb_path.exists():
            log.info("Loading pre-trained AQI models …")
            state.aqi_suite = AQIForecastSuite.load(models_dir)
        else:
            log.info("No saved AQI models found — training on synthetic data …")
            aqi_df = generate_aqi_timeseries()
            df_feat = build_aqi_feature_matrix(aqi_df)
            X_df_aqi, X_scaled_aqi, targets, scaler_aqi, feat_cols_aqi = split_targets(df_feat)
            state.aqi_suite = AQIForecastSuite()
            state.aqi_suite.train(df_feat, X_df_aqi, X_scaled_aqi, targets,
                                   scaler_aqi, feat_cols_aqi, aqi_df)
            state.aqi_suite.save(models_dir)

        # ── Load synthetic data for demo endpoints ────────────────
        grid_csv = ROOT / "data" / "synthetic" / "vizag_grid_synthetic.csv"
        aqi_csv  = ROOT / "data" / "synthetic" / "vizag_aqi_synthetic.csv"

        if grid_csv.exists():
            state.grid_df = pd.read_csv(grid_csv)
        if aqi_csv.exists():
            state.aqi_df = pd.read_csv(aqi_csv, parse_dates=["timestamp"])

        # ── Unified index ─────────────────────────────────────────
        state.unified_index = UnifiedIndex()

        log.info("=== CERIP API ready ===")
    except Exception as e:
        log.error("Startup error: %s", e)
        log.warning("API will run in limited mode (no pre-loaded models).")

    yield  # app runs here

    log.info("CERIP API shutting down.")


# =============================================================
# FastAPI app
# =============================================================

app = FastAPI(
    title="CERIP — Combined Environmental Risk Intelligence Platform",
    description=(
        "AI/ML-powered Urban Heat Risk + Air Quality Forecast API for Visakhapatnam.\n\n"
        "⚠️ **NOT MEDICAL ADVICE** — informational use only.\n"
        "🔬 **SYNTHETIC DATA** — prototype demonstration.\n\n"
        "Endpoints:\n"
        "- `GET /heat-risk` — all-grid heat risk scores\n"
        "- `POST /heat-risk` — single-coordinate heat risk + SHAP\n"
        "- `GET /aqi-forecast` — 72-hour AQI forecast\n"
        "- `POST /activity-recommendation` — personalised activity advice\n"
        "- `POST /unified-index` — Unified Outdoor Safety Index\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include routers ───────────────────────────────────────────
from src.api.routers import heat_risk, aqi_forecast, activity_rec, unified_index

app.include_router(heat_risk.router,     prefix="",  tags=["Heat Risk"])
app.include_router(aqi_forecast.router,  prefix="",  tags=["AQI Forecast"])
app.include_router(activity_rec.router,  prefix="",  tags=["Activity Recommendation"])
app.include_router(unified_index.router, prefix="",  tags=["Unified Index"])


@app.get("/", response_model=dict, tags=["Root"])
async def root():
    return {
        "message": "CERIP API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "disclaimer": "NOT MEDICAL ADVICE — informational only",
        "data_notice": "SYNTHETIC DATA — not for operational use",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "heat_model_loaded": state.heat_suite is not None,
        "aqi_model_loaded":  state.aqi_suite  is not None,
        "grid_rows":         len(state.grid_df) if state.grid_df is not None else 0,
        "aqi_rows":          len(state.aqi_df)  if state.aqi_df  is not None else 0,
    }
