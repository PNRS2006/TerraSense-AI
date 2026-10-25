"""
CERIP — Heat Risk Router
GET  /heat-risk          → summary stats for all grid cells
POST /heat-risk          → risk + SHAP for a specific coordinate
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from src.api.schemas import HeatRiskRequest, HeatRiskResponse, SHAPDriver

router = APIRouter()


def _get_state():
    from src.api.main import state
    return state


@router.get("/heat-risk", summary="Heat risk summary for all grid cells")
async def get_heat_risk_summary():
    """
    Returns heat risk class distribution across all Vizag grid cells.
    Use POST /heat-risk for a specific coordinate.
    """
    state = _get_state()
    if state.grid_df is None:
        raise HTTPException(503, "Grid data not loaded. Run synthetic_generator.py first.")

    df = state.grid_df
    dist = df["heat_risk_label"].value_counts().to_dict()
    high_risk_cells = df[df["heat_risk_class"] >= 4][["cell_id", "lat", "lon",
                                                        "lst_celsius", "heat_risk_label"]].head(10).to_dict("records")

    return {
        "total_cells": len(df),
        "risk_distribution": dist,
        "top_10_high_risk_cells": high_risk_cells,
        "disclaimer": "NOT MEDICAL ADVICE — informational only",
        "data_notice": "SYNTHETIC DATA — not for operational use",
    }


@router.post("/heat-risk", response_model=HeatRiskResponse,
             summary="Heat risk for a specific coordinate")
async def get_heat_risk_point(req: HeatRiskRequest):
    """
    Find the nearest grid cell to the provided lat/lon and return
    its heat risk classification + optional SHAP explanation.
    """
    state = _get_state()

    if state.grid_df is None:
        raise HTTPException(503, "Grid data not loaded.")

    # ── Find nearest cell ──────────────────────────────────────
    df = state.grid_df
    dist = (df["lat"] - req.lat) ** 2 + (df["lon"] - req.lon) ** 2
    idx = dist.idxmin()
    row = df.iloc[idx]

    # ── Predict using loaded model ─────────────────────────────
    lst_pred  = float(row.get("lst_celsius", 35.0))
    risk_class = int(row.get("heat_risk_class", 3))
    risk_label = str(row.get("heat_risk_label", "Moderate"))

    shap_drivers = None
    if req.include_shap and state.heat_suite is not None:
        try:
            from src.features.heat_features import build_heat_feature_matrix
            feat_df, X_scaled, _, _, scaler, feat_cols = build_heat_feature_matrix(
                df.iloc[[idx]].assign(lat=req.lat, lon=req.lon),
                fit_scaler=True,
            )
            from src.explainability.shap_explainer import explain_heat_model, get_top_drivers
            shap_result = explain_heat_model(
                state.heat_suite.best_regressor, feat_df, feat_cols, save_plots=False
            )
            top_df = get_top_drivers(shap_result["shap_values"], feat_cols, 0, n=5)
            shap_drivers = [
                SHAPDriver(
                    feature=r["feature"],
                    shap_value=round(float(r["shap_value"]), 4),
                    direction=r["direction"],
                )
                for _, r in top_df.iterrows()
            ]
        except Exception as e:
            shap_drivers = None  # SHAP is best-effort

    return HeatRiskResponse(
        cell_id=int(row.get("cell_id", idx)),
        lat=round(float(row["lat"]), 5),
        lon=round(float(row["lon"]), 5),
        lst_predicted_celsius=round(lst_pred, 2),
        heat_risk_class=risk_class,
        heat_risk_label=risk_label,
        top_shap_drivers=shap_drivers,
    )
