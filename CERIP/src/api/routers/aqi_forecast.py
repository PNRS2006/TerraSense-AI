"""
CERIP — AQI Forecast Router
GET  /aqi-forecast?horizon_hours=24|48|72
"""

import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query, HTTPException
from src.api.schemas import AQIForecastResponse, HourlyAQIPoint

log = logging.getLogger(__name__)
router = APIRouter()

AQI_CATEGORY_MAP = {
    (0, 50):   "Good",
    (51, 100): "Moderate",
    (101, 150): "Unhealthy-Sensitive",
    (151, 200): "Unhealthy",
    (201, 300): "Very Unhealthy",
    (301, 999): "Hazardous",
}


def _aqi_category(aqi: float) -> str:
    for (lo, hi), label in AQI_CATEGORY_MAP.items():
        if lo <= aqi <= hi:
            return label
    return "Hazardous"


def _get_state():
    from src.api.main import state
    return state


@router.get("/aqi-forecast", response_model=AQIForecastResponse,
            summary="AQI + PM2.5 forecast for 24, 48, or 72 hours")
async def get_aqi_forecast(
    horizon_hours: int = Query(24, description="Forecast horizon: 24, 48, or 72")
):
    """
    Returns hourly AQI forecast for the specified horizon using the best
    model selected during training (LSTM, Prophet, or GradientBoosting).

    Hourly points are sampled from a synthetic forecast based on the
    last available AQI data.
    """
    if horizon_hours not in (24, 48, 72):
        raise HTTPException(400, "horizon_hours must be 24, 48, or 72")

    state = _get_state()

    if state.aqi_df is None:
        raise HTTPException(503, "AQI data not loaded.")

    # ── Get most recent AQI as baseline ─────────────────────────
    aqi_df = state.aqi_df.sort_values("timestamp")
    current_aqi  = float(aqi_df["aqi"].iloc[-1])
    current_pm25 = float(aqi_df["pm25_ugm3"].iloc[-1])
    current_cat  = _aqi_category(current_aqi)

    # ── Produce forecast using GB model if available ──────────
    model_used = "synthetic_trend"
    mae_val = rmse_val = None

    try:
        if state.aqi_suite and hasattr(state.aqi_suite, "gb_models"):
            from src.features.aqi_features import build_aqi_feature_matrix, split_targets

            df_feat = build_aqi_feature_matrix(aqi_df.tail(500))
            X_df, X_scaled, targets, scaler, feat_cols = split_targets(df_feat)

            forecast_result = state.aqi_suite.forecast_next_hours(X_scaled[-horizon_hours:], horizon_hours)
            model_used = forecast_result["model_used"]
            raw_preds = np.array(forecast_result["aqi_forecast"])

            # Resample to exactly horizon_hours points
            if len(raw_preds) >= horizon_hours:
                forecast_aqi = raw_preds[:horizon_hours]
            else:
                forecast_aqi = np.interp(
                    np.linspace(0, len(raw_preds)-1, horizon_hours),
                    np.arange(len(raw_preds)), raw_preds
                )

            m = state.aqi_suite.results.get("GradientBoosting", {}).get(horizon_hours, {})
            mae_val  = m.get("MAE")
            rmse_val = m.get("RMSE")
        else:
            raise ValueError("No AQI suite")
    except Exception as e:
        log.warning("Model forecast exception (%s); using trend fallback.", e)
        # Fallback: simple trend from recent data
        recent = aqi_df["aqi"].tail(72).values
        noise  = np.random.default_rng(42).normal(0, 5, horizon_hours)
        base   = np.mean(recent[-24:]) if len(recent) >= 24 else current_aqi
        forecast_aqi = np.clip(base + noise, 10, 400)
        model_used = "recent_mean_fallback"

    forecast_pm25 = forecast_aqi * 0.6 + np.random.default_rng(42).normal(0, 3, len(forecast_aqi))
    forecast_pm25 = np.clip(forecast_pm25, 5, 300)

    hourly_points = [
        HourlyAQIPoint(
            hour_offset=h + 1,
            aqi_forecast=round(float(forecast_aqi[h]), 1),
            pm25_forecast=round(float(forecast_pm25[h]), 1),
            category=_aqi_category(float(forecast_aqi[h])),
        )
        for h in range(len(forecast_aqi))
    ]

    return AQIForecastResponse(
        horizon_hours=horizon_hours,
        model_used=model_used,
        current_aqi=round(current_aqi, 1),
        current_category=current_cat,
        forecast=hourly_points,
        mae_on_validation=mae_val,
        rmse_on_validation=rmse_val,
    )
