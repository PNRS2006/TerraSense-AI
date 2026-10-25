"""
CERIP — Pydantic Schemas (M7)
==============================
Request and response models for all FastAPI endpoints.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


DISCLAIMER = (
    "⚕️ NOT MEDICAL ADVICE. This platform is for informational purposes only. "
    "Consult a qualified health professional for any health-related decisions."
)
SYNTHETIC_NOTE = "⚠️ SYNTHETIC DATA — NOT FOR OPERATIONAL USE."


# =============================================================
# Common
# =============================================================

class APIInfo(BaseModel):
    message: str
    version: str
    disclaimer: str = DISCLAIMER
    data_notice: str = SYNTHETIC_NOTE


# =============================================================
# /heat-risk
# =============================================================

class HeatRiskRequest(BaseModel):
    lat: float = Field(..., ge=17.60, le=17.85, description="Latitude (Vizag bounds)")
    lon: float = Field(..., ge=83.15, le=83.45, description="Longitude (Vizag bounds)")
    include_shap: bool = Field(True, description="Include top-5 SHAP drivers in response")


class SHAPDriver(BaseModel):
    feature: str
    shap_value: float
    direction: str


class HeatRiskResponse(BaseModel):
    cell_id: Optional[int] = None
    lat: float
    lon: float
    lst_predicted_celsius: float
    heat_risk_class: int
    heat_risk_label: str
    top_shap_drivers: Optional[list[SHAPDriver]] = None
    disclaimer: str = DISCLAIMER
    data_notice: str = SYNTHETIC_NOTE


# =============================================================
# /aqi-forecast
# =============================================================

class AQIForecastRequest(BaseModel):
    horizon_hours: int = Field(24, description="Forecast horizon: 24, 48, or 72")

    @field_validator("horizon_hours")
    @classmethod
    def check_horizon(cls, v):
        if v not in (24, 48, 72):
            raise ValueError("horizon_hours must be 24, 48, or 72")
        return v


class HourlyAQIPoint(BaseModel):
    hour_offset: int
    aqi_forecast: float
    pm25_forecast: Optional[float] = None
    category: str


class AQIForecastResponse(BaseModel):
    horizon_hours: int
    model_used: str
    current_aqi: float
    current_category: str
    forecast: list[HourlyAQIPoint]
    mae_on_validation: Optional[float] = None
    rmse_on_validation: Optional[float] = None
    disclaimer: str = DISCLAIMER
    data_notice: str = SYNTHETIC_NOTE


# =============================================================
# /activity-recommendation
# =============================================================

VALID_ACTIVITIES = [
    "Running", "Walking", "Cycling", "Outdoor Yoga",
    "Construction Work", "Elderly Stroll"
]

VALID_PROFILES = [
    "Healthy Adult", "Child", "Elderly",
    "Respiratory Patient", "Cardiac Patient"
]


class ActivityRequest(BaseModel):
    activity: str = Field(..., description=f"One of: {VALID_ACTIVITIES}")
    user_profile: str = Field(..., description=f"One of: {VALID_PROFILES}")
    aqi_value: float = Field(..., ge=0, description="Current or forecast AQI value")
    heat_risk_class: int = Field(..., ge=1, le=5, description="Heat risk class 1-5")

    @field_validator("activity")
    @classmethod
    def check_activity(cls, v):
        if v not in VALID_ACTIVITIES:
            raise ValueError(f"activity must be one of {VALID_ACTIVITIES}")
        return v

    @field_validator("user_profile")
    @classmethod
    def check_profile(cls, v):
        if v not in VALID_PROFILES:
            raise ValueError(f"user_profile must be one of {VALID_PROFILES}")
        return v


class ActivityResponse(BaseModel):
    activity: str
    user_profile: str
    safety_label: str
    activity_score: float
    uosi_category: str
    recommendation: str
    best_time_hint: str
    compounding_risk: bool
    disclaimer: str = DISCLAIMER
    data_notice: str = SYNTHETIC_NOTE


# =============================================================
# /unified-index
# =============================================================

class UnifiedIndexRequest(BaseModel):
    heat_risk_class: int = Field(..., ge=1, le=5)
    aqi_value: float = Field(..., ge=0)
    lst_celsius: Optional[float] = None
    pm25_ugm3: Optional[float] = None


class UnifiedIndexResponse(BaseModel):
    heat_risk_class: int
    heat_risk_label: str
    heat_score: float
    aqi_value: float
    aqi_category: str
    aqi_score: float
    uosi_raw: float
    compounding_applied: bool
    compound_bonus_applied: float
    uosi_category: str
    uosi_emoji: str
    uosi_color: str
    advisory: str
    formula_note: str = (
        "UOSI_raw = 0.5 × HeatScore + 0.5 × AQIScore "
        "+ 0.5 (compounding bonus if both scores ≥ 3)"
    )
    disclaimer: str = DISCLAIMER
    data_notice: str = SYNTHETIC_NOTE
