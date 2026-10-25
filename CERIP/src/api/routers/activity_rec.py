"""
CERIP — Activity Recommendation Router
POST /activity-recommendation
"""

from fastapi import APIRouter, HTTPException
from src.api.schemas import ActivityRequest, ActivityResponse

router = APIRouter()


def _get_state():
    from src.api.main import state
    return state


@router.post(
    "/activity-recommendation",
    response_model=ActivityResponse,
    summary="Personalised outdoor activity safety recommendation",
)
async def activity_recommendation(req: ActivityRequest):
    """
    Returns a personalised activity safety score (Safe / Caution / Avoid)
    based on the current AQI, heat risk class, activity type, and user profile.

    The score accounts for:
    - Activity intensity (running = higher ventilation = more pollution inhaled)
    - Physiological sensitivity (respiratory patients have lower thresholds)
    - Compounding heat + air quality risk
    - AQI guidance from CPCB / US EPA adapted thresholds
    """
    state = _get_state()

    if state.unified_index is None:
        from src.models.unified_index import UnifiedIndex
        state.unified_index = UnifiedIndex()

    ui = state.unified_index
    uosi_result = ui.compute(
        heat_risk_class=req.heat_risk_class,
        aqi_value=req.aqi_value,
    )

    from src.models.unified_index import compute_activity_safety
    act_result = compute_activity_safety(uosi_result, req.activity, req.user_profile)

    return ActivityResponse(
        activity=act_result["activity"],
        user_profile=act_result["user_profile"],
        safety_label=act_result["safety_label"],
        activity_score=act_result["activity_score"],
        uosi_category=act_result["uosi_category"],
        recommendation=act_result["recommendation"],
        best_time_hint=act_result["best_time_hint"],
        compounding_risk=act_result["compounding"],
    )
