"""
CERIP — Unified Index Router
POST /unified-index
"""

from fastapi import APIRouter
from src.api.schemas import UnifiedIndexRequest, UnifiedIndexResponse

router = APIRouter()


def _get_state():
    from src.api.main import state
    return state


@router.post(
    "/unified-index",
    response_model=UnifiedIndexResponse,
    summary="Compute the Unified Outdoor Safety Index (UOSI)",
)
async def compute_unified_index(req: UnifiedIndexRequest):
    """
    Fuses Heat Risk (Module A) and Air Quality Risk (Module B) into a single
    Unified Outdoor Safety Index: **Safe / Caution / Limit / Avoid**.

    Formula:
    ```
    UOSI_raw = 0.5 × HeatScore + 0.5 × AQIScore
    + 0.5 compound_bonus  (if both scores ≥ 3)
    ```

    Weights are configurable in `config/settings.yaml`.
    """
    state = _get_state()

    if state.unified_index is None:
        from src.models.unified_index import UnifiedIndex
        state.unified_index = UnifiedIndex()

    result = state.unified_index.compute(
        heat_risk_class=req.heat_risk_class,
        aqi_value=req.aqi_value,
        lst_celsius=req.lst_celsius or 0.0,
        pm25_ugm3=req.pm25_ugm3 or 0.0,
    )
    return UnifiedIndexResponse(**result.to_dict())
