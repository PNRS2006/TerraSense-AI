"""
CERIP — Tests: Unified Index
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.models.unified_index import UnifiedIndex, compute_activity_safety


@pytest.fixture
def ui():
    return UnifiedIndex()


def test_safe_result(ui):
    r = ui.compute(heat_risk_class=1, aqi_value=30)
    assert r.uosi_category == "Safe"


def test_avoid_result(ui):
    r = ui.compute(heat_risk_class=5, aqi_value=250)
    assert r.uosi_category == "Avoid"


def test_compounding_triggers(ui):
    """Both scores >= 3 should trigger compounding."""
    r = ui.compute(heat_risk_class=4, aqi_value=165)
    assert r.compounding_applied is True
    assert r.compound_bonus_applied == 0.5


def test_no_compounding_when_one_low(ui):
    """Compounding should NOT trigger if one score is low."""
    r = ui.compute(heat_risk_class=1, aqi_value=165)
    assert r.compounding_applied is False


def test_uosi_raw_bounds(ui):
    """UOSI raw should be in range [0, 4.5] (max + bonus)."""
    for hrc in range(1, 6):
        for aqi in [10, 80, 150, 220]:
            r = ui.compute(heat_risk_class=hrc, aqi_value=aqi)
            assert 0 <= r.uosi_raw <= 4.6, f"Out of range: {r.uosi_raw}"


def test_custom_weights_sum_to_one():
    ui_var = UnifiedIndex(w_heat=0.3, w_aqi=0.7)
    r = ui_var.compute(heat_risk_class=3, aqi_value=100)
    assert r.uosi_raw >= 0


def test_weight_mismatch_raises():
    with pytest.raises(AssertionError):
        UnifiedIndex(w_heat=0.4, w_aqi=0.4)


def test_activity_safety_returns_dict(ui):
    r = ui.compute(heat_risk_class=3, aqi_value=100)
    act = compute_activity_safety(r, "Running", "Healthy Adult")
    assert "safety_label" in act
    assert act["safety_label"] in ("Safe", "Caution", "Avoid")


def test_activity_sensitive_profile_stricter(ui):
    """Respiratory patient should have equal or stricter safety than healthy adult."""
    r = ui.compute(heat_risk_class=3, aqi_value=130)
    healthy = compute_activity_safety(r, "Running", "Healthy Adult")
    resp    = compute_activity_safety(r, "Running", "Respiratory Patient")
    RANK = {"Safe": 0, "Caution": 1, "Avoid": 2}
    assert RANK[resp["safety_label"]] >= RANK[healthy["safety_label"]]


def test_disclaimer_present(ui):
    r = ui.compute(heat_risk_class=2, aqi_value=80)
    assert "NOT MEDICAL ADVICE" in r.disclaimer.upper() or "informational" in r.disclaimer.lower()
