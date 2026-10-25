"""
CERIP — Unified Outdoor Safety Index (M6)
==========================================
Fuses Module A (Heat Risk) and Module B (AQI Risk) into a single
Unified Outdoor Safety Index (UOSI).

Formula (documented and configurable via config/settings.yaml)
--------------------------------------------------------------
  HeatScore  ∈ {0, 1, 2, 3, 4}   (mapped from risk class 1-5)
  AQIScore   ∈ {0, 1, 2, 3, 4}   (mapped from AQI category)

  UOSI_raw = w_heat × HeatScore + w_aqi × AQIScore

  COMPOUNDING RULE:
    If HeatScore >= compound_threshold AND AQIScore >= compound_threshold:
        UOSI_raw += compound_bonus
    (Days when BOTH heat and poor air quality are elevated are rated
     worse than either alone — captures synergistic health risk)

  UOSI category:
    < 1.5  → Safe    ✅  (green)
    1.5-2.5 → Caution ⚠️  (amber)
    2.5-3.5 → Limit  🚫  (orange)
    >= 3.5  → Avoid  ☠️  (red)

Usage
-----
    from src.models.unified_index import UnifiedIndex
    ui = UnifiedIndex()
    result = ui.compute(heat_risk_class=4, aqi_value=145)
    print(result)
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


# =============================================================
# Configuration (loaded from settings.yaml)
# =============================================================

W_HEAT       = float(CFG["unified_index"]["w_heat"])
W_AQI        = float(CFG["unified_index"]["w_aqi"])
COMPOUND_THR = int(CFG["unified_index"]["compound_threshold"])
COMPOUND_BONUS = float(CFG["unified_index"]["compound_bonus"])
THRESHOLDS   = CFG["unified_index"]["thresholds"]

# Heat risk class → numeric score (1=Very Low → 0, 5=Very High → 4)
HEAT_SCORE_MAP = {int(k): int(v)
                  for k, v in CFG["heat_risk"]["heat_score_map"].items()}

# AQI value → numeric score
AQI_CATEGORIES = CFG["air_quality"]["aqi_categories"]


# =============================================================
# Data classes
# =============================================================

@dataclass
class HeatInput:
    risk_class: int               # 1-5
    lst_celsius: float = 0.0      # continuous LST (°C), informational
    shap_drivers: list = field(default_factory=list)


@dataclass
class AQIInput:
    aqi_value: float
    pm25_ugm3: float = 0.0
    category: str = ""


@dataclass
class UOSIResult:
    # Raw inputs
    heat_risk_class: int
    heat_risk_label: str
    heat_score: float
    aqi_value: float
    aqi_category: str
    aqi_score: float

    # Fusion
    uosi_raw: float
    compounding_applied: bool
    compound_bonus_applied: float

    # Final output
    uosi_category: str
    uosi_emoji: str
    uosi_color: str

    # Advisory
    advisory: str
    disclaimer: str = (
        "⚕️ NOT MEDICAL ADVICE. For informational purposes only. "
        "Consult a qualified health professional for medical decisions."
    )

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================
# Core index computation
# =============================================================

class UnifiedIndex:
    """
    Computes the Unified Outdoor Safety Index.

    Parameters (configurable via settings.yaml)
    -------------------------------------------
    w_heat           : weight for heat risk score (default 0.5)
    w_aqi            : weight for AQI score (default 0.5)
    compound_threshold : both scores must be >= this for bonus (default 3)
    compound_bonus   : added when compounding rule triggers (default 0.5)
    """

    def __init__(
        self,
        w_heat: float = W_HEAT,
        w_aqi: float = W_AQI,
        compound_threshold: int = COMPOUND_THR,
        compound_bonus: float = COMPOUND_BONUS,
    ):
        assert abs(w_heat + w_aqi - 1.0) < 1e-6, "w_heat + w_aqi must equal 1.0"
        self.w_heat            = w_heat
        self.w_aqi             = w_aqi
        self.compound_threshold = compound_threshold
        self.compound_bonus    = compound_bonus

    # ── Score mappers ─────────────────────────────────────────────

    @staticmethod
    def heat_class_to_score(heat_risk_class: int) -> float:
        """Map heat risk class 1-5 to numeric score 0-4."""
        return float(HEAT_SCORE_MAP.get(int(heat_risk_class), 2))

    @staticmethod
    def aqi_to_score(aqi_value: float) -> float:
        """Map AQI value to numeric score 0-4."""
        for cat_key, cat in AQI_CATEGORIES.items():
            if cat["min"] <= aqi_value <= cat["max"]:
                return float(cat["score"])
        return 4.0   # hazardous default

    @staticmethod
    def aqi_to_category(aqi_value: float) -> str:
        """Return AQI category label."""
        for cat_key, cat in AQI_CATEGORIES.items():
            if cat["min"] <= aqi_value <= cat["max"]:
                return cat["label"]
        return "Hazardous"

    # ── Category resolver ─────────────────────────────────────────

    @staticmethod
    def uosi_raw_to_category(uosi_raw: float) -> tuple[str, str, str]:
        """Return (label, emoji, color) for a raw UOSI score."""
        for cat_key, cat in THRESHOLDS.items():
            if cat["min"] <= uosi_raw < cat["max"]:
                return cat["label"], cat["emoji"], cat["color"]
        # Fallback to worst
        last = list(THRESHOLDS.values())[-1]
        return last["label"], last["emoji"], last["color"]

    # ── Advisory text ─────────────────────────────────────────────

    @staticmethod
    def build_advisory(
        heat_label: str, aqi_cat: str, uosi_cat: str, compounding: bool
    ) -> str:
        advisories = {
            "Safe": (
                "Outdoor conditions are favourable. All activities are safe "
                "for healthy adults and most sensitive groups."
            ),
            "Caution": (
                "Conditions are acceptable but monitor for changes. "
                "Sensitive groups (children, elderly, respiratory patients) should "
                "limit prolonged exertion."
            ),
            "Limit": (
                "Elevated risk. Shorten outdoor activities and avoid peak heat hours "
                "(11am–3pm). Sensitive individuals should stay indoors."
            ),
            "Avoid": (
                "Very high risk. Avoid all non-essential outdoor activities. "
                "Stay in cool, air-conditioned spaces. "
                "Emergency workers should take maximum precautions."
            ),
        }
        text = advisories.get(uosi_cat, "No advisory available.")
        if compounding:
            text += (
                " ⚠️ COMPOUNDING ALERT: Both heat and air quality are simultaneously "
                "elevated — health risk is greater than either factor alone."
            )
        return text

    # ── Main compute method ───────────────────────────────────────

    def compute(
        self,
        heat_risk_class: int,
        aqi_value: float,
        lst_celsius: float = 0.0,
        pm25_ugm3: float = 0.0,
    ) -> UOSIResult:
        """
        Compute the Unified Outdoor Safety Index for one observation.

        Parameters
        ----------
        heat_risk_class : int 1-5
        aqi_value       : float (AQI index value)
        lst_celsius     : float (informational, not used in formula)
        pm25_ugm3       : float (informational)

        Returns
        -------
        UOSIResult dataclass
        """
        heat_score = self.heat_class_to_score(heat_risk_class)
        aqi_score  = self.aqi_to_score(aqi_value)
        aqi_cat    = self.aqi_to_category(aqi_value)
        heat_label = CFG["heat_risk"]["risk_labels"].get(heat_risk_class, "Unknown")

        # Core formula
        uosi_raw = self.w_heat * heat_score + self.w_aqi * aqi_score

        # Compounding-risk rule
        compounding = (heat_score >= self.compound_threshold
                       and aqi_score >= self.compound_threshold)
        bonus_applied = 0.0
        if compounding:
            uosi_raw   += self.compound_bonus
            bonus_applied = self.compound_bonus
            log.debug("Compounding risk triggered: heat_score=%.1f aqi_score=%.1f",
                      heat_score, aqi_score)

        uosi_cat, uosi_emoji, uosi_color = self.uosi_raw_to_category(uosi_raw)
        advisory = self.build_advisory(heat_label, aqi_cat, uosi_cat, compounding)

        return UOSIResult(
            heat_risk_class=heat_risk_class,
            heat_risk_label=heat_label,
            heat_score=heat_score,
            aqi_value=float(aqi_value),
            aqi_category=aqi_cat,
            aqi_score=aqi_score,
            uosi_raw=round(uosi_raw, 4),
            compounding_applied=compounding,
            compound_bonus_applied=bonus_applied,
            uosi_category=uosi_cat,
            uosi_emoji=uosi_emoji,
            uosi_color=uosi_color,
            advisory=advisory,
        )

    def compute_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute UOSI for every row in a DataFrame.
        Expects columns: heat_risk_class, aqi_value (minimum).
        """
        results = [
            self.compute(
                heat_risk_class=int(row["heat_risk_class"]),
                aqi_value=float(row["aqi_value"]),
                lst_celsius=float(row.get("lst_celsius", 0)),
                pm25_ugm3=float(row.get("pm25_ugm3", 0)),
            ).to_dict()
            for _, row in df.iterrows()
        ]
        return pd.DataFrame(results)


# =============================================================
# Activity Safety Score
# =============================================================

# Sensitivity weights per activity (how much more exposed than normal)
ACTIVITY_SENSITIVITY = {
    "Running":           1.5,   # high ventilation rate
    "Walking":           1.0,   # baseline
    "Cycling":           1.3,
    "Outdoor Yoga":      1.1,
    "Construction Work": 1.4,   # prolonged exposure + physical exertion
    "Elderly Stroll":    0.8,   # low intensity but high physiological vulnerability
}

# Sensitivity multipliers per user profile
PROFILE_MULTIPLIER = {
    "Healthy Adult":       1.0,
    "Child":               1.3,
    "Elderly":             1.4,
    "Respiratory Patient": 1.5,
    "Cardiac Patient":     1.5,
}

# Activity safety thresholds on UOSI_raw (after activity/profile adjustment)
SAFETY_THRESHOLDS = {
    "Safe":    (0.0, 1.5),
    "Caution": (1.5, 2.5),
    "Avoid":   (2.5, 99.0),
}


def compute_activity_safety(
    uosi_result: UOSIResult,
    activity: str,
    user_profile: str,
    aqi_guidance: dict | None = None,
) -> dict:
    """
    Derive an activity-specific safety score and recommendation.

    Parameters
    ----------
    uosi_result  : output from UnifiedIndex.compute()
    activity     : one of ACTIVITY_SENSITIVITY keys
    user_profile : one of PROFILE_MULTIPLIER keys

    Returns
    -------
    dict with:
        activity_score  : float (adjusted UOSI)
        safety_label    : Safe / Caution / Avoid
        recommendation  : str (personalised advice)
        best_time_hint  : str (hint about when to go out)
    """
    act_w   = ACTIVITY_SENSITIVITY.get(activity, 1.0)
    prof_w  = PROFILE_MULTIPLIER.get(user_profile, 1.0)

    # Adjusted score
    adj_score = uosi_result.uosi_raw * act_w * prof_w

    # Safety label
    if adj_score < 1.5:
        safety_label = "Safe"
    elif adj_score < 2.5:
        safety_label = "Caution"
    else:
        safety_label = "Avoid"

    # AQI-specific guidance from config
    if aqi_guidance is None:
        aqi_guidance = CFG["air_quality"]["aqi_guidance"]

    # Recommendation text
    profile_key = user_profile.lower().replace(" ", "_")
    aqi_thresholds = aqi_guidance.get(profile_key, {"caution": 100, "avoid": 200})

    if uosi_result.aqi_value > aqi_thresholds["avoid"]:
        aqi_advice = f"AQI={uosi_result.aqi_value:.0f} exceeds safe limit for {user_profile}. Avoid outdoors."
    elif uosi_result.aqi_value > aqi_thresholds["caution"]:
        aqi_advice = f"AQI={uosi_result.aqi_value:.0f} is at cautionary level for {user_profile}. Reduce exertion."
    else:
        aqi_advice = f"AQI={uosi_result.aqi_value:.0f} is acceptable for {user_profile}."

    heat_advice = ""
    if uosi_result.heat_risk_class >= 4:
        heat_advice = " High heat risk — prefer early morning (before 8am) or evening (after 6pm)."

    recommendation = f"{aqi_advice}{heat_advice}"

    best_time = "Early morning (6-8am) or after sunset" if uosi_result.heat_risk_class >= 3 \
        else "Any time, but avoid midday"

    return {
        "activity":       activity,
        "user_profile":   user_profile,
        "activity_score": round(adj_score, 4),
        "safety_label":   safety_label,
        "recommendation": recommendation,
        "best_time_hint": best_time,
        "uosi_raw":       uosi_result.uosi_raw,
        "uosi_category":  uosi_result.uosi_category,
        "compounding":    uosi_result.compounding_applied,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ui = UnifiedIndex()

    print("\n=== UNIFIED INDEX — DEMO ===\n")
    test_cases = [
        (1, 35,  "Cool, clean day"),
        (3, 95,  "Moderate heat + moderate AQI"),
        (4, 155, "High heat + unhealthy AQI"),
        (5, 210, "Extreme heat + very unhealthy AQI → compounding"),
    ]
    for hrc, aqi, desc in test_cases:
        r = ui.compute(heat_risk_class=hrc, aqi_value=aqi)
        print(f"  {desc}")
        print(f"    Heat: class={hrc} ({r.heat_risk_label})  "
              f"AQI={aqi} ({r.aqi_category})")
        print(f"    UOSI_raw={r.uosi_raw:.3f}  Compounding={r.compounding_applied}  "
              f"Category: {r.uosi_emoji} {r.uosi_category}")
        print(f"    → {r.advisory[:80]}...")
        print()

    # Activity demo
    r_high = ui.compute(heat_risk_class=4, aqi_value=145)
    for activity in ["Running", "Walking", "Elderly Stroll"]:
        act = compute_activity_safety(r_high, activity, "Elderly")
        print(f"  {activity} (Elderly): {act['safety_label']} | {act['recommendation'][:60]}...")
