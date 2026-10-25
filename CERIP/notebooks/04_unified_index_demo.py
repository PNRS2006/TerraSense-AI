"""
CERIP — Notebook 04: Unified Index Demo (M6)
=============================================
Run: python notebooks/04_unified_index_demo.py

Demonstrates the Unified Outdoor Safety Index (UOSI) fusion engine,
compounding-risk rule, and activity safety scoring.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.unified_index import UnifiedIndex, compute_activity_safety

ui = UnifiedIndex()

print("\n" + "=" * 70)
print("CERIP — Unified Outdoor Safety Index Demo")
print("=" * 70)

cases = [
    (1, 30,  "Cool, clean day (early morning)"),
    (2, 75,  "Mild heat + moderate AQI"),
    (3, 105, "Moderate heat + unhealthy-sensitive AQI"),
    (4, 165, "High heat + unhealthy AQI"),
    (5, 215, "Extreme heat + very unhealthy → COMPOUNDING"),
    (4, 220, "High heat + very unhealthy → COMPOUNDING"),
    (5, 155, "Extreme heat + unhealthy → COMPOUNDING"),
]

for hrc, aqi, desc in cases:
    r = ui.compute(heat_risk_class=hrc, aqi_value=aqi)
    print(f"\n  Scenario: {desc}")
    print(f"  Input  : Heat class={hrc} ({r.heat_risk_label}) | AQI={aqi} ({r.aqi_category})")
    print(f"  Scores : HeatScore={r.heat_score} | AQIScore={r.aqi_score}")
    print(f"  UOSI   : raw={r.uosi_raw:.3f}  Compounding={r.compounding_applied}  "
          f"→ {r.uosi_emoji} {r.uosi_category}")
    print(f"  Advisory: {r.advisory[:90]}...")

print("\n" + "=" * 70)
print("Activity Safety Demo — High Heat (class=4) + High AQI (165)")
print("=" * 70)

r_high = ui.compute(heat_risk_class=4, aqi_value=165)
activities = ["Running", "Walking", "Cycling", "Outdoor Yoga",
              "Construction Work", "Elderly Stroll"]
profiles   = ["Healthy Adult", "Elderly", "Respiratory Patient"]

for profile in profiles:
    print(f"\n  Profile: {profile}")
    for activity in activities:
        a = compute_activity_safety(r_high, activity, profile)
        emoji = {"Safe": "✅", "Caution": "⚠️", "Avoid": "🚫"}[a["safety_label"]]
        print(f"    {emoji} {activity:<22} score={a['activity_score']:.3f}  → {a['safety_label']}")

print("\n" + "=" * 70)
print("Weight Sensitivity Demo — same inputs, different w_heat")
print("=" * 70)

for w_h in [0.3, 0.5, 0.7]:
    ui_var = UnifiedIndex(w_heat=w_h, w_aqi=round(1 - w_h, 2))
    r_var  = ui_var.compute(heat_risk_class=4, aqi_value=165)
    print(f"  w_heat={w_h}  w_aqi={1-w_h:.1f}  →  UOSI_raw={r_var.uosi_raw:.3f}  {r_var.uosi_emoji} {r_var.uosi_category}")

print("\n✅  Unified Index Demo Complete.")
