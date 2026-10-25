"""
CERIP Dashboard — Page 4: Unified Safety Index
================================================
- Live UOSI calculator (slider inputs)
- Visual gauge + compounding risk alert
- Formula documentation
- Multi-day UOSI trend from synthetic data
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Unified Safety Index — CERIP", page_icon="🔗", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.uosi-safe    { background: linear-gradient(135deg, #0d3b2e, #155d3a);
                border-radius: 16px; padding: 28px 32px; color: white; text-align: center; }
.uosi-caution { background: linear-gradient(135deg, #3b2a00, #6d4c00);
                border-radius: 16px; padding: 28px 32px; color: white; text-align: center; }
.uosi-limit   { background: linear-gradient(135deg, #3b1800, #7a3500);
                border-radius: 16px; padding: 28px 32px; color: white; text-align: center; }
.uosi-avoid   { background: linear-gradient(135deg, #3b0a0a, #7a1212);
                border-radius: 16px; padding: 28px 32px; color: white; text-align: center; }
.compound-alert {
    background: #3b2000; border: 2px solid #e67e22; border-radius: 10px;
    padding: 14px 18px; color: #ffd07b; margin-top: 12px; font-weight: 600;
}
.formula-box {
    background: #1a1a2e; border-radius: 10px; padding: 16px 20px;
    color: #a8d8a8; font-family: monospace; font-size: 0.9rem; border: 1px solid #2ecc71;
}
.disclaimer-box {
    background: #2d1b00; border: 1px solid #e67e22; border-radius: 8px;
    padding: 10px 14px; color: #f39c12; font-size: 0.82rem; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔗 Unified Outdoor Safety Index (UOSI)")
st.markdown('<div class="disclaimer-box">⚕️ NOT MEDICAL ADVICE &nbsp;|&nbsp; 🔬 SYNTHETIC DATA</div>',
            unsafe_allow_html=True)

# ── Formula documentation ─────────────────────────────────────
with st.expander("📐 Formula & Weights", expanded=False):
    st.markdown("""
    <div class="formula-box">
    HeatScore  ∈ {0, 1, 2, 3, 4}   (mapped from risk class 1-5)<br>
    AQIScore   ∈ {0, 1, 2, 3, 4}   (mapped from AQI Good→Hazardous)<br><br>
    UOSI_raw = w_heat × HeatScore + w_aqi × AQIScore<br><br>
    COMPOUNDING RULE:<br>
    &nbsp;&nbsp;IF HeatScore ≥ 3 AND AQIScore ≥ 3:<br>
    &nbsp;&nbsp;&nbsp;&nbsp;UOSI_raw += 0.5  ← days when both are elevated<br><br>
    UOSI Category:<br>
    &nbsp;&nbsp;< 1.5  → ✅ Safe<br>
    &nbsp;&nbsp;1.5–2.5 → ⚠️  Caution<br>
    &nbsp;&nbsp;2.5–3.5 → 🚫 Limit<br>
    &nbsp;&nbsp;≥ 3.5  → ☠️  Avoid<br><br>
    Default weights: w_heat=0.50, w_aqi=0.50  (editable in config/settings.yaml)
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Live calculator ───────────────────────────────────────────
st.subheader("🧮 Live UOSI Calculator")
col_form, col_result = st.columns([1, 1])

with col_form:
    heat_class = st.slider("🌡️ Heat Risk Class", 1, 5, 3,
                            help="1=Very Low … 5=Very High")
    aqi_val    = st.slider("💨 AQI Value", 0, 400, 90, step=5)

    import yaml
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    st.markdown("**Custom weights** (must sum to 1.0):")
    w_heat = st.slider("w_heat", 0.0, 1.0,
                        float(cfg["unified_index"]["w_heat"]), step=0.05)
    w_aqi  = round(1.0 - w_heat, 2)
    st.info(f"w_aqi = {w_aqi} (auto-calculated)")

with col_result:
    from src.models.unified_index import UnifiedIndex
    ui = UnifiedIndex(w_heat=w_heat, w_aqi=w_aqi)
    result = ui.compute(heat_risk_class=heat_class, aqi_value=aqi_val)

    css_map = {"Safe": "uosi-safe", "Caution": "uosi-caution",
               "Limit": "uosi-limit", "Avoid": "uosi-avoid"}
    css_cls = css_map.get(result.uosi_category, "uosi-caution")

    RISK_LABELS = {1:"Very Low", 2:"Low", 3:"Moderate", 4:"High", 5:"Very High"}

    st.markdown(f"""
    <div class="{css_cls}">
      <h1 style="font-size:3.5rem; margin:0;">{result.uosi_emoji}</h1>
      <h2 style="margin:4px 0;">{result.uosi_category}</h2>
      <p style="opacity:0.85;">UOSI raw = <strong>{result.uosi_raw:.3f}</strong></p>
      <p style="font-size:0.9rem;">
        Heat: {RISK_LABELS[heat_class]} (score={result.heat_score}) &nbsp;|&nbsp;
        AQI: {result.aqi_category} (score={result.aqi_score})
      </p>
    </div>
    """, unsafe_allow_html=True)

    if result.compounding_applied:
        st.markdown("""
        <div class="compound-alert">
          ⚡ COMPOUNDING RISK TRIGGERED<br>
          Both heat and air quality are simultaneously elevated.<br>
          Health impact is greater than either factor alone.
          +0.5 bonus applied to UOSI.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"**Advisory:** {result.advisory}")

# ── Gauge chart ───────────────────────────────────────────────
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=result.uosi_raw,
    title={"text": "UOSI Score", "font": {"color": "white", "size": 16}},
    gauge={
        "axis": {"range": [0, 4.5], "tickcolor": "white"},
        "bar":  {"color": "#fdcb6e"},
        "steps": [
            {"range": [0, 1.5],   "color": "#2ecc71"},
            {"range": [1.5, 2.5], "color": "#f39c12"},
            {"range": [2.5, 3.5], "color": "#e67e22"},
            {"range": [3.5, 4.5], "color": "#e74c3c"},
        ],
        "threshold": {
            "line": {"color": "white", "width": 3},
            "thickness": 0.8,
            "value": result.uosi_raw,
        },
    },
    number={"font": {"color": "white"}},
))
fig_gauge.update_layout(
    paper_bgcolor="#0f0f1a", font={"color": "white"},
    height=280, margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig_gauge, use_container_width=True)

# ── Multi-day trend ───────────────────────────────────────────
st.markdown("---")
st.subheader("📅 7-Day UOSI Trend")

@st.cache_data
def compute_daily_uosi():
    aqi_p  = ROOT / "data" / "synthetic" / "vizag_aqi_synthetic.csv"
    grid_p = ROOT / "data" / "synthetic" / "vizag_grid_synthetic.csv"
    if not aqi_p.exists() or not grid_p.exists():
        return None

    aqi_df  = pd.read_csv(aqi_p, parse_dates=["timestamp"])
    grid_df = pd.read_csv(grid_p)

    # Daily mean AQI for last 7 days
    daily_aqi = aqi_df.groupby(aqi_df["timestamp"].dt.date)["aqi"].mean().tail(7)
    # Representative heat class (median across grid)
    med_heat  = int(grid_df["heat_risk_class"].median())

    ui_inner = UnifiedIndex()
    records = []
    for date, aqi in daily_aqi.items():
        r = ui_inner.compute(heat_risk_class=med_heat, aqi_value=float(aqi))
        records.append({
            "date":     pd.Timestamp(date),
            "aqi":      aqi,
            "heat_cls": med_heat,
            "uosi_raw": r.uosi_raw,
            "category": r.uosi_category,
            "emoji":    r.uosi_emoji,
        })
    return pd.DataFrame(records)


trend_df = compute_daily_uosi()
if trend_df is not None:
    CAT_COLOR = {"Safe": "#2ecc71", "Caution": "#f39c12",
                 "Limit": "#e67e22", "Avoid": "#e74c3c"}

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=trend_df["date"].dt.strftime("%b %d"),
        y=trend_df["uosi_raw"],
        marker_color=[CAT_COLOR.get(c, "#aaa") for c in trend_df["category"]],
        text=[f"{r['emoji']} {r['category']}" for _, r in trend_df.iterrows()],
        textposition="outside",
    ))
    # Add threshold lines
    for y, label, color in [(1.5, "Safe|Caution", "#f39c12"),
                             (2.5, "Caution|Limit", "#e67e22"),
                             (3.5, "Limit|Avoid", "#e74c3c")]:
        fig_trend.add_hline(y=y, line_dash="dash", line_color=color,
                            annotation_text=label, annotation_position="right")

    fig_trend.update_layout(
        title="Daily UOSI Score — Last 7 Days (grid-median heat, daily-mean AQI)",
        xaxis_title="Date", yaxis_title="UOSI Raw",
        template="plotly_dark", height=360,
        yaxis_range=[0, 5],
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Generate synthetic data first: `python src/data/synthetic_generator.py`")

st.caption("⚕️ NOT MEDICAL ADVICE | 🔬 SYNTHETIC DATA")
