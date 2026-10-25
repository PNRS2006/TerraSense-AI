"""
CERIP Dashboard — Page 3: Activity Advisor
===========================================
User-profile form → personalised activity recommendation
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Activity Advisor — CERIP", page_icon="🏃", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.result-safe    { background: #0d3b2e; border-left: 5px solid #2ecc71; border-radius: 10px;
                  padding: 20px 24px; color: white; margin: 12px 0; }
.result-caution { background: #3b2a00; border-left: 5px solid #f39c12; border-radius: 10px;
                  padding: 20px 24px; color: white; margin: 12px 0; }
.result-avoid   { background: #3b0a0a; border-left: 5px solid #e74c3c; border-radius: 10px;
                  padding: 20px 24px; color: white; margin: 12px 0; }
.disclaimer-box {
    background: #2d1b00; border: 1px solid #e67e22; border-radius: 8px;
    padding: 10px 14px; color: #f39c12; font-size: 0.82rem; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🏃 Outdoor Activity Advisor")
st.markdown('<div class="disclaimer-box">⚕️ NOT MEDICAL ADVICE — consult a health professional for personal health decisions. &nbsp;|&nbsp; 🔬 SYNTHETIC DATA</div>',
            unsafe_allow_html=True)

ACTIVITIES = ["Running", "Walking", "Cycling", "Outdoor Yoga",
              "Construction Work", "Elderly Stroll"]
PROFILES   = ["Healthy Adult", "Child", "Elderly",
              "Respiratory Patient", "Cardiac Patient"]
RISK_LABELS = {1:"Very Low", 2:"Low", 3:"Moderate", 4:"High", 5:"Very High"}

# ── Form ──────────────────────────────────────────────────────
with st.form("activity_form"):
    st.subheader("Your Outdoor Plan")
    col1, col2 = st.columns(2)

    with col1:
        activity = st.selectbox("🏅 Activity type", ACTIVITIES)
        profile  = st.selectbox("👤 Your health profile", PROFILES)
        aqi_val  = st.slider("💨 Current / expected AQI", 10, 400, 85, step=5)

    with col2:
        heat_class = st.slider("🌡️ Heat risk class", 1, 5, 3, step=1,
                                help="1=Very Low, 5=Very High (from heat risk map)")
        hour_pref = st.slider("⏰ Planned start hour (24h)", 0, 23, 7)
        duration  = st.slider("⏱️ Planned duration (minutes)", 15, 180, 45, step=15)

    submitted = st.form_submit_button("🔍 Get Recommendation", use_container_width=True)

if submitted:
    # ── Compute ───────────────────────────────────────────────
    from src.models.unified_index import UnifiedIndex, compute_activity_safety
    ui = UnifiedIndex()
    uosi = ui.compute(heat_risk_class=heat_class, aqi_value=aqi_val)
    act  = compute_activity_safety(uosi, activity, profile)

    safety = act["safety_label"]
    css_cls = {"Safe": "result-safe", "Caution": "result-caution", "Avoid": "result-avoid"}.get(safety, "result-caution")
    emoji   = {"Safe": "✅", "Caution": "⚠️", "Avoid": "🚫"}.get(safety, "⚠️")

    st.markdown(f"""
    <div class="{css_cls}">
      <h2>{emoji} {safety}</h2>
      <p><strong>Activity:</strong> {activity} &nbsp;|&nbsp;
         <strong>Profile:</strong> {profile}</p>
      <p><strong>Recommendation:</strong> {act['recommendation']}</p>
      <p><strong>Best time:</strong> {act['best_time_hint']}</p>
      {'<p>⚠️ <strong>COMPOUNDING RISK:</strong> Heat AND air quality are simultaneously elevated — risk is greater than either alone.</p>' if act['compounding'] else ''}
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────
    st.markdown("---")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: st.metric("UOSI Category", f"{uosi.uosi_emoji} {uosi.uosi_category}")
    with mc2: st.metric("UOSI Raw Score", f"{uosi.uosi_raw:.3f} / 4.0")
    with mc3: st.metric("Activity Score (adj.)", f"{act['activity_score']:.3f}")
    with mc4: st.metric("Compounding Risk", "YES ⚠️" if act["compounding"] else "No")

    # ── Hourly AQI bar chart (best window) ───────────────────
    st.markdown("---")
    st.subheader("🕐 Estimated AQI through the day")

    @st.cache_data
    def load_today_aqi():
        p = ROOT / "data" / "synthetic" / "vizag_aqi_synthetic.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p, parse_dates=["timestamp"])
        # Use the last 24h as "today"
        return df.tail(24)

    today_aqi = load_today_aqi()
    if today_aqi is not None:
        hours = list(range(24))
        aqi_vals  = today_aqi["aqi"].values
        bar_colors = ["#2ecc71" if v <= 50 else "#f39c12" if v <= 100
                      else "#e67e22" if v <= 150 else "#e74c3c"
                      for v in aqi_vals]

        fig = go.Figure(go.Bar(
            x=hours,
            y=aqi_vals,
            marker_color=bar_colors,
            text=[f"{v:.0f}" for v in aqi_vals],
            textposition="outside",
        ))
        # Highlight planned hour
        fig.add_vrect(
            x0=hour_pref - 0.5, x1=hour_pref + duration / 60 - 0.5,
            fillcolor="rgba(255,255,255,0.15)", line_width=0,
            annotation_text=f"Planned: {hour_pref:02d}:00", annotation_position="top left"
        )
        fig.update_layout(
            title="Hourly AQI — Last 24h (proxy for today)",
            xaxis_title="Hour of Day", yaxis_title="AQI",
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
            template="plotly_dark", height=350,
            yaxis_range=[0, max(aqi_vals) * 1.25],
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── All activities comparison ─────────────────────────────
    st.markdown("---")
    st.subheader(f"All activities — {profile} @ AQI={aqi_val}, Heat class={heat_class}")

    rows = []
    for act_name in ACTIVITIES:
        r = compute_activity_safety(uosi, act_name, profile)
        rows.append({
            "Activity": act_name,
            "Safety": r["safety_label"],
            "Score": f"{r['activity_score']:.3f}",
            "Recommendation": r["recommendation"][:60] + "…",
        })
    cmp_df = pd.DataFrame(rows)

    def color_safety(val):
        colors = {"Safe": "background-color:#0d3b2e; color:#2ecc71",
                  "Caution": "background-color:#3b2a00; color:#f39c12",
                  "Avoid": "background-color:#3b0a0a; color:#e74c3c"}
        return colors.get(val, "")

    styled = cmp_df.style.applymap(color_safety, subset=["Safety"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

st.caption("⚕️ NOT MEDICAL ADVICE | 🔬 SYNTHETIC DATA")
