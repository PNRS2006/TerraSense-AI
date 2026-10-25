"""
CERIP — Streamlit Dashboard
============================
Run: streamlit run dashboard/app.py

4 pages:
  1. Heat Risk Map      — 500m grid choropleth + SHAP "why" panel
  2. AQI Forecast       — 72-hour AQI + PM2.5 chart
  3. Activity Advisor   — user-profile form + recommendation
  4. Unified Safety     — combined UOSI view + compounding risk

This is the main entry / home page.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CERIP — Outdoor Safety Platform",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    border-radius: 16px;
    padding: 40px 48px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-card h1 { font-size: 2.4rem; font-weight: 700; margin-bottom: 8px; }
.hero-card p  { font-size: 1.05rem; opacity: 0.85; }

.metric-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 20px 24px;
    border-left: 4px solid;
    color: white;
}
.safe-card    { border-color: #2ecc71; }
.caution-card { border-color: #f39c12; }
.limit-card   { border-color: #e67e22; }
.avoid-card   { border-color: #e74c3c; }

.disclaimer-box {
    background: #2d1b00;
    border: 1px solid #e67e22;
    border-radius: 8px;
    padding: 12px 16px;
    color: #f39c12;
    font-size: 0.85rem;
    margin-top: 12px;
}
.synthetic-box {
    background: #1a0030;
    border: 1px solid #9b59b6;
    border-radius: 8px;
    padding: 12px 16px;
    color: #d2a8ff;
    font-size: 0.85rem;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Hero section ─────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
  <h1>🌡️💨 CERIP</h1>
  <p><strong>Combined Environmental Risk Intelligence Platform</strong></p>
  <p>Pilot City: <strong>Visakhapatnam (Vizag)</strong>, Andhra Pradesh, India &nbsp;|&nbsp;
     Powered by Random Forest · XGBoost · LightGBM · LSTM · Prophet · SHAP</p>
</div>
""", unsafe_allow_html=True)

# ── Notices ───────────────────────────────────────────────────
st.markdown("""
<div class="synthetic-box">
  🔬 <strong>SYNTHETIC DATA NOTICE:</strong> All data on this platform is synthetically generated
  with realistic statistical properties for Vizag. It is <strong>NOT real observational data</strong>
  and must not be used for operational decisions.
</div>
<div class="disclaimer-box">
  ⚕️ <strong>NOT MEDICAL ADVICE:</strong> This platform is for informational and research purposes only.
  Always consult a qualified health professional for health-related decisions.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Module overview cards ─────────────────────────────────────
st.subheader("Platform Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 🌡️ Module A — Urban Heat Risk
    - 500 m × 500 m geospatial grid
    - Land Surface Temperature prediction (regression)
    - Heat Risk Classification: Very Low → Very High
    - SHAP "Why is this cell risky?" panel
    """)

with col2:
    st.markdown("""
    #### 💨 Module B — AQI Forecast
    - 24h / 48h / 72h AQI + PM2.5 forecasts
    - Models: LSTM · Prophet · Gradient Boosting
    - Best model per horizon auto-selected
    - Activity-specific safety scoring
    """)

with col3:
    st.markdown("""
    #### 🔗 Unified Safety Index (UOSI)
    - Fuses Heat + AQI risks: **Safe / Caution / Limit / Avoid**
    - Compounding-risk bonus when both elevated
    - Configurable weights (settings.yaml)
    - Personalised activity recommendations
    """)

st.markdown("---")

# ── UOSI legend ───────────────────────────────────────────────
st.subheader("Unified Outdoor Safety Index — Legend")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card safe-card"><h3>✅ Safe</h3><p>UOSI &lt; 1.5<br>All activities OK</p></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card caution-card"><h3>⚠️ Caution</h3><p>1.5 – 2.5<br>Sensitive groups take care</p></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card limit-card"><h3>🚫 Limit</h3><p>2.5 – 3.5<br>Shorten outdoor time</p></div>',
                unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card avoid-card"><h3>☠️ Avoid</h3><p>≥ 3.5<br>Stay indoors</p></div>',
                unsafe_allow_html=True)

st.markdown("---")

# ── Navigate ──────────────────────────────────────────────────
st.info("👈 Use the **sidebar** to navigate to the individual analysis pages.")

st.caption("CERIP v1.0 — Visakhapatnam, India | Built with Python · scikit-learn · XGBoost · TensorFlow · Prophet · SHAP · Streamlit")
