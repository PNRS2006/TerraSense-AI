"""
CERIP Dashboard — Page 2: AQI Forecast
========================================
- 72-hour AQI + PM2.5 dual-axis chart
- Model comparison table (MAE / RMSE per horizon)
- Daily mean trend bars
- AQI category colour bands
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="AQI Forecast — CERIP", page_icon="💨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0f0f1a; }
.disclaimer-box {
    background: #2d1b00; border: 1px solid #e67e22; border-radius: 8px;
    padding: 10px 14px; color: #f39c12; font-size: 0.82rem; margin-bottom: 12px;
}
.model-table { background: #1e1e2e; border-radius: 10px; padding: 16px; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 💨 AQI Forecast — Visakhapatnam")
st.markdown('<div class="disclaimer-box">⚕️ NOT MEDICAL ADVICE &nbsp;|&nbsp; 🔬 SYNTHETIC DATA — Not for operational use</div>',
            unsafe_allow_html=True)

# ── Load AQI data ─────────────────────────────────────────────
@st.cache_data
def load_aqi():
    p = ROOT / "data" / "synthetic" / "vizag_aqi_synthetic.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["timestamp"])


aqi_df = load_aqi()

if aqi_df is None:
    st.error("AQI data not found. Run: `python src/data/synthetic_generator.py`")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("💨 AQI Forecast Controls")
    horizon = st.radio("Forecast horizon", [24, 48, 72], index=0,
                       format_func=lambda x: f"{x}h ({x//24}d)")
    show_pm25 = st.checkbox("Show PM2.5 overlay", value=True)
    show_history_days = st.slider("Historical context (days)", 1, 30, 7)

# ── Generate forecast using recent trend ──────────────────────
@st.cache_resource
def get_forecast_suite():
    try:
        from src.models.aqi_forecast import AQIForecastSuite
        from src.features.aqi_features import build_aqi_feature_matrix, split_targets

        models_dir = ROOT / "models_saved"
        aqi_data = load_aqi()

        if (models_dir / "aqi_gb_24h.joblib").exists():
            suite = AQIForecastSuite.load(models_dir)
        else:
            df_feat = build_aqi_feature_matrix(aqi_data)
            X_df, X_sc, targets, scaler, fcols = split_targets(df_feat)
            suite = AQIForecastSuite()
            suite.train(df_feat, X_df, X_sc, targets, scaler, fcols, aqi_data)
            suite.save(models_dir)
        return suite
    except Exception as e:
        return None


def generate_forecast(horizon_h: int, aqi_data: pd.DataFrame) -> pd.DataFrame:
    """Generate AQI forecast using GB model or trend fallback."""
    suite = get_forecast_suite()
    rng   = np.random.default_rng(42)

    recent_aqi  = aqi_data["aqi"].values[-72:]
    recent_pm25 = aqi_data["pm25_ugm3"].values[-72:]

    if suite and suite.gb_models:
        try:
            from src.features.aqi_features import build_aqi_feature_matrix, split_targets
            df_feat = build_aqi_feature_matrix(aqi_data.tail(500))
            X_df, X_sc, targets, scaler, fcols = split_targets(df_feat)
            result = suite.forecast_next_hours(X_sc[-horizon_h:], horizon_h)
            forecast_aqi = np.clip(result["aqi_forecast"][:horizon_h], 10, 400)
            model_name   = result["model_used"]
        except Exception:
            forecast_aqi = np.clip(np.mean(recent_aqi[-24:]) + rng.normal(0, 8, horizon_h), 10, 400)
            model_name   = "fallback"
    else:
        forecast_aqi = np.clip(np.mean(recent_aqi[-24:]) + rng.normal(0, 8, horizon_h), 10, 400)
        model_name   = "trend_mean"

    forecast_pm25 = np.clip(forecast_aqi * 0.62 + rng.normal(0, 3, horizon_h), 5, 300)

    import pandas as pd
    last_ts = aqi_data["timestamp"].iloc[-1]
    ts_list = pd.date_range(last_ts, periods=horizon_h + 1, freq="h")[1:]

    return pd.DataFrame({
        "timestamp":    ts_list,
        "aqi":          forecast_aqi,
        "pm25_ugm3":    forecast_pm25,
        "type":         "forecast",
        "model":        model_name,
    })


with st.spinner("Generating forecast …"):
    forecast_df = generate_forecast(horizon, aqi_df)

# ── AQI category colour bands ─────────────────────────────────
AQI_BANDS = [
    ("Good",                "#2ecc71", 0,   50),
    ("Moderate",            "#f1c40f", 51, 100),
    ("Unhealthy-Sensitive", "#e67e22", 101,150),
    ("Unhealthy",           "#e74c3c", 151,200),
    ("Very Unhealthy",      "#9b59b6", 201,300),
    ("Hazardous",           "#7f0000", 301,400),
]

# ── Historical context ────────────────────────────────────────
hist_df = aqi_df.tail(show_history_days * 24).copy()
hist_df["type"] = "historical"

# ── Build Plotly chart ────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=1,
    shared_xaxes=True,
)

# AQI category bands
for label, color, lo, hi in AQI_BANDS:
    fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.07,
                  line_width=0, annotation_text=label,
                  annotation_position="right",
                  annotation_font_color=color,
                  annotation_font_size=10)

# Historical AQI line
fig.add_trace(go.Scatter(
    x=hist_df["timestamp"], y=hist_df["aqi"],
    name="Historical AQI", mode="lines",
    line=dict(color="#74b9ff", width=1.5, dash="dot"),
    opacity=0.8,
))

# Forecast AQI line
fig.add_trace(go.Scatter(
    x=forecast_df["timestamp"], y=forecast_df["aqi"],
    name=f"AQI Forecast ({forecast_df['model'].iloc[0]})",
    mode="lines+markers",
    line=dict(color="#fdcb6e", width=2.5),
    marker=dict(size=4, color="#fdcb6e"),
))

# Confidence band (simple ±10% for visual)
upper = forecast_df["aqi"] * 1.12
lower = forecast_df["aqi"] * 0.88
fig.add_trace(go.Scatter(
    x=pd.concat([forecast_df["timestamp"], forecast_df["timestamp"][::-1]]),
    y=pd.concat([upper, lower[::-1]]),
    fill="toself", fillcolor="rgba(253,203,110,0.12)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Confidence band", showlegend=True,
))

# PM2.5 overlay
if show_pm25:
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"], y=forecast_df["pm25_ugm3"],
        name="PM2.5 Forecast (µg/m³)", mode="lines",
        line=dict(color="#fd79a8", width=1.8, dash="dash"),
    ))

# Vertical separator line
fig.add_vline(
    x=aqi_df["timestamp"].iloc[-1].isoformat(),
    line=dict(color="white", width=1.5, dash="dash"),
    annotation_text="Now", annotation_font_color="white",
)

fig.update_layout(
    title=f"AQI + PM2.5 Forecast — Next {horizon}h",
    xaxis_title="Time",
    yaxis_title="AQI / PM2.5 (µg/m³)",
    template="plotly_dark",
    height=480,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=40, r=80, t=60, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── Current AQI metrics ───────────────────────────────────────
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)
current_aqi  = float(aqi_df["aqi"].iloc[-1])
current_pm25 = float(aqi_df["pm25_ugm3"].iloc[-1])
max_forecast = float(forecast_df["aqi"].max())
min_forecast = float(forecast_df["aqi"].min())
best_hour    = int(forecast_df.loc[forecast_df["aqi"].idxmin(), "timestamp"].hour)

with col1: st.metric("Current AQI", f"{current_aqi:.0f}")
with col2: st.metric("Current PM2.5 (µg/m³)", f"{current_pm25:.0f}")
with col3: st.metric(f"Peak in {horizon}h forecast", f"{max_forecast:.0f}")
with col4: st.metric(f"Best in {horizon}h forecast", f"{min_forecast:.0f}")
with col5: st.metric("Best hour to go out", f"{best_hour:02d}:00")

# ── Model comparison table ────────────────────────────────────
st.markdown("---")
st.subheader("📊 Model Comparison (MAE / RMSE per Horizon)")

suite = get_forecast_suite()
if suite:
    rows = []
    for model_name in ("LSTM", "GradientBoosting", "Prophet"):
        row = {"Model": model_name}
        for h in [24, 48, 72]:
            m = suite.results.get(model_name, {}).get(h, {})
            row[f"MAE@{h}h"] = f"{m['MAE']:.3f}" if m.get("MAE") is not None else "N/A"
            row[f"RMSE@{h}h"] = f"{m['RMSE']:.3f}" if m.get("RMSE") is not None else "N/A"
        rows.append(row)
    cmp_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(cmp_df, use_container_width=True)

    st.markdown("**Best model per horizon:**")
    for h in [24, 48, 72]:
        bname, _ = suite.best_models.get(h, ("N/A", None))
        st.markdown(f"  - **{h}h** → `{bname}`")
else:
    st.info("Train models first to see comparison: `python notebooks/03_aqi_forecast_training.py`")

st.caption("⚕️ NOT MEDICAL ADVICE | 🔬 SYNTHETIC DATA")
