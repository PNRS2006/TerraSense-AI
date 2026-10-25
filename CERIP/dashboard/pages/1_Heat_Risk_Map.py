"""
CERIP Dashboard — Page 1: Heat Risk Map
========================================
- Folium choropleth of the 500m grid coloured by heat risk class
- Sidebar controls: risk class filter, SHAP cell selector
- SHAP "why" panel for any selected cell
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Heat Risk Map — CERIP", page_icon="🌡️", layout="wide")

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.shap-card {
    background: #1e1e2e; border-radius: 12px; padding: 16px 20px; color: white;
    border-left: 4px solid #e74c3c; margin-top: 8px;
}
.disclaimer-box {
    background: #2d1b00; border: 1px solid #e67e22; border-radius: 8px;
    padding: 10px 14px; color: #f39c12; font-size: 0.82rem; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🌡️ Urban Heat Risk Map — Visakhapatnam")
st.markdown('<div class="disclaimer-box">⚕️ NOT MEDICAL ADVICE &nbsp;|&nbsp; 🔬 SYNTHETIC DATA — Not for operational use</div>',
            unsafe_allow_html=True)

# ── Load data ───────────────────────────────────────────────
@st.cache_data
def load_grid():
    csv_path = ROOT / "data" / "synthetic" / "vizag_grid_synthetic.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # Use centroid_lat/lon if available, else lat/lon columns
    if "centroid_lat" not in df.columns:
        df["centroid_lat"] = df["lat"]
        df["centroid_lon"] = df["lon"]
    return df


grid_df = load_grid()

if grid_df is None:
    st.error("Grid data not found. Run: `python src/data/synthetic_generator.py`")
    st.stop()

# ── Sidebar controls ─────────────────────────────────────────
with st.sidebar:
    st.header("🌡️ Heat Risk Controls")
    risk_filter = st.multiselect(
        "Show risk classes",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
        format_func=lambda x: {1:"Very Low", 2:"Low", 3:"Moderate",
                                4:"High", 5:"Very High"}[x],
    )
    max_cells = st.slider("Max cells to display", 100, min(2000, len(grid_df)), 800, step=100)
    selected_cell = st.number_input("Cell ID for SHAP explanation",
                                    min_value=0,
                                    max_value=len(grid_df)-1,
                                    value=0)
    st.caption(f"Total cells: {len(grid_df):,}")

# ── Filter & sample ───────────────────────────────────────────
filtered = grid_df[grid_df["heat_risk_class"].isin(risk_filter)].copy()
if len(filtered) > max_cells:
    filtered = filtered.sample(max_cells, random_state=42)

# ── Color map ─────────────────────────────────────────────────
RISK_COLORS = {
    1: "#2ecc71",   # Very Low  — green
    2: "#a8d8a8",   # Low       — light green
    3: "#f39c12",   # Moderate  — amber
    4: "#e67e22",   # High      — orange
    5: "#e74c3c",   # Very High — red
}
RISK_LABELS = {1:"Very Low", 2:"Low", 3:"Moderate", 4:"High", 5:"Very High"}

# ── Build Folium map ──────────────────────────────────────────
m = folium.Map(
    location=[17.72, 83.30],
    zoom_start=12,
    tiles="CartoDB dark_matter",
)

# Add legend (HTML)
legend_html = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
            background: rgba(0,0,0,0.8); border-radius: 10px; padding: 14px 18px;
            font-family: Inter, sans-serif; color: white; font-size: 13px;">
  <b>Heat Risk Class</b><br>
  <span style="color:#2ecc71">■</span> Very Low &nbsp;
  <span style="color:#a8d8a8">■</span> Low<br>
  <span style="color:#f39c12">■</span> Moderate &nbsp;
  <span style="color:#e67e22">■</span> High<br>
  <span style="color:#e74c3c">■</span> Very High
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Plot cells as CircleMarkers (faster than full polygons for prototype)
for _, row in filtered.iterrows():
    risk_cls = int(row["heat_risk_class"])
    color    = RISK_COLORS.get(risk_cls, "#aaa")
    popup_html = f"""
    <b>Cell {int(row['cell_id'])}</b><br>
    Risk: {RISK_LABELS[risk_cls]}<br>
    LST: {row['lst_celsius']:.1f}°C<br>
    NDVI: {row['ndvi']:.3f}<br>
    Building density: {row['building_density']:.2f}<br>
    Vulnerability: {row['vulnerability_index']:.3f}
    """
    folium.CircleMarker(
        location=[row["centroid_lat"], row["centroid_lon"]],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
        popup=folium.Popup(popup_html, max_width=220),
        tooltip=f"Class {risk_cls}: {RISK_LABELS[risk_cls]}",
    ).add_to(m)

# ── Layout ────────────────────────────────────────────────────
col_map, col_shap = st.columns([3, 1])

with col_map:
    st.subheader(f"Heat Risk Grid — {len(filtered):,} cells shown")
    map_out = st_folium(m, width=None, height=520)

with col_shap:
    st.subheader("📊 SHAP Explanation")

    if 0 <= selected_cell < len(grid_df):
        row = grid_df.iloc[selected_cell]
        risk_cls = int(row["heat_risk_class"])
        color    = RISK_COLORS.get(risk_cls, "#aaa")

        st.markdown(f"""
        <div class="shap-card">
        <b>Cell {selected_cell}</b><br>
        Risk Class: <b>{RISK_LABELS[risk_cls]}</b><br>
        LST: <b>{row['lst_celsius']:.1f}°C</b><br>
        NDVI: {row['ndvi']:.3f} | NDBI: {row['ndbi']:.3f}
        </div>
        """, unsafe_allow_html=True)

        # ── Run SHAP (cached) ──────────────────────────────────
        @st.cache_resource
        def get_heat_model_and_shap():
            """Load or train heat model and compute SHAP values."""
            from src.models.heat_model import HeatRiskModelSuite
            from src.features.heat_features import build_heat_feature_matrix
            from src.explainability.shap_explainer import explain_heat_model

            models_dir = ROOT / "models_saved"
            grid_data = load_grid()

            if (models_dir / "heat_regressor_best.joblib").exists():
                suite = HeatRiskModelSuite.load(models_dir)
            else:
                X_df, X_sc, y_r, y_c, scaler, fcols = build_heat_feature_matrix(grid_data)
                suite = HeatRiskModelSuite()
                suite.train(X_df, X_sc, y_r, y_c, fcols, scaler)
                suite.save(models_dir)

            X_df, _, _, _, _, feat_cols = build_heat_feature_matrix(grid_data)
            shap_result = explain_heat_model(
                suite.best_regressor, X_df, feat_cols, save_plots=True
            )
            return shap_result, feat_cols, X_df

        with st.spinner("Computing SHAP values …"):
            try:
                shap_result, feat_cols, X_df = get_heat_model_and_shap()
                from src.explainability.shap_explainer import get_top_drivers
                top_df = get_top_drivers(shap_result["shap_values"], feat_cols,
                                          selected_cell, n=8)
                st.markdown("**Top drivers for this cell:**")
                for _, dr_row in top_df.iterrows():
                    arrow = "🔴" if dr_row["shap_value"] > 0 else "🟢"
                    feat_val = ""
                    if dr_row["feature"] in X_df.columns:
                        feat_val = f"= {X_df.iloc[selected_cell][dr_row['feature']]:.3f}"
                    st.markdown(
                        f"{arrow} **{dr_row['feature']}** {feat_val}  "
                        f"({dr_row['shap_value']:+.3f})"
                    )
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")

        # ── SHAP bar plot image ────────────────────────────────
        bar_img = ROOT / "reports" / "figures" / "shap_bar_heat_regressor.png"
        if bar_img.exists():
            st.markdown("**Global feature importance:**")
            st.image(str(bar_img), use_column_width=True)

# ── Stats panel ───────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Grid Statistics")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.metric("Mean LST (°C)", f"{grid_df['lst_celsius'].mean():.1f}")
with col_s2:
    st.metric("High Risk Cells (≥4)", int((grid_df["heat_risk_class"] >= 4).sum()))
with col_s3:
    st.metric("Mean NDVI", f"{grid_df['ndvi'].mean():.3f}")
with col_s4:
    st.metric("Mean Vulnerability", f"{grid_df['vulnerability_index'].mean():.3f}")

st.caption("⚕️ NOT MEDICAL ADVICE | 🔬 SYNTHETIC DATA")
