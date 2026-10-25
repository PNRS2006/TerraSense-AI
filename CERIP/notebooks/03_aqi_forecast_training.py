"""
CERIP — Notebook 03: AQI Forecast Model Training (M4)
=======================================================
Run: python notebooks/03_aqi_forecast_training.py

Sections
--------
  1. Load AQI time-series
  2. Feature engineering (lag, rolling, calendar, interactions)
  3. Train LSTM + Prophet + Gradient Boosting
  4. Evaluation: MAE / RMSE per horizon (24h, 48h, 72h)
  5. Model comparison table
  6. SHAP for GradientBoosting model
  7. Save models

Outputs
-------
  models_saved/aqi_gb_24h.joblib   (+ 48h, 72h)
  models_saved/aqi_lstm_24h.keras  (+ 48h, 72h)
  models_saved/aqi_prophet.pkl
  models_saved/aqi_scaler.joblib
  reports/figures/shap_beeswarm_aqi_gb.png
  reports/figures/shap_bar_aqi_gb.png
"""

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ============================================================
# Section 1 — Load Data
# ============================================================
log.info("=" * 60)
log.info("SECTION 1: Loading AQI time-series")
log.info("=" * 60)

import pandas as pd
import numpy as np

parquet_path = ROOT / "data" / "processed" / "aqi_timeseries.parquet"
if not parquet_path.exists():
    log.info("Generating synthetic AQI data …")
    from src.data.synthetic_generator import main as gen_main
    gen_main()

aqi_df = pd.read_parquet(parquet_path)
log.info("Loaded AQI: %d rows  |  %s → %s",
         len(aqi_df),
         aqi_df["timestamp"].min().date(),
         aqi_df["timestamp"].max().date())
log.info("AQI stats:\n%s", aqi_df["aqi"].describe().round(1).to_string())
log.info("AQI category distribution:\n%s",
         aqi_df["aqi_category"].value_counts().to_string())


# ============================================================
# Section 2 — Feature Engineering
# ============================================================
log.info("\n" + "=" * 60)
log.info("SECTION 2: Feature Engineering")
log.info("=" * 60)

from src.features.aqi_features import build_aqi_feature_matrix, split_targets

df_feat = build_aqi_feature_matrix(aqi_df)
X_df, X_scaled, targets, scaler, feature_cols = split_targets(df_feat)

log.info("Feature matrix: %s  |  Features: %d", X_df.shape, len(feature_cols))
for h in [24, 48, 72]:
    log.info("  Target AQI @%dh: mean=%.1f  std=%.1f",
             h, targets[h].mean(), targets[h].std())

print("\n[Feature columns sample]")
for fc in feature_cols[:10]:
    print(f"  {fc}")
print(f"  … (total: {len(feature_cols)} features)")


# ============================================================
# Section 3 — Train Models
# ============================================================
log.info("\n" + "=" * 60)
log.info("SECTION 3: Training LSTM + Prophet + GradientBoosting")
log.info("=" * 60)

from src.models.aqi_forecast import AQIForecastSuite

suite = AQIForecastSuite()
suite.train(
    df_feat=df_feat,
    X_df=X_df,
    X_scaled=X_scaled,
    targets=targets,
    scaler=scaler,
    feature_cols=feature_cols,
    aqi_df=aqi_df,
)


# ============================================================
# Section 4 + 5 — Evaluation & Model Comparison
# ============================================================
log.info("\n" + "=" * 60)
log.info("SECTIONS 4+5: Evaluation Report & Model Comparison")
log.info("=" * 60)

suite.print_report()


# ============================================================
# Section 6 — SHAP (GradientBoosting, fastest explainer)
# ============================================================
log.info("\n" + "=" * 60)
log.info("SECTION 6: SHAP for GradientBoosting (24h horizon)")
log.info("=" * 60)

from src.explainability.shap_explainer import explain_aqi_model, get_top_drivers

if 24 in suite.gb_models:
    # Use most recent 1000 rows for SHAP (keeps it fast)
    X_shap = X_df.tail(1000).reset_index(drop=True)
    shap_result = explain_aqi_model(
        suite.gb_models[24], X_shap, feature_cols,
        label="aqi_gb", save_plots=True
    )

    print("\n[AQI Model — Global Feature Importance (Top 10)]")
    print(shap_result["global_importance"].head(10)[["feature", "mean_abs_shap"]].to_string(index=False))

    # Per-observation narrative (most recent data point)
    from src.explainability.shap_explainer import explain_single_prediction
    exp = explain_single_prediction(shap_result, idx=-1, n_top=5, feature_value_df=X_shap)
    print("\n[SHAP Narrative — Most Recent AQI Observation]")
    print(exp["narrative"])
else:
    log.warning("GB model for 24h not available for SHAP.")


# ============================================================
# Section 7 — Save Models
# ============================================================
log.info("\n" + "=" * 60)
log.info("SECTION 7: Saving models")
log.info("=" * 60)

suite.save()

print("\n" + "=" * 60)
print("✅  AQI Forecast Training Complete!")
print(f"    Models saved : {ROOT / 'models_saved'}")
print(f"    SHAP plots   : {ROOT / 'reports' / 'figures'}")
for h in [24, 48, 72]:
    best_name, _ = suite.best_models.get(h, ("N/A", None))
    print(f"    Best @{h}h     : {best_name}")
print("=" * 60)
