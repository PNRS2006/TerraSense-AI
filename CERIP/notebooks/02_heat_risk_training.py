"""
CERIP — Notebook 02: Heat Risk Model Training (M3)
====================================================
Run: python notebooks/02_heat_risk_training.py

Sections
--------
  1. Load synthetic grid data
  2. Feature engineering
  3. Train RF + XGBoost + LightGBM (regression + classification)
  4. Evaluation: RMSE / MAE / R2  +  Accuracy / F1 / Confusion Matrix
  5. SHAP global summary + per-cell waterfall
  6. Save models

Outputs
-------
  models_saved/heat_regressor_best.joblib
  models_saved/heat_classifier_best.joblib
  models_saved/heat_scaler.joblib
  reports/figures/shap_beeswarm_heat_regressor.png
  reports/figures/shap_bar_heat_regressor.png
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
log.info("SECTION 1: Loading synthetic grid data")
log.info("=" * 60)

import pandas as pd

parquet_path = ROOT / "data" / "processed" / "grid_features.parquet"
if not parquet_path.exists():
    log.info("Parquet not found. Generating synthetic data …")
    from src.data.synthetic_generator import main as gen_main
    gen_main()

grid_df = pd.read_parquet(parquet_path)
log.info("Loaded grid: %d rows, %d columns", *grid_df.shape)
log.info("Heat risk class distribution:\n%s",
         grid_df["heat_risk_label"].value_counts().to_string())


# ============================================================
# Section 2 — Feature Engineering
# ============================================================
log.info("\n" + "=" * 60)
log.info("SECTION 2: Feature Engineering")
log.info("=" * 60)

from src.features.heat_features import build_heat_feature_matrix, get_feature_descriptions

X_df, X_scaled, y_reg, y_clf, scaler, feature_cols = build_heat_feature_matrix(grid_df)

log.info("Feature matrix: %s", X_df.shape)
log.info("Regression target (LST): mean=%.2f°C  std=%.2f°C",
         y_reg.mean(), y_reg.std())
log.info("Classification target classes: %s", sorted(set(y_clf)))

print("\n[Feature descriptions]")
feat_desc = get_feature_descriptions()
for fc in feature_cols[:8]:
    print(f"  {fc:<30} → {feat_desc.get(fc, '')}")
print("  … (total features:", len(feature_cols), ")")


# ============================================================
# Section 3 — Train Models
# ============================================================
log.info("\n" + "=" * 60)
log.info("SECTION 3: Training RF + XGBoost + LightGBM")
log.info("=" * 60)

from src.models.heat_model import HeatRiskModelSuite

suite = HeatRiskModelSuite()
suite.train(X_df, X_scaled, y_reg, y_clf, feature_cols, scaler)


# ============================================================
# Section 4 — Evaluation Report
# ============================================================
log.info("\n" + "=" * 60)
log.info("SECTION 4: Evaluation Report")
log.info("=" * 60)

suite.print_report()

# Also print confusion matrix for best classifier
import numpy as np
best_clf_name = max(suite.clf_results,
                    key=lambda k: suite.clf_results[k]["F1_weighted"])
cm = suite.clf_results[best_clf_name]["confusion_matrix"]
print(f"\n[Confusion Matrix — {best_clf_name}]")
print("         VLow  Low  Mod  High  VHigh")
for i, row_cm in enumerate(cm):
    label = {1:"VLow", 2:"Low", 3:"Mod", 4:"High", 5:"VHigh"}[i+1]
    print(f"  {label:<6} {row_cm}")


# ============================================================
# Section 5 — SHAP Explainability
# ============================================================
log.info("\n" + "=" * 60)
log.info("SECTION 5: SHAP Explainability")
log.info("=" * 60)

from src.explainability.shap_explainer import (
    explain_heat_model, explain_single_prediction, plot_cell_waterfall
)

# Global SHAP (saves beeswarm + bar plots automatically)
log.info("Computing global SHAP for best regressor …")
shap_result = explain_heat_model(
    suite.best_regressor, X_df, feature_cols, save_plots=True
)

print("\n[Global Feature Importance — Top 10]")
print(shap_result["global_importance"].head(10)[["feature", "mean_abs_shap"]].to_string(index=False))

# Per-cell waterfall (cell 0 and highest-risk cell)
high_risk_idx = int(np.argmax(y_reg))
log.info("Generating waterfall for cell %d (highest LST = %.2f°C) …",
         high_risk_idx, y_reg[high_risk_idx])

plot_cell_waterfall(
    shap_result, high_risk_idx,
    title=f"SHAP Waterfall — Highest Risk Cell {high_risk_idx}"
)

# Per-cell narrative
explanation = explain_single_prediction(shap_result, high_risk_idx,
                                         feature_value_df=X_df)
print(f"\n[SHAP Narrative — Cell {high_risk_idx}]")
print(explanation["narrative"])


# ============================================================
# Section 6 — Save Models
# ============================================================
log.info("\n" + "=" * 60)
log.info("SECTION 6: Saving models")
log.info("=" * 60)

suite.save()

print("\n" + "=" * 60)
print("✅  Heat Risk Training Complete!")
print(f"    Best regressor : {max(suite.reg_results, key=lambda k: suite.reg_results[k]['R2'])}")
print(f"    Best classifier: {max(suite.clf_results, key=lambda k: suite.clf_results[k]['F1_weighted'])}")
print(f"    SHAP plots     : {ROOT / 'reports' / 'figures'}")
print(f"    Models saved   : {ROOT / 'models_saved'}")
print("=" * 60)
