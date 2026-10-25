"""
CERIP — Heat Risk Models (M3)
==============================
Module A: Urban Heat Risk Intelligence System

Trains, evaluates, and serialises two models:
  1. Regressor  → continuous Land Surface Temperature (LST, °C)
  2. Classifier → heat risk class (1=Very Low … 5=Very High)

Models trained
--------------
  - Random Forest (baseline)
  - XGBoost
  - LightGBM
  Best model (by val metric) is saved to models_saved/.

Usage
-----
    from src.models.heat_model import HeatRiskModelSuite
    suite = HeatRiskModelSuite()
    suite.train(X_df, X_scaled, y_reg, y_clf, feature_names)
    suite.evaluate()
    suite.save()
    preds = suite.predict(X_new_scaled)
"""

import logging
import warnings
import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models_saved"
MODELS_DIR.mkdir(exist_ok=True)

with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

RF_CFG  = CFG["models"]["random_forest"]
XGB_CFG = CFG["models"]["xgboost"]
LGB_CFG = CFG["models"]["lightgbm"]


# =============================================================
# Utility helpers
# =============================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")
    f1_m = f1_score(y_true, y_pred, average="macro")
    cm   = confusion_matrix(y_true, y_pred)
    rpt  = classification_report(y_true, y_pred,
                                  target_names=[CFG["heat_risk"]["risk_labels"][i]
                                                for i in sorted(CFG["heat_risk"]["risk_labels"])])
    return {
        "Accuracy":  round(acc, 4),
        "F1_weighted": round(f1_w, 4),
        "F1_macro":  round(f1_m, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": rpt,
    }


# =============================================================
# Model Suite
# =============================================================

class HeatRiskModelSuite:
    """
    Trains and evaluates a suite of regression + classification models
    for the Urban Heat Risk module.

    Attributes (set after train())
    --------------------------------
    best_regressor    : fitted best regression model
    best_classifier   : fitted best classification model
    reg_results       : dict {model_name: regression_metrics}
    clf_results       : dict {model_name: classification_metrics}
    feature_names     : list[str]
    scaler            : fitted StandardScaler from feature engineering
    """

    def __init__(self):
        self.best_regressor  : Any = None
        self.best_classifier : Any = None
        self.reg_results     : dict = {}
        self.clf_results     : dict = {}
        self.feature_names   : list = []
        self.scaler          = None
        self._X_train = self._X_val = None
        self._y_reg_train = self._y_reg_val = None
        self._y_clf_train = self._y_clf_val = None
        self._X_train_unscaled = None   # kept for SHAP (tree models prefer raw)

    # ── Data split ───────────────────────────────────────────────
    def prepare_data(
        self,
        X_df: pd.DataFrame,
        X_scaled: np.ndarray,
        y_reg: np.ndarray,
        y_clf: np.ndarray,
        test_size: float = 0.2,
        random_state: int = RF_CFG["random_state"],
    ):
        idx = np.arange(len(X_df))
        tr_idx, va_idx = train_test_split(idx, test_size=test_size,
                                           random_state=random_state,
                                           stratify=y_clf)

        self._X_train = X_scaled[tr_idx]
        self._X_val   = X_scaled[va_idx]
        self._X_train_unscaled = X_df.iloc[tr_idx].values
        self._X_val_unscaled   = X_df.iloc[va_idx].values
        self._y_reg_train = y_reg[tr_idx]
        self._y_reg_val   = y_reg[va_idx]
        self._y_clf_train = y_clf[tr_idx]
        self._y_clf_val   = y_clf[va_idx]
        log.info("Train/Val split: %d / %d", len(tr_idx), len(va_idx))

    # ── Train all models ─────────────────────────────────────────
    def train(
        self,
        X_df: pd.DataFrame,
        X_scaled: np.ndarray,
        y_reg: np.ndarray,
        y_clf: np.ndarray,
        feature_names: list[str],
        scaler=None,
    ):
        self.feature_names = feature_names
        self.scaler = scaler
        self.prepare_data(X_df, X_scaled, y_reg, y_clf)

        log.info("=== Training regression models ===")
        self._train_regressors()

        log.info("=== Training classification models ===")
        self._train_classifiers()

    # ── Regressors ────────────────────────────────────────────────
    def _train_regressors(self):
        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=RF_CFG["n_estimators"],
                max_depth=RF_CFG["max_depth"],
                random_state=RF_CFG["random_state"],
                n_jobs=RF_CFG["n_jobs"],
            ),
            "XGBoost": xgb.XGBRegressor(
                n_estimators=XGB_CFG["n_estimators"],
                max_depth=XGB_CFG["max_depth"],
                learning_rate=XGB_CFG["learning_rate"],
                random_state=XGB_CFG["random_state"],
                verbosity=0,
            ),
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=LGB_CFG["n_estimators"],
                max_depth=LGB_CFG["max_depth"],
                learning_rate=LGB_CFG["learning_rate"],
                random_state=LGB_CFG["random_state"],
                verbose=-1,
            ),
        }

        best_rmse   = float("inf")
        best_name   = None

        for name, model in models.items():
            log.info("  Fitting %s …", name)
            # Tree models work fine on raw features
            X_tr = self._X_train_unscaled if name in ("XGBoost", "LightGBM", "RandomForest") \
                   else self._X_train
            X_va = self._X_val_unscaled if name in ("XGBoost", "LightGBM", "RandomForest") \
                   else self._X_val

            model.fit(X_tr, self._y_reg_train)
            preds = model.predict(X_va)
            m = regression_metrics(self._y_reg_val, preds)
            self.reg_results[name] = m
            log.info("  %s reg: RMSE=%.4f  MAE=%.4f  R²=%.4f",
                     name, m["RMSE"], m["MAE"], m["R2"])

            if m["RMSE"] < best_rmse:
                best_rmse = m["RMSE"]
                best_name = name
                self.best_regressor = model

        log.info("Best regressor: %s  (RMSE=%.4f)", best_name, best_rmse)

    # ── Classifiers ───────────────────────────────────────────────
    def _train_classifiers(self):
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=RF_CFG["n_estimators"],
                max_depth=RF_CFG["max_depth"],
                random_state=RF_CFG["random_state"],
                n_jobs=RF_CFG["n_jobs"],
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=XGB_CFG["n_estimators"],
                max_depth=XGB_CFG["max_depth"],
                learning_rate=XGB_CFG["learning_rate"],
                random_state=XGB_CFG["random_state"],
                eval_metric="mlogloss",
                verbosity=0,
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=LGB_CFG["n_estimators"],
                max_depth=LGB_CFG["max_depth"],
                learning_rate=LGB_CFG["learning_rate"],
                random_state=LGB_CFG["random_state"],
                verbose=-1,
            ),
        }

        best_f1   = 0
        best_name = None

        for name, model in models.items():
            log.info("  Fitting %s …", name)
            # Adjust class labels for XGBoost (needs 0-indexed)
            y_tr = self._y_clf_train - 1 if name == "XGBoost" else self._y_clf_train
            y_va = self._y_clf_val   - 1 if name == "XGBoost" else self._y_clf_val

            model.fit(self._X_train_unscaled, y_tr)
            raw_pred = model.predict(self._X_val_unscaled)
            preds = raw_pred + 1 if name == "XGBoost" else raw_pred
            y_va_adj = y_va + 1 if name == "XGBoost" else y_va

            m = classification_metrics(y_va_adj, preds)
            self.clf_results[name] = m
            log.info("  %s clf: Acc=%.4f  F1_w=%.4f  F1_m=%.4f",
                     name, m["Accuracy"], m["F1_weighted"], m["F1_macro"])

            if m["F1_weighted"] > best_f1:
                best_f1   = m["F1_weighted"]
                best_name = name
                self.best_classifier = model

        log.info("Best classifier: %s  (F1_weighted=%.4f)", best_name, best_f1)

    # ── Prediction API ────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> dict:
        """
        Run both best models on X.

        Returns
        -------
        dict with:
            lst_pred   : continuous LST predictions
            risk_class : integer class 1-5
            risk_label : string label
        """
        lst_pred   = self.best_regressor.predict(X)
        risk_class = self.best_classifier.predict(X)
        # Handle XGBoost 0-indexed classes
        if hasattr(self.best_classifier, "classes_") and 0 in self.best_classifier.classes_:
            risk_class = risk_class + 1
        risk_label = [CFG["heat_risk"]["risk_labels"].get(int(c), "Unknown")
                      for c in risk_class]
        return {
            "lst_pred":        lst_pred,
            "risk_class":      risk_class,
            "heat_risk_class": risk_class,
            "risk_label":      risk_label,
        }

    def predict_single(self, x_row: np.ndarray) -> dict:
        """Predict for a single grid cell (1D array or 2D row)."""
        if x_row.ndim == 1:
            x_row = x_row.reshape(1, -1)
        result = self.predict(x_row)
        return {k: v[0] if isinstance(v, (list, np.ndarray)) else v
                for k, v in result.items()}

    # ── Serialise ─────────────────────────────────────────────────
    def save(self, directory: Path = MODELS_DIR):
        joblib.dump(self.best_regressor,  directory / "heat_regressor_best.joblib")
        joblib.dump(self.best_classifier, directory / "heat_classifier_best.joblib")
        if self.scaler is not None:
            joblib.dump(self.scaler, directory / "heat_scaler.joblib")
        log.info("Models saved to %s", directory)

    @classmethod
    def load(cls, directory: Path = MODELS_DIR) -> "HeatRiskModelSuite":
        suite = cls()
        suite.best_regressor  = joblib.load(directory / "heat_regressor_best.joblib")
        suite.best_classifier = joblib.load(directory / "heat_classifier_best.joblib")
        scaler_path = directory / "heat_scaler.joblib"
        if scaler_path.exists():
            suite.scaler = joblib.load(scaler_path)
        log.info("Models loaded from %s", directory)
        return suite

    # ── Summary report ────────────────────────────────────────────
    def print_report(self):
        print("\n" + "="*60)
        print("MODULE A — HEAT RISK MODEL EVALUATION REPORT")
        print("="*60)
        print("\n[REGRESSION — Land Surface Temperature Prediction]")
        for name, m in self.reg_results.items():
            print(f"  {name:<15}  RMSE={m['RMSE']:.4f}°C  MAE={m['MAE']:.4f}°C  R²={m['R2']:.4f}")

        print("\n[CLASSIFICATION — Heat Risk Class (1=Very Low … 5=Very High)]")
        for name, m in self.clf_results.items():
            print(f"  {name:<15}  Accuracy={m['Accuracy']:.4f}  F1_weighted={m['F1_weighted']:.4f}  F1_macro={m['F1_macro']:.4f}")

        # Print full report for best classifier
        best_clf_name = max(self.clf_results, key=lambda k: self.clf_results[k]["F1_weighted"])
        print(f"\n[CLASSIFICATION REPORT — {best_clf_name}]")
        print(self.clf_results[best_clf_name]["classification_report"])
