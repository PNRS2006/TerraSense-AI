"""
CERIP — Tests: Heat Model
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np
import pandas as pd
from src.data.synthetic_generator import generate_grid_features
from src.features.heat_features import build_heat_feature_matrix
from src.models.heat_model import HeatRiskModelSuite, regression_metrics, classification_metrics


@pytest.fixture(scope="module")
def trained_suite():
    """Train a small heat model suite for testing."""
    grid_df = generate_grid_features(n=200, seed=0)  # small for speed
    X_df, X_scaled, y_reg, y_clf, scaler, feat_cols = build_heat_feature_matrix(grid_df)
    suite = HeatRiskModelSuite()
    suite.train(X_df, X_scaled, y_reg, y_clf, feat_cols, scaler)
    return suite, X_df, X_scaled, y_reg, y_clf, feat_cols


def test_regression_metrics_keys():
    y_true = np.array([30, 35, 40])
    y_pred = np.array([31, 34, 41])
    m = regression_metrics(y_true, y_pred)
    assert set(m.keys()) == {"RMSE", "MAE", "R2"}


def test_classification_metrics_keys():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    m = classification_metrics(y_true, y_pred)
    assert "Accuracy" in m and "F1_weighted" in m and "confusion_matrix" in m


def test_suite_trains_three_regressors(trained_suite):
    suite, *_ = trained_suite
    assert len(suite.reg_results) == 3
    for name, m in suite.reg_results.items():
        assert m["RMSE"] >= 0


def test_suite_trains_three_classifiers(trained_suite):
    suite, *_ = trained_suite
    assert len(suite.clf_results) == 3
    for name, m in suite.clf_results.items():
        assert 0 <= m["Accuracy"] <= 1


def test_suite_best_regressor_set(trained_suite):
    suite, *_ = trained_suite
    assert suite.best_regressor is not None


def test_suite_predict_shape(trained_suite):
    suite, X_df, X_scaled, y_reg, y_clf, feat_cols = trained_suite
    preds = suite.predict(X_df.values[:10])
    assert len(preds["lst_pred"]) == 10
    assert len(preds["risk_class"]) == 10
    assert len(preds["risk_label"]) == 10


def test_suite_predict_single(trained_suite):
    suite, X_df, *_ = trained_suite
    result = suite.predict_single(X_df.values[0])
    assert "lst_pred" in result
    assert result["heat_risk_class"] in [1, 2, 3, 4, 5] or result["heat_risk_class"] in range(6)


def test_regression_r2_positive(trained_suite):
    """R² should be positive on synthetic data (well-structured)."""
    suite, *_ = trained_suite
    best_name = max(suite.reg_results, key=lambda k: suite.reg_results[k]["R2"])
    assert suite.reg_results[best_name]["R2"] > 0.5
