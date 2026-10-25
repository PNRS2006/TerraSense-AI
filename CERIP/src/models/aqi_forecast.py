"""
CERIP — AQI Forecast Models (M4)
==================================
Module B: Air Quality Prediction & Activity Recommendation

Trains and evaluates three model types for 24h / 48h / 72h AQI forecasting:
  1. LSTM              (deep learning, sequence-based)
  2. Prophet           (additive time-series decomposition)
  3. Gradient Boosting (sklearn, tabular, fastest to train)

Evaluation: MAE and RMSE per horizon (24h, 48h, 72h).
Best model per horizon is logged and saved.

Usage
-----
    from src.models.aqi_forecast import AQIForecastSuite
    suite = AQIForecastSuite()
    suite.train(df_feat, X_df, X_scaled, targets, scaler, feature_cols, aqi_df)
    suite.evaluate()
    suite.save()
    forecast = suite.predict_next_72h(recent_X, recent_timestamps)
"""

import logging
import warnings
import pickle
import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yaml

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models_saved"
MODELS_DIR.mkdir(exist_ok=True)

with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

HORIZONS  = CFG["air_quality"]["forecast_horizons"]   # [24, 48, 72]
LSTM_CFG  = CFG["models"]["lstm"]
GB_CFG    = CFG["models"]["gradient_boosting"]


# =============================================================
# Utility
# =============================================================

def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4)}


# =============================================================
# LSTM Model
# =============================================================

def build_lstm_model(seq_len: int, n_features: int, units: int = 64,
                     dropout: float = 0.2) -> "keras.Model":
    """Build a single-layer LSTM model (laptop-friendly)."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("tensorflow not installed. Run: pip install tensorflow")

    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(units, dropout=dropout, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dropout(dropout / 2),
        layers.Dense(1),
    ], name="aqi_lstm")

    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model


def train_lstm(
    X_scaled: np.ndarray,
    y: np.ndarray,
    seq_len: int = 24,
    units: int = LSTM_CFG["units"],
    dropout: float = LSTM_CFG["dropout"],
    epochs: int = LSTM_CFG["epochs"],
    batch_size: int = LSTM_CFG["batch_size"],
    validation_split: float = 0.15,
) -> tuple:
    """
    Train LSTM on sequence data.
    Returns (model, history, train_idx, val_idx).
    """
    from src.features.aqi_features import make_lstm_sequences

    # Chronological split (not random, to avoid data leakage)
    split = int(len(X_scaled) * (1 - validation_split))
    X_tr, X_va = X_scaled[:split], X_scaled[split:]
    y_tr, y_va = y[:split], y[split:]

    X_tr_seq, y_tr_seq = make_lstm_sequences(X_tr, y_tr, seq_len)
    X_va_seq, y_va_seq = make_lstm_sequences(X_va, y_va, seq_len)

    log.info("LSTM train sequences: %s  val: %s", X_tr_seq.shape, X_va_seq.shape)

    model = build_lstm_model(seq_len, X_scaled.shape[1], units, dropout)

    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(patience=3, factor=0.5, verbose=0),
        ]
        history = model.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_va_seq, y_va_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )
    except Exception as e:
        log.warning("LSTM training error: %s. Trying without callbacks.", e)
        history = model.fit(X_tr_seq, y_tr_seq, epochs=min(epochs, 10),
                            batch_size=batch_size, verbose=0)

    return model, history, X_va_seq, y_va_seq


# =============================================================
# Prophet Model
# =============================================================

def train_prophet(aqi_df: pd.DataFrame) -> tuple:
    """
    Train a Prophet model on the full AQI time-series.
    Prophet expects a DataFrame with columns: ds (datetime), y (value).
    Returns (model, forecast_df).
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("prophet not installed. Run: pip install prophet")

    prophet_df = aqi_df[["timestamp", "aqi"]].rename(
        columns={"timestamp": "ds", "aqi": "y"}
    )
    # Sub-sample to daily mean to speed up fitting
    prophet_daily = (
        prophet_df.set_index("ds")
        .resample("D")
        .mean()
        .reset_index()
        .rename(columns={"ds": "ds", "y": "y"})
    )

    log.info("Fitting Prophet on %d daily points …", len(prophet_daily))

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
    )
    # Add Indian monsoon seasonality
    model.add_seasonality(name="monsoon", period=365.25 / 4, fourier_order=5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_daily)

    # In-sample forecast for evaluation
    future = model.make_future_dataframe(periods=3, freq="D")
    forecast = model.predict(future)
    return model, forecast, prophet_daily


# =============================================================
# Gradient Boosting Model
# =============================================================

def train_gb(
    X_scaled: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> tuple:
    """Train sklearn GradientBoostingRegressor (tabular baseline)."""
    # Chronological split
    split = int(len(X_scaled) * (1 - test_size))
    X_tr, X_va = X_scaled[:split], X_scaled[split:]
    y_tr, y_va = y[:split], y[split:]

    model = GradientBoostingRegressor(
        n_estimators=GB_CFG["n_estimators"],
        max_depth=GB_CFG["max_depth"],
        learning_rate=GB_CFG["learning_rate"],
        random_state=GB_CFG["random_state"],
    )
    log.info("Fitting GradientBoosting …")
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    m = forecast_metrics(y_va, preds)
    log.info("GB: MAE=%.4f  RMSE=%.4f", m["MAE"], m["RMSE"])
    return model, preds, y_va, m


# =============================================================
# Forecast Suite
# =============================================================

class AQIForecastSuite:
    """
    Trains LSTM, Prophet, and Gradient Boosting for 3 horizons.

    Attributes (set after train())
    --------------------------------
    lstm_models     : dict {horizon: keras Model}
    prophet_model   : fitted Prophet model
    gb_models       : dict {horizon: sklearn GB model}
    results         : dict {model_name: {horizon: metrics}}
    best_models     : dict {horizon: (model_name, model_object)}
    """

    def __init__(self):
        self.lstm_models   : dict = {}
        self.prophet_model  = None
        self.gb_models      : dict = {}
        self.results        : dict = {"LSTM": {}, "Prophet": {}, "GradientBoosting": {}}
        self.best_models    : dict = {}
        self.feature_cols   : list = []
        self.scaler         = None
        self._aqi_df        = None

    def train(
        self,
        df_feat: pd.DataFrame,
        X_df: pd.DataFrame,
        X_scaled: np.ndarray,
        targets: dict,           # {24: y_24, 48: y_48, 72: y_72}
        scaler,
        feature_cols: list[str],
        aqi_df: pd.DataFrame,   # raw time-series for Prophet
    ):
        self.feature_cols = feature_cols
        self.scaler       = scaler
        self._aqi_df      = aqi_df

        # ── Gradient Boosting (all horizons) ──────────────────────
        log.info("=== Training Gradient Boosting ===")
        for h in HORIZONS:
            y = targets[h]
            gb_model, gb_preds, y_va, m = train_gb(X_scaled, y)
            self.gb_models[h] = gb_model
            self.results["GradientBoosting"][h] = m
            log.info("  GB @%dh: MAE=%.4f  RMSE=%.4f", h, m["MAE"], m["RMSE"])

        # ── LSTM (24h primary, 48h, 72h) ─────────────────────────
        log.info("=== Training LSTM ===")
        seq_len = LSTM_CFG["sequence_length"]
        for h in HORIZONS:
            y = targets[h]
            try:
                lstm_model, _, X_va_seq, y_va_seq = train_lstm(
                    X_scaled, y, seq_len=seq_len
                )
                lstm_preds = lstm_model.predict(X_va_seq, verbose=0).flatten()
                m = forecast_metrics(y_va_seq, lstm_preds)
                self.lstm_models[h] = lstm_model
                self.results["LSTM"][h] = m
                log.info("  LSTM @%dh: MAE=%.4f  RMSE=%.4f", h, m["MAE"], m["RMSE"])
            except Exception as e:
                log.error("LSTM @%dh failed: %s. Skipping.", h, e)
                self.results["LSTM"][h] = {"MAE": None, "RMSE": None}

        # ── Prophet (daily granularity, horizon in days) ──────────
        log.info("=== Training Prophet ===")
        try:
            p_model, p_forecast, p_daily = train_prophet(aqi_df)
            self.prophet_model = p_model
            # Evaluate on last 3 days (held-out)
            n_eval = min(3, len(p_daily) - 1)
            y_true_p = p_daily["y"].values[-n_eval:]
            y_pred_p = p_forecast["yhat"].values[-(n_eval + 3):-3][:n_eval]
            for i, h in enumerate(HORIZONS):
                # Use daily granularity proxy
                day_idx = (h // 24) - 1
                if 0 <= day_idx < n_eval:
                    m_p = {
                        "MAE":  round(float(np.abs(y_true_p[day_idx] - y_pred_p[day_idx]).mean()), 4),
                        "RMSE": round(float(np.sqrt(((y_true_p[day_idx] - y_pred_p[day_idx]) ** 2).mean())), 4),
                    }
                else:
                    m_p = {"MAE": None, "RMSE": None}
                self.results["Prophet"][h] = m_p
                log.info("  Prophet @%dh (daily proxy): MAE=%.4f  RMSE=%.4f",
                         h, m_p["MAE"] or 0, m_p["RMSE"] or 0)
        except Exception as e:
            log.error("Prophet training failed: %s", e)
            for h in HORIZONS:
                self.results["Prophet"][h] = {"MAE": None, "RMSE": None}

        # ── Select best model per horizon ─────────────────────────
        self._select_best()

    def _select_best(self):
        """Choose the model with lowest MAE per horizon."""
        for h in HORIZONS:
            best_name, best_mae = None, float("inf")
            for model_name in ("LSTM", "GradientBoosting", "Prophet"):
                m = self.results[model_name].get(h, {})
                mae = m.get("MAE")
                if mae is not None and mae < best_mae:
                    best_mae  = mae
                    best_name = model_name
            if best_name == "LSTM":
                obj = self.lstm_models.get(h)
            elif best_name == "GradientBoosting":
                obj = self.gb_models.get(h)
            else:
                obj = self.prophet_model
            self.best_models[h] = (best_name, obj)
            log.info("Best model @%dh: %s (MAE=%.4f)", h, best_name, best_mae)

    def predict_gb(self, X_scaled: np.ndarray, horizon: int) -> np.ndarray:
        """Point forecast using GB model."""
        return self.gb_models[horizon].predict(X_scaled)

    def forecast_next_hours(
        self, X_recent_scaled: np.ndarray, horizon: int
    ) -> dict:
        """
        Produce a forecast for the next `horizon` hours.
        Uses best model for the given horizon.
        Returns dict with predictions and model used.
        """
        model_name, model_obj = self.best_models.get(
            horizon, ("GradientBoosting", self.gb_models.get(horizon))
        )
        if model_name == "GradientBoosting":
            preds = model_obj.predict(X_recent_scaled)
        elif model_name == "LSTM" and model_obj is not None:
            seq_len = LSTM_CFG["sequence_length"]
            if X_recent_scaled.shape[0] >= seq_len:
                X_seq = X_recent_scaled[-seq_len:].reshape(1, seq_len, -1)
                preds = model_obj.predict(X_seq, verbose=0).flatten()
            else:
                preds = self.gb_models[horizon].predict(X_recent_scaled)
        else:
            preds = np.zeros(horizon)

        return {
            "horizon_h": horizon,
            "model_used": model_name,
            "aqi_forecast": preds.tolist() if hasattr(preds, "tolist") else preds,
        }

    def save(self, directory: Path = MODELS_DIR):
        # Save GB models
        for h, m in self.gb_models.items():
            joblib.dump(m, directory / f"aqi_gb_{h}h.joblib")
        # Save LSTM models
        for h, m in self.lstm_models.items():
            try:
                m.save(str(directory / f"aqi_lstm_{h}h.keras"))
            except Exception as e:
                log.warning("Could not save LSTM @%dh: %s", h, e)
        # Save Prophet
        if self.prophet_model is not None:
            with open(directory / "aqi_prophet.pkl", "wb") as f:
                pickle.dump(self.prophet_model, f)
        # Save scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, directory / "aqi_scaler.joblib")
        log.info("AQI models saved to %s", directory)

    @classmethod
    def load(cls, directory: Path = MODELS_DIR) -> "AQIForecastSuite":
        suite = cls()
        for h in HORIZONS:
            p = directory / f"aqi_gb_{h}h.joblib"
            if p.exists():
                suite.gb_models[h] = joblib.load(p)
        prophet_path = directory / "aqi_prophet.pkl"
        if prophet_path.exists():
            with open(prophet_path, "rb") as f:
                suite.prophet_model = pickle.load(f)
        scaler_path = directory / "aqi_scaler.joblib"
        if scaler_path.exists():
            suite.scaler = joblib.load(scaler_path)
        log.info("AQI models loaded from %s", directory)
        return suite

    def print_report(self):
        print("\n" + "="*65)
        print("MODULE B — AQI FORECAST MODEL COMPARISON REPORT")
        print("="*65)
        print(f"{'Model':<20}  {'24h MAE':>8}  {'24h RMSE':>9}  "
              f"{'48h MAE':>8}  {'48h RMSE':>9}  "
              f"{'72h MAE':>8}  {'72h RMSE':>9}")
        print("-"*65)
        for model_name in ("LSTM", "GradientBoosting", "Prophet"):
            row = [f"{model_name:<20}"]
            for h in HORIZONS:
                m = self.results[model_name].get(h, {})
                mae  = f"{m['MAE']:.4f}"  if m.get("MAE")  is not None else "  N/A  "
                rmse = f"{m['RMSE']:.4f}" if m.get("RMSE") is not None else "  N/A  "
                row += [f"{mae:>8}", f"{rmse:>9}"]
            print("  ".join(row))
        print("-"*65)
        print("\nBest model per horizon:")
        for h in HORIZONS:
            name, _ = self.best_models.get(h, ("N/A", None))
            print(f"  {h}h  →  {name}")
