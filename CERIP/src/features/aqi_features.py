"""
CERIP — AQI Feature Engineering (M2)
======================================
Transforms the hourly AQI time-series into a supervised learning dataset
suitable for LSTM, Prophet, and Gradient Boosting models.

Key transformations
-------------------
  1. Lag features        (AQI/PM2.5 at t-1, t-3, t-6, t-12, t-24 hours)
  2. Rolling statistics  (mean, std, min, max over 3, 6, 12, 24-hour windows)
  3. Calendar features   (hour-of-day, day-of-week, month, season, is_weekend)
  4. Weather features    (temperature, humidity, wind, pressure, rainfall)
  5. Meteorological interactions (temp × humidity, wind × aqi_lag)
  6. Target windows      (AQI at t+24, t+48, t+72 hours for multi-horizon)

Usage
-----
    from src.features.aqi_features import build_aqi_feature_matrix
    df_feat = build_aqi_feature_matrix(aqi_df)
    X, y24, y48, y72, feature_cols = split_targets(df_feat)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# Lag steps in hours
LAG_STEPS = [1, 3, 6, 12, 24, 48]
# Rolling window sizes in hours
ROLL_WINDOWS = [3, 6, 12, 24]
# Target horizons in hours
HORIZONS = [24, 48, 72]


def add_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    """Add lag features for specified columns and lag steps."""
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, cols: list[str], windows: list[int]
) -> pd.DataFrame:
    """Add rolling mean, std, min, max for specified columns."""
    for col in cols:
        for w in windows:
            df[f"{col}_roll{w}h_mean"] = df[col].shift(1).rolling(w).mean()
            df[f"{col}_roll{w}h_std"]  = df[col].shift(1).rolling(w).std()
            df[f"{col}_roll{w}h_max"]  = df[col].shift(1).rolling(w).max()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical encodings for time features (sin/cos to avoid discontinuity)."""
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # Season: 1=Winter, 2=Pre-monsoon, 3=Monsoon, 4=Post-monsoon
    df["season"] = pd.cut(
        df["month"],
        bins=[0, 2, 5, 9, 11, 12],
        labels=[1, 2, 3, 4, 1],
        ordered=False,
    ).astype(int)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add meteorological interaction terms."""
    # Heat-humidity index (apparent temperature proxy)
    df["heat_humidity_idx"] = df["temperature_c"] * df["humidity_pct"] / 100

    # Wind dispersion: higher wind disperses pollutants
    aqi_col = "aqi_lag1h" if "aqi_lag1h" in df.columns else "aqi"
    df["wind_dispersion"] = df["wind_speed_ms"] / (df[aqi_col].clip(1) + 1)

    # Inversion index: low wind + high humidity favours pollution buildup
    df["inversion_index"] = (df["humidity_pct"] / 100) * (1 / (df["wind_speed_ms"] + 0.1))

    # Rainfall washout: rain reduces AQI
    df["rain_washout"] = df["rainfall_mm"].clip(0, 50) / 50  # normalised 0-1

    return df


def add_targets(df: pd.DataFrame, horizons: list[int] = None) -> pd.DataFrame:
    """Add future AQI values as targets (shifted forward)."""
    if horizons is None:
        horizons = HORIZONS
    for h in horizons:
        df[f"target_aqi_{h}h"] = df["aqi"].shift(-h)
        df[f"target_pm25_{h}h"] = df["pm25_ugm3"].shift(-h)
    return df


def build_aqi_feature_matrix(
    aqi_df: pd.DataFrame,
    lag_steps: list[int] = None,
    roll_windows: list[int] = None,
    horizons: list[int] = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline for AQI time-series.

    Parameters
    ----------
    aqi_df      : Raw AQI DataFrame from synthetic_generator or real loader

    Returns
    -------
    DataFrame with all lag, rolling, calendar, interaction, and target columns.
    Rows with NaN (from lags/targets at boundaries) are dropped.
    """
    lag_steps   = lag_steps   or LAG_STEPS
    roll_windows = roll_windows or ROLL_WINDOWS
    horizons    = horizons    or HORIZONS

    df = aqi_df.copy().sort_values("timestamp").reset_index(drop=True)

    log.info("Building AQI feature matrix. Input rows: %d", len(df))

    # ── Lag features ─────────────────────────────────────────────
    df = add_lag_features(df, cols=["aqi", "pm25_ugm3", "temperature_c",
                                    "humidity_pct", "wind_speed_ms"],
                          lags=lag_steps)

    # ── Rolling features ─────────────────────────────────────────
    df = add_rolling_features(df, cols=["aqi", "pm25_ugm3"],
                              windows=roll_windows)

    # ── Calendar features ─────────────────────────────────────────
    df = add_calendar_features(df)

    # ── Interaction features ──────────────────────────────────────
    df = add_interaction_features(df)

    # ── Target columns ────────────────────────────────────────────
    df = add_targets(df, horizons=horizons)

    # ── Drop NaN boundary rows ────────────────────────────────────
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    log.info("Feature matrix ready: %d rows (dropped %d boundary rows)",
             len(df), before - len(df))

    return df


def split_targets(
    df: pd.DataFrame,
    horizons: list[int] = None,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, dict[int, np.ndarray], StandardScaler, list[str]]:
    """
    Split the feature matrix into X and per-horizon targets.

    Returns
    -------
    X_df          : unscaled feature DataFrame (for SHAP)
    X_scaled      : np.ndarray scaled (for model input)
    targets       : dict {24: y_24h, 48: y_48h, 72: y_72h}
    scaler        : fitted StandardScaler
    feature_cols  : list[str]
    """
    if horizons is None:
        horizons = HORIZONS

    # Identify feature columns (exclude raw AQI, meta, and target cols)
    exclude_prefixes = ("target_", "timestamp", "data_source", "aqi_category",
                        "aqi", "pm25_ugm3", "hour", "day_of_week", "month",
                        "is_monsoon", "is_weekend", "season")
    feature_cols = [
        c for c in df.columns
        if not any(c.startswith(p) or c == p for p in exclude_prefixes)
    ]

    # Add back lag features explicitly (they start with aqi/pm25 but are lags)
    lag_and_roll_cols = [c for c in df.columns
                         if any(c.startswith(p) and ("_lag" in c or "_roll" in c)
                                for p in ("aqi", "pm25"))]
    # Deduplicate and preserve order
    all_feat_cols = lag_and_roll_cols + [c for c in feature_cols
                                          if c not in lag_and_roll_cols]

    X_df = df[all_feat_cols].copy()

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X_scaled = scaler.fit_transform(X_df)
    else:
        X_scaled = scaler.transform(X_df)

    targets = {h: df[f"target_aqi_{h}h"].values for h in horizons}

    return X_df, X_scaled, targets, scaler, all_feat_cols


def make_lstm_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape flat feature array into LSTM sequences.

    Parameters
    ----------
    X       : shape (n_samples, n_features)
    y       : shape (n_samples,)
    seq_len : look-back window in hours

    Returns
    -------
    X_seq   : shape (n_samples - seq_len, seq_len, n_features)
    y_seq   : shape (n_samples - seq_len,)
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i, :])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)
