"""
CERIP — Heat Risk Feature Engineering (M2)
===========================================
Transforms raw grid data into model-ready feature matrices for Module A.

Feature groups produced
-----------------------
  1. Raw meteorological    (air_temp, humidity, wind, rainfall, pressure)
  2. Satellite indices     (lst, ndvi, ndbi, evi)
  3. Urban form            (building_density, road_density, green_frac, dist_water)
  4. Population            (pop_density, pct_elderly, pct_children, poverty)
  5. Interaction / ratios  (heat_load, green_deficiency, urban_heat_x_vulnerability)
  6. Spatial context       (lat, lon as positional features)

Usage
-----
    from src.features.heat_features import build_heat_feature_matrix
    X, y_reg, y_clf = build_heat_feature_matrix(df)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# Feature columns (raw) expected in the input DataFrame
RAW_FEATURE_COLS = [
    # Meteorological
    "air_temp_max_c", "humidity_pct", "wind_speed_ms",
    "rainfall_mm_annual", "pressure_hpa",
    # Satellite
    "ndvi", "ndbi", "evi",
    # Urban form
    "building_density", "road_density", "green_frac", "dist_to_water_km",
    # Population (Census 2011 — flagged outdated)
    "pop_density_per_km2", "pct_elderly", "pct_children", "pct_below_poverty",
    "green_space_access",
    # Derived / spatial
    "vulnerability_index", "heat_island_intensity",
    "lat", "lon",
]


def build_heat_feature_matrix(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """
    Build the final feature matrix X and both targets for Module A.

    Parameters
    ----------
    df          : Raw grid DataFrame (output of synthetic_generator or real loader)
    scaler      : Pre-fitted StandardScaler (pass when transforming test data)
    fit_scaler  : If True, fit a new scaler on df

    Returns
    -------
    X_df        : pd.DataFrame of engineered features (unscaled, for SHAP)
    X_scaled    : np.ndarray scaled version (for model input)
    y_reg       : np.ndarray — continuous LST (regression target)
    y_clf       : np.ndarray — integer class 1-5 (classification target)
    scaler      : Fitted StandardScaler
    feature_names : list[str]
    """
    df = df.copy()

    # ── Rename lat/lon from grid if present ─────────────────────
    if "centroid_lat" in df.columns and "lat" not in df.columns:
        df["lat"] = df["centroid_lat"]
        df["lon"] = df["centroid_lon"]

    # ── Ensure required columns exist ───────────────────────────
    missing = [c for c in RAW_FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning("Missing columns (will fill with median): %s", missing)
        for c in missing:
            df[c] = df.get(c, pd.Series(dtype=float)).fillna(0)

    # ── Engineered interaction features ─────────────────────────
    #
    # Heat load: combines temperature and humidity (heat index proxy)
    df["heat_load"] = (
        0.5 * df["air_temp_max_c"] + 0.5 * df["humidity_pct"] / 10
    )

    # Green deficiency: vegetation shortfall (negative NDVI / high NDBI)
    df["green_deficiency"] = (
        df["ndbi"] - df["ndvi"] + df["building_density"]
    ).clip(0, 2)

    # Urban heat × vulnerability compounding
    df["heat_x_vulnerability"] = (
        df["heat_island_intensity"].clip(0) * df["vulnerability_index"]
    )

    # Population exposure: pop density × vulnerability
    df["pop_exposure"] = (
        df["pop_density_per_km2"] / 10_000 * df["vulnerability_index"]
    )

    # Wind cooling effect (higher wind → lower effective heat)
    df["wind_cooling"] = df["wind_speed_ms"] * (-0.3)

    # Impervious surface (proxy from building + road density)
    df["impervious_surface"] = (
        df["building_density"] + df["road_density"]
    ).clip(0, 1)

    # ── Final feature columns ────────────────────────────────────
    feature_cols = RAW_FEATURE_COLS + [
        "heat_load",
        "green_deficiency",
        "heat_x_vulnerability",
        "pop_exposure",
        "wind_cooling",
        "impervious_surface",
    ]

    X_df = df[feature_cols].fillna(0).reset_index(drop=True)

    # ── Targets ──────────────────────────────────────────────────
    y_reg = df["lst_celsius"].values.astype(float)
    y_clf = df["heat_risk_class"].values.astype(int)

    # ── Scale ────────────────────────────────────────────────────
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X_scaled = scaler.fit_transform(X_df)
    else:
        X_scaled = scaler.transform(X_df)

    log.info("Feature matrix shape: %s | Features: %d", X_df.shape, len(feature_cols))
    return X_df, X_scaled, y_reg, y_clf, scaler, feature_cols


def get_feature_descriptions() -> dict[str, str]:
    """Human-readable descriptions for dashboard SHAP panel."""
    return {
        "air_temp_max_c":       "Max air temperature (°C)",
        "humidity_pct":         "Relative humidity (%)",
        "wind_speed_ms":        "Wind speed (m/s)",
        "rainfall_mm_annual":   "Annual rainfall (mm)",
        "pressure_hpa":         "Atmospheric pressure (hPa)",
        "ndvi":                 "NDVI — vegetation greenness",
        "ndbi":                 "NDBI — built-up index",
        "evi":                  "EVI — enhanced vegetation index",
        "building_density":     "Building footprint fraction",
        "road_density":         "Road network density fraction",
        "green_frac":           "Green space fraction",
        "dist_to_water_km":     "Distance to nearest water body (km)",
        "pop_density_per_km2":  "Population density (per km²) [Census 2011]",
        "pct_elderly":          "% elderly (65+) [Census 2011]",
        "pct_children":         "% children (<14) [Census 2011]",
        "pct_below_poverty":    "% below poverty line [Census 2011]",
        "green_space_access":   "Green space accessibility score",
        "vulnerability_index":  "Composite vulnerability index",
        "heat_island_intensity":"Urban heat island intensity (°C vs city mean)",
        "lat":                  "Latitude (positional)",
        "lon":                  "Longitude (positional)",
        "heat_load":            "Heat load index (temp × humidity proxy)",
        "green_deficiency":     "Green deficiency (NDBI - NDVI + building)",
        "heat_x_vulnerability": "Heat island × vulnerability (compounding)",
        "pop_exposure":         "Population exposure (density × vulnerability)",
        "wind_cooling":         "Wind cooling effect (negative = cooling)",
        "impervious_surface":   "Impervious surface fraction (building + road)",
    }
