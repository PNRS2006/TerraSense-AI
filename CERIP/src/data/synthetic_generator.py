"""
CERIP — Synthetic Data Generator (M1)
======================================
Generates realistic synthetic datasets for Visakhapatnam, India:
  1. vizag_grid_synthetic.csv  — 500m x 500m grid cells with spatial features
  2. vizag_aqi_synthetic.csv   — 2-year hourly AQI + PM2.5 time-series

All values are statistically grounded in Vizag's known climate and
urban characteristics. See data/synthetic/SYNTHETIC_NOTICE.md.

Run:
    python src/data/synthetic_generator.py

Outputs (relative to project root):
    data/synthetic/vizag_grid_synthetic.csv
    data/synthetic/vizag_aqi_synthetic.csv
    data/processed/grid_features.parquet
    data/processed/aqi_timeseries.parquet
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ── project root (two levels up from this file) ──────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── load configuration ────────────────────────────────────────
with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SEED     = CFG["synthetic"]["random_seed"]
N_CELLS  = CFG["synthetic"]["n_grid_cells"]
AQI_DAYS = CFG["synthetic"]["aqi_timeseries_days"]

rng = np.random.default_rng(SEED)


# =============================================================
# PART 1 — Spatial Grid Features (Module A)
# =============================================================

def _spatial_covariance_field(n: int, scale: float = 0.3) -> np.ndarray:
    """
    Generate a spatially auto-correlated random field via distance-based
    smoothing.  `scale` controls the bandwidth of the Gaussian kernel.
    """
    raw = rng.standard_normal(n)
    # Simple exponential smoothing on sorted indices as a proxy
    alpha = 1 - np.exp(-scale)
    smoothed = np.zeros(n)
    smoothed[0] = raw[0]
    for i in range(1, n):
        smoothed[i] = alpha * raw[i] + (1 - alpha) * smoothed[i - 1]
    return (smoothed - smoothed.mean()) / (smoothed.std() + 1e-9)


def generate_grid_features(n: int = N_CELLS, seed: int = SEED) -> pd.DataFrame:
    """
    Generate one row per 500 m × 500 m grid cell with realistic
    spatial features for Visakhapatnam.

    Feature groups
    --------------
    Meteorological  : air_temp_max, humidity_pct, wind_speed_ms
    Satellite       : lst_celsius (land surface temp), ndvi, ndbi
    Urban form      : building_density, road_density, green_frac,
                      dist_to_water_km
    Population      : pop_density, pct_elderly, pct_children,
                      pct_below_poverty
    Derived         : heat_island_intensity, vulnerability_index
    Target (reg)    : lst_celsius (also used as regression target)
    Target (clf)    : heat_risk_class  (1=Very Low … 5=Very High)
    """
    log.info("Generating %d grid cells …", n)
    np.random.seed(seed)

    # ── Grid IDs & approximate coordinates (within Vizag bbox) ──
    lons = rng.uniform(83.15, 83.45, n)
    lats = rng.uniform(17.60, 17.85, n)

    # ── Spatially correlated base fields ─────────────────────────
    urb_field  = _spatial_covariance_field(n, scale=0.4)  # urbanisation proxy
    veg_field  = -urb_field + 0.3 * _spatial_covariance_field(n, scale=0.2)
    coast_field = -np.abs(lons - 83.30) * 5               # nearness to coast

    # ── Meteorological features ───────────────────────────────────
    # Vizag mean summer max ~36°C, winter ~28°C (using neutral month)
    air_temp_max = (32 + 3 * urb_field
                    + 0.5 * rng.standard_normal(n)).clip(24, 45)
    humidity_pct = (72 - 8 * urb_field + 5 * coast_field
                    + 3 * rng.standard_normal(n)).clip(30, 98)
    wind_speed_ms = (4.5 - 1.5 * urb_field
                     + rng.standard_normal(n)).clip(0.5, 15)
    rainfall_mm  = (850 + 300 * veg_field
                    + 50 * rng.standard_normal(n)).clip(200, 2000)
    pressure_hpa = (1010 + 2 * coast_field
                    + 0.5 * rng.standard_normal(n)).clip(1000, 1020)

    # ── Satellite-derived features ────────────────────────────────
    # LST: urban areas hotter, vegetated areas cooler, coastal cooler
    lst_celsius = (38 + 5 * urb_field - 3 * veg_field - 2 * coast_field
                   + rng.standard_normal(n)).clip(25, 52)

    # NDVI: 0=bare/built, 1=dense vegetation; inverse of urbanisation
    ndvi = (0.45 - 0.3 * urb_field + 0.2 * veg_field
            + 0.05 * rng.standard_normal(n)).clip(0, 1)

    # NDBI: built-up index; positive = built-up, negative = not
    ndbi = (0.1 + 0.35 * urb_field - 0.2 * veg_field
            + 0.05 * rng.standard_normal(n)).clip(-0.5, 0.7)

    # EVI (Enhanced Vegetation Index, similar range to NDVI but better in urban)
    evi  = (ndvi * 0.85 + 0.05 * rng.standard_normal(n)).clip(0, 1)

    # ── Urban form features ───────────────────────────────────────
    building_density = (0.35 + 0.35 * urb_field
                        + 0.05 * rng.standard_normal(n)).clip(0, 0.95)
    road_density     = (0.25 + 0.25 * urb_field
                        + 0.03 * rng.standard_normal(n)).clip(0, 0.8)
    green_frac       = (1 - building_density - road_density
                        + 0.05 * rng.standard_normal(n)).clip(0.01, 0.99)
    dist_to_water_km = np.clip(np.abs(3.0 - 2.5 * coast_field
                        + rng.standard_normal(n)), 0.05, 20)

    # ── Population features (Census 2011 — labelled outdated) ─────
    pop_density          = (8000 + 5000 * urb_field
                            + 500 * rng.standard_normal(n)).clip(100, 30000)
    pct_elderly          = (7 + 2 * rng.standard_normal(n)).clip(2, 20)
    pct_children         = (28 + 4 * rng.standard_normal(n)).clip(10, 50)
    pct_below_poverty    = (18 - 8 * urb_field
                            + 3 * rng.standard_normal(n)).clip(2, 60)
    green_space_access   = (1 - 0.5 * urb_field
                            + 0.1 * rng.standard_normal(n)).clip(0, 1)

    # ── Derived features ──────────────────────────────────────────
    # Urban Heat Island intensity = LST difference vs city-wide mean
    heat_island_intensity = lst_celsius - lst_celsius.mean()

    # Vulnerability index (0-1): higher = more vulnerable
    # Combines poverty, elderly, children, low green access
    vulnerability_index = (
        0.35 * (pct_below_poverty / 60)
        + 0.25 * (pct_elderly / 20)
        + 0.20 * (pct_children / 50)
        + 0.20 * (1 - green_space_access)
    ).clip(0, 1)

    # ── Classification target (heat risk class 1-5) ───────────────
    thresholds = CFG["heat_risk"]["lst_thresholds"]  # [30, 34, 37, 40]
    heat_risk_class = np.digitize(lst_celsius, bins=thresholds) + 1  # 1..5

    # ── Assemble DataFrame ─────────────────────────────────────────
    df = pd.DataFrame({
        # IDs
        "cell_id":              np.arange(n),
        "lon":                  np.round(lons, 5),
        "lat":                  np.round(lats, 5),
        # Meteorological
        "air_temp_max_c":       np.round(air_temp_max, 2),
        "humidity_pct":         np.round(humidity_pct, 1),
        "wind_speed_ms":        np.round(wind_speed_ms, 2),
        "rainfall_mm_annual":   np.round(rainfall_mm, 1),
        "pressure_hpa":         np.round(pressure_hpa, 1),
        # Satellite
        "lst_celsius":          np.round(lst_celsius, 2),   # REGRESSION TARGET
        "ndvi":                 np.round(ndvi, 4),
        "ndbi":                 np.round(ndbi, 4),
        "evi":                  np.round(evi, 4),
        # Urban form
        "building_density":     np.round(building_density, 3),
        "road_density":         np.round(road_density, 3),
        "green_frac":           np.round(green_frac, 3),
        "dist_to_water_km":     np.round(dist_to_water_km, 3),
        # Population (Census 2011 — outdated)
        "pop_density_per_km2":  np.round(pop_density, 0).astype(int),
        "pct_elderly":          np.round(pct_elderly, 1),
        "pct_children":         np.round(pct_children, 1),
        "pct_below_poverty":    np.round(pct_below_poverty, 1),
        "green_space_access":   np.round(green_space_access, 3),
        # Derived
        "heat_island_intensity": np.round(heat_island_intensity, 2),
        "vulnerability_index":   np.round(vulnerability_index, 4),
        # Targets
        "heat_risk_class":       heat_risk_class.astype(int),
        "heat_risk_label":       [CFG["heat_risk"]["risk_labels"][c]
                                  for c in heat_risk_class],
        # Metadata
        "data_source":          "SYNTHETIC — NOT REAL DATA",
        "census_year_note":     "Population data from Census 2011 (outdated)",
    })

    log.info("Grid features generated. Class distribution:\n%s",
             df["heat_risk_label"].value_counts().to_string())
    return df


# =============================================================
# PART 2 — AQI Time-Series (Module B)
# =============================================================

def generate_aqi_timeseries(days: int = AQI_DAYS, seed: int = SEED) -> pd.DataFrame:
    """
    Generate hourly AQI + PM2.5 time-series for Vizag (2 years).

    Statistical properties based on Vizag's known air quality:
    - Winter months (Nov-Jan): higher AQI (~80-120) due to inversion + fog
    - Summer months (Apr-Jun): high AQI (~70-100) + heat
    - Monsoon months (Jul-Sep): lower AQI (~40-60) due to rain washout
    - Daily pattern: morning peak (07-09), evening peak (18-21)
    - Industrial areas: PM2.5 elevated due to steel plants & port
    - Typical AQI range: 40–180 (occasional spikes >200)
    """
    log.info("Generating %d days of hourly AQI time-series …", days)
    np.random.seed(seed + 1)

    start_dt = datetime(2023, 1, 1, 0, 0)
    n_hours  = days * 24
    timestamps = [start_dt + timedelta(hours=h) for h in range(n_hours)]

    hours   = np.array([t.hour       for t in timestamps])
    months  = np.array([t.month      for t in timestamps])
    dow     = np.array([t.weekday()  for t in timestamps])  # 0=Mon

    # ── Seasonal component ────────────────────────────────────────
    # Vizag AQI peaks in winter/summer, dips in monsoon
    seasonal = (
        20 * np.sin(2 * np.pi * (months - 3) / 12)    # summer peak ~Jun
        + 15 * np.cos(2 * np.pi * (months - 10) / 12) # winter peak ~Jan
    )

    # ── Diurnal component (morning + evening traffic peaks) ───────
    diurnal = (
        15 * np.exp(-0.5 * ((hours - 8.0) / 1.5) ** 2)   # morning peak 08:00
        + 12 * np.exp(-0.5 * ((hours - 19.0) / 1.5) ** 2) # evening peak 19:00
        - 8  * (hours < 5).astype(float)                   # nighttime lower
    )

    # ── Weekend effect (slightly lower AQI on weekends) ───────────
    weekend = -5 * (dow >= 5).astype(float)

    # ── Monsoon washout effect ────────────────────────────────────
    monsoon_mask = (months >= 6) & (months <= 9)
    monsoon = -20 * monsoon_mask.astype(float)

    # ── Industrial spike model (occasional bad days) ──────────────
    spike_days = rng.choice(days, size=int(days * 0.05), replace=False)
    spike_hours = np.zeros(n_hours)
    for d in spike_days:
        start_h = d * 24 + rng.integers(6, 18)
        duration = rng.integers(3, 12)
        for h in range(start_h, min(start_h + duration, n_hours)):
            spike_hours[h] = rng.uniform(40, 120)

    # ── Long-term trend (slight improvement with policy) ──────────
    trend = -0.005 * np.arange(n_hours)  # ~-1 AQI per ~200 hours

    # ── Base AQI noise ────────────────────────────────────────────
    noise_aqi = 8 * rng.standard_normal(n_hours)

    # ── Compose AQI ───────────────────────────────────────────────
    aqi = (75 + seasonal + diurnal + weekend + monsoon
           + spike_hours + trend + noise_aqi)
    aqi = np.clip(aqi, 10, 400).round(1)

    # ── PM2.5 (correlated with AQI, separate noise) ───────────────
    pm25_factor = 0.6 + 0.1 * rng.standard_normal(n_hours)
    pm25 = (aqi * pm25_factor + 5 * rng.standard_normal(n_hours)).clip(5, 300).round(1)

    # ── Weather covariates ────────────────────────────────────────
    temperature = (
        28 + 6 * np.sin(2 * np.pi * (months - 4) / 12)
        - 3 * np.cos(2 * np.pi * hours / 24)
        + 2 * rng.standard_normal(n_hours)
    ).clip(18, 46).round(1)

    humidity = (
        70 - 0.5 * (temperature - 28) + 10 * monsoon_mask
        + 5 * rng.standard_normal(n_hours)
    ).clip(25, 99).round(1)

    wind_speed = (
        4.5 + 1.5 * np.sin(2 * np.pi * hours / 24)
        - 1.0 * monsoon_mask
        + rng.standard_normal(n_hours)
    ).clip(0.2, 20).round(2)

    pressure = (
        1010 + 2 * np.cos(2 * np.pi * months / 12)
        + 0.5 * rng.standard_normal(n_hours)
    ).clip(995, 1025).round(1)

    rainfall_mm = np.zeros(n_hours)
    rain_idx = rng.choice(
        np.where(monsoon_mask)[0], size=int(np.sum(monsoon_mask) * 0.15), replace=False
    )
    rainfall_mm[rain_idx] = rng.exponential(8, size=len(rain_idx)).clip(0, 150)

    # ── AQI category ─────────────────────────────────────────────
    def aqi_to_cat(a):
        if a <= 50:   return "Good"
        if a <= 100:  return "Moderate"
        if a <= 150:  return "Unhealthy-Sensitive"
        if a <= 200:  return "Unhealthy"
        if a <= 300:  return "Very Unhealthy"
        return "Hazardous"

    df = pd.DataFrame({
        "timestamp":        timestamps,
        "hour":             hours,
        "day_of_week":      dow,
        "month":            months,
        "aqi":              aqi,
        "pm25_ugm3":        pm25,
        "aqi_category":     [aqi_to_cat(a) for a in aqi],
        "temperature_c":    temperature,
        "humidity_pct":     humidity,
        "wind_speed_ms":    wind_speed,
        "pressure_hpa":     pressure,
        "rainfall_mm":      rainfall_mm.round(1),
        "is_monsoon":       monsoon_mask.astype(int),
        "is_weekend":       (dow >= 5).astype(int),
        "data_source":      "SYNTHETIC — NOT REAL DATA",
    })

    log.info("AQI time-series generated. AQI stats:\n%s",
             df["aqi"].describe().round(1).to_string())
    log.info("AQI category distribution:\n%s",
             df["aqi_category"].value_counts().to_string())
    return df


# =============================================================
# MAIN — Save to disk
# =============================================================

def main():
    out_syn = ROOT / "data" / "synthetic"
    out_proc = ROOT / "data" / "processed"
    out_syn.mkdir(parents=True, exist_ok=True)
    out_proc.mkdir(parents=True, exist_ok=True)

    # --- Grid ---
    grid_df = generate_grid_features()
    grid_csv = out_syn / "vizag_grid_synthetic.csv"
    grid_parquet = out_proc / "grid_features.parquet"
    grid_df.to_csv(grid_csv, index=False)
    grid_df.to_parquet(grid_parquet, index=False)
    log.info("Saved grid: %s", grid_csv)
    log.info("Saved grid: %s", grid_parquet)

    # --- AQI ---
    aqi_df = generate_aqi_timeseries()
    aqi_csv = out_syn / "vizag_aqi_synthetic.csv"
    aqi_parquet = out_proc / "aqi_timeseries.parquet"
    aqi_df.to_csv(aqi_csv, index=False)
    aqi_df.to_parquet(aqi_parquet, index=False)
    log.info("Saved AQI: %s", aqi_csv)
    log.info("Saved AQI: %s", aqi_parquet)

    print("\n✅  Synthetic data generation complete.")
    print(f"    Grid cells : {len(grid_df):,}")
    print(f"    AQI rows   : {len(aqi_df):,}  ({AQI_DAYS} days × 24 hours)")
    print(f"    Outputs    : {out_syn}  &  {out_proc}")


if __name__ == "__main__":
    main()
