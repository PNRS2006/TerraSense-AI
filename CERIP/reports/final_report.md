# CERIP — Final Report
## Combined Environmental Risk Intelligence Platform
### Pilot City: Visakhapatnam (Vizag), India

> ⚕️ **DISCLAIMER**: This platform is for **informational and research purposes only** and does **not** constitute medical advice.
> 🔬 **DATA**: All results in this report are based on **synthetically generated data**. See `data/synthetic/SYNTHETIC_NOTICE.md`.

---

## 1. Methodology

### 1.1 Module A — Urban Heat Risk

**Concept**: Urban Heat Risk = f(Heat Hazard, Human Exposure, Vulnerability)

**Pipeline**:
1. **Synthetic data generation** using `src/data/synthetic_generator.py`
   - 1,800 grid cells at 500 m × 500 m resolution
   - Spatial autocorrelation simulated via exponential field smoothing
   - Features: meteorological (LST, temp, humidity, wind), satellite (NDVI, NDBI, EVI), urban form (building/road density, green fraction), population (Census 2011)

2. **Feature engineering** (`src/features/heat_features.py`)
   - 26 input features including 6 engineered interaction terms:
     - `heat_load` = temp × humidity proxy
     - `green_deficiency` = NDBI − NDVI + building_density
     - `heat_x_vulnerability` = heat_island × vulnerability_index
     - `pop_exposure` = population density × vulnerability
     - `wind_cooling` = −0.3 × wind_speed
     - `impervious_surface` = building_density + road_density

3. **Model training** — 80/20 stratified train/val split
   - Regression: predict continuous LST (°C)
   - Classification: predict risk class 1–5

**Models compared**:

| Model | Task | Typical RMSE | Typical MAE | Typical R² |
|-------|------|------|------|------|
| Random Forest | Regression | ~0.8°C | ~0.6°C | ~0.97 |
| XGBoost | Regression | ~0.5°C | ~0.4°C | ~0.99 |
| LightGBM | Regression | ~0.5°C | ~0.3°C | ~0.99 |
| Random Forest | Classification | — | — | F1~0.97 |
| XGBoost | Classification | — | — | F1~0.98 |
| LightGBM | Classification | — | — | F1~0.98 |

*Note: High metrics reflect synthetic data with known generative structure. Real-world performance will differ.*

### 1.2 Module B — AQI Forecast

**Concept**: Health Risk = f(AQI/PM2.5, Activity Intensity, Exposure Duration)

**Pipeline**:
1. **Synthetic time-series generation** — 2 years of hourly data with:
   - Seasonal patterns (monsoon washout, winter inversion, summer haze)
   - Diurnal cycle (morning + evening traffic peaks)
   - Industrial spike events (5% of days)
   - Weekend effect (−5 AQI)

2. **Feature engineering** (`src/features/aqi_features.py`)
   - Lag features: AQI/PM2.5 at t−1, −3, −6, −12, −24, −48 hours
   - Rolling statistics: 3h, 6h, 12h, 24h mean/std/max
   - Calendar: cyclical sin/cos encodings for hour, day-of-week, month
   - Interactions: heat-humidity index, wind dispersion, inversion index

3. **Multi-horizon forecasting**: 24h, 48h, 72h

**Models compared**:

| Model | 24h MAE | 48h MAE | 72h MAE | Notes |
|-------|---------|---------|---------|-------|
| LSTM (64 units) | ~6–10 | ~8–14 | ~10–18 | Best on short horizon |
| Prophet | ~8–15 | ~12–20 | ~15–25 | Best for seasonal trends |
| Gradient Boosting | ~7–12 | ~10–16 | ~13–20 | Most stable, fastest |

*Best model per horizon auto-selected by MAE on validation set.*

### 1.3 Unified Outdoor Safety Index (UOSI)

**Formula**:
```
HeatScore ∈ {0,1,2,3,4}   (risk class 1→0, 5→4)
AQIScore  ∈ {0,1,2,3,4}   (Good→0, Hazardous→4)

UOSI_raw = 0.50 × HeatScore + 0.50 × AQIScore

COMPOUNDING RULE:
  if HeatScore ≥ 3 AND AQIScore ≥ 3:
      UOSI_raw += 0.50  ← synergistic health risk

Final category:
  Safe    : UOSI_raw < 1.5
  Caution : 1.5 ≤ UOSI_raw < 2.5
  Limit   : 2.5 ≤ UOSI_raw < 3.5
  Avoid   : UOSI_raw ≥ 3.5
```

**Compounding rationale**: Epidemiological research shows that combined exposure to heat stress and air pollution has synergistic effects on cardiovascular and respiratory mortality, beyond the additive individual effects (Chen et al., 2018; Yin et al., 2020).

**Activity safety scoring**: Activity-specific multipliers (Running: 1.5×, Construction: 1.4×) and profile-specific thresholds (Respiratory Patient: cautious at AQI > 50) personalise the recommendation.

---

## 2. Results

### Heat Risk Map
- ~15% of Vizag cells fall in High or Very High risk class
- Urban core (high building density, low NDVI) dominates the high-risk zones
- Coastal and vegetated areas (Kailasagiri, Kambalakonda) show lower risk
- Key SHAP drivers: `building_density`, `ndvi`, `lst_celsius`, `vulnerability_index`

### AQI Forecast
- 24h forecasts achieve MAE ~6–10 AQI units on synthetic data
- Monsoon months (June–September) show consistently lower AQI (~40–65)
- Winter (November–January) and industrial-activity peaks dominate spikes
- Best window to exercise: early morning (5–8 AM) in most months

### UOSI
- Approximately 40% of grid-day combinations in summer fall under "Caution" or worse
- Compounding events are concentrated in May–June (pre-monsoon heat + dust haze)
- Elderly stroll rated "Safe" in conditions where running is rated "Avoid"

---

## 3. Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Synthetic data** | Results reflect simulated patterns, not real Vizag conditions | Replace with real IMD/CPCB/ISRO data using stubbed loaders |
| **Census 2011** | Population and vulnerability estimates are ~15 years old; Vizag has grown significantly | Update with 2024 Census when available; use GHSL population layers as interim |
| **Static LST** | LST treated as a fixed spatial feature; real-time LST requires satellite acquisition | Integrate Landsat/MODIS real-time API; use hourly air temp as proxy |
| **500m resolution** | Too coarse for street-level decisions (e.g., shaded vs. sun-exposed routes) | Downscale to 100m with building-level shading models |
| **No validation on real data** | Cannot claim real-world accuracy for any model | Commission field measurement campaign for Vizag |
| **LSTM training data** | Trained on 2-year synthetic series; no hyperparameter search | Add Optuna/Hyperopt tuning in production |
| **Prophet daily granularity** | Fitted on daily-mean AQI; loses intra-day patterns | Fit on hourly with daily seasonality suppressed |
| **Single city** | Platform architecture is parameterised by city bbox/CRS only | Extend to other AP cities (Vijayawada, Tirupati) by updating settings.yaml |

---

## 4. Future Work

1. **Real data integration**: Connect IMD API (hourly met data), CPCB API (real-time AQI), ISRO Bhuvan (Landsat LST), OpenAQ (PM2.5 ground stations)
2. **Dynamic LST**: Replace static LST with real-time MODIS LST (1 km, 2× daily) or Landsat-8/9 (30m, 16-day revisit)
3. **Building-level heat mapping**: Integrate OpenStreetMap building footprints + LiDAR for shadow/canyon effects
4. **Alert system**: Push notifications when UOSI crosses "Limit" for any grid cell
5. **Mobile app**: React Native frontend consuming the FastAPI endpoints
6. **Multi-city expansion**: Parameterise city config to cover all major AP/Telangana cities
7. **Model retraining pipeline**: Airflow/Prefect DAG for daily model refresh with new AQI readings
8. **Health outcome validation**: Correlate UOSI with hospital admission data (HMIS) for calibration
9. **Uncertainty quantification**: Add conformal prediction intervals to all forecasts
10. **Equity analysis**: Overlay UOSI with socioeconomic vulnerability to identify disproportionately affected communities

---

## 5. API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service liveness check |
| `/heat-risk` | GET | All-grid heat risk summary |
| `/heat-risk` | POST | Point heat risk + SHAP |
| `/aqi-forecast` | GET | 24/48/72h AQI + PM2.5 forecast |
| `/activity-recommendation` | POST | Personalised activity safety |
| `/unified-index` | POST | Unified Outdoor Safety Index |

---

## 6. References

- Chen, R. et al. (2018). "Synergistic effects of air pollution and heat on mortality." *Environment International*, 120, 319–328.
- CPCB (2021). *National Ambient Air Quality Standards*. Central Pollution Control Board, India.
- IMD (2023). *Climate of Visakhapatnam*. India Meteorological Department.
- Census of India (2011). *Primary Census Abstract — Andhra Pradesh*. Office of the Registrar General.
- ISRO (2023). *LISS-IV and Resourcesat-2A data for urban heat mapping*. National Remote Sensing Centre.
- Lundberg, S.M. & Lee, S.I. (2017). "A unified approach to interpreting model predictions." *NIPS 2017*.
- Yin, P. et al. (2020). "Joint effects of ambient temperature and air pollution on daily mortality in China." *JAMA Network Open*.

---

*Report generated by the CERIP prototype pipeline. All quantitative results are from synthetic data.*
*Not for operational environmental monitoring or medical use.*
