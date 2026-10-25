# 🌡️💨 CERIP — Combined Environmental Risk Intelligence Platform
### Pilot City: Visakhapatnam (Vizag), India

> ⚠️ **DISCLAIMER**: This platform is for **informational and research purposes only** and does **not** constitute medical advice. Always consult qualified health professionals for health-related decisions.

> 🔬 **DATA NOTICE**: This prototype uses **synthetic data** generated with realistic statistical properties for Visakhapatnam. Real-data loaders are stubbed for IMD, ISRO, Landsat, Sentinel-2, CPCB, and Census sources. See `data/synthetic/SYNTHETIC_NOTICE.md`.

---

## Overview

CERIP integrates two AI/ML risk engines into a single **Unified Outdoor Safety Index (UOSI)**:

| Module | Engine | Output |
|--------|--------|--------|
| **A** | Urban Heat Risk Intelligence | Heat risk per 500 m x 500 m grid cell |
| **B** | Air Quality Prediction & Activity Advisor | 72-hour AQI/PM2.5 forecast + activity recommendations |
| **Unified** | UOSI Fusion | Safe / Caution / Limit / Avoid |

---

## Architecture

```
Layer 1 - Data Sources
  IMD / ISRO/Bhuvan / Landsat/USGS / Sentinel-2 / CPCB
  Census 2011 (outdated - flagged) / OpenAQ

Layer 2 - Ingestion & Feature Layer
  src/data/ -> src/features/
  500m grid engine / Spatial-temporal alignment
  Time-series feature store (lag, rolling, calendar)

Layer 3 - Modeling Layer
  Module A: Random Forest + XGBoost/LightGBM + SHAP
  Module B: LSTM + Prophet + Gradient Boosting
  Fusion: Unified Index (configurable weights)

Layer 4 - Serving Layer
  FastAPI REST <-> Streamlit Dashboard
  /heat-risk / /aqi-forecast
  /activity-recommendation / /unified-index
```

---

## Quick Start

### 1. Install dependencies
```bash
cd c:\FILE\DE_ML\CERIP
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
```bash
python src/data/synthetic_generator.py
```

### 3. Train Models
```bash
python notebooks/02_heat_risk_training.py
python notebooks/03_aqi_forecast_training.py
```

### 4. Run FastAPI
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 5. Run Dashboard
```bash
streamlit run dashboard/app.py
```

---

## Unified Outdoor Safety Index Formula

```
UOSI_raw = w_heat * HeatScore + w_aqi * AQIScore
if HeatScore >= 3 AND AQIScore >= 3:
    UOSI_raw += compound_bonus  # default 0.5
UOSI: Safe(<1.5) | Caution(1.5-2.5) | Limit(2.5-3.5) | Avoid(>=3.5)
```

Defaults: w_heat=0.5, w_aqi=0.5, compound_bonus=0.5 (all in config/settings.yaml)

---

## Key Limitations

- **Census 2011**: Population/vulnerability data are ~15 years old for a fast-growing city.
- **Synthetic Data**: All data in this prototype is synthetically generated.
- **LSTM**: Trained on synthetic time-series; real-world accuracy requires real AQI data.
- **Grid Resolution**: 500m cells are adequate for city-planning, not street-level decisions.

---

*Built as a research prototype. Not for operational or medical use.*
