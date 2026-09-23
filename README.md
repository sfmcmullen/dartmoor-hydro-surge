# Dartmoor Hydro-Surge 🌊

A full-stack, real-time hydrologic forecasting dashboard predicting river stage changes for the River Dart at Austin's Bridge, Dartmoor. 

This project integrates real-time environmental telemetry from the UK Environment Agency API with custom XGBoost regression models to predict river level surges up to 3 hours in advance.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-000000?style=for-the-badge&logo=vercel)](https://dartmoor-hydro-surge.vercel.app/)

---

## 🛠️ System Architecture & Tech Stack

```
[ EA Real-Time Telemetry API ] 
              │
              ▼
    [ Python / FastAPI ]  ──────►  [ Feature Engineering Pipeline ]
              │                               │
              │                               ▼
              │                     [ XGBoost Regressors ]
              │                      (1h & 3h Δ Models)
              │                               │
              ▼                               ▼
    [ REST Endpoint /api/predict ] ◄──────────┘
              │
              ▼
  [ React + TypeScript Frontend ] ──► [ Interactive Hydrograph ]
```

### Backend
* **FastAPI:** Asynchronous Python API serving inference endpoints and handling telemetry ingestion.
* **XGBoost:** Gradient boosting models trained to predict relative stage changes ($\Delta\text{stage}$) using stationary lag and rolling temporal features.
* **Pandas & NumPy:** Data cleaning, interpolation, and feature engineering (cyclical time encoding, rolling stats).
* **Docker & Uvicorn:** Containerized server environment for consistent local and production execution.

### Frontend
* **React + TypeScript:** Strongly-typed component interface built with Vite.
* **Recharts:** Responsive SVG rendering for the live hydrograph trajectory.
* **Lucide React:** Minimalist iconography for directional metrics and system alerts.
* **Pure CSS:** Custom styling with CSS variables and flex/grid layouts without utility framework overhead.

---

## 🔬 Core Machine Learning Workflow

1. **Target Stationarity:** Rather than predicting raw river height directly, as it shifts seasonally, the models predict relative elevation change ($\Delta\text{stage} = \text{stage}_{t+k} - \text{stage}_t$).
2. **Feature Engineering:**
   * **Hydrologic Context:** 7-day rolling mean stage to establish seasonal baseline context (wet winter vs. dry summer).
   * **Precipitation Aggregations:** Accumulated rainfall windows (1h, 3h, 6h, 12h, 24h, 48h) to capture catchment lag.
   * **Cyclical Temporal Encoding:** Sine/cosine transformations of month and day of year to handle seasonal inertia.
3. **Model Validation:** 
   * **1-Hour Model ($R^2 = 0.9738$):** Validation RMSE of $0.0062\text{ m}$ ($6.2\text{ mm}$).
   * **3-Hour Model ($R^2 = 0.8919$):** Validation RMSE of $0.0125\text{ m}$ ($12.5\text{ mm}$).

---

## 💡 Key Learning Outcomes & Future Improvements

While functional, this project was built primarily as a practical sandbox to explore end-to-end full-stack development, modern API design, and machine learning pipelines. The current implementation represents a baseline prototype with clear opportunities for refinement.

### **Engineering & Technical Skills Developed**
* **Asynchronous Web Services with FastAPI:** Designed modular API endpoints, structured data-ingestion pipelines, handled request lifecycles, and configured CORS for cross-origin browser requests.
* **React & TypeScript Proficiency:** Reinforced component architecture, asynchronous state management for live API polling, and strict interface definitions for API response schemas.
* **XGBoost & Time-Series Modeling:** Gained hands-on experience with gradient boosting algorithms, stationary target transformations ($\Delta\text{stage}$), lag feature generation, and avoiding data leakage.
* **Full-Stack Containerization:** Set up clean directory structures, virtual environments, Docker containerization, and source-control hygiene for Python and Node environments.

### **Limitations & Path to Higher Accuracy**
The current model performance serves as a foundational proof-of-concept rather than a production-grade hydrological tool:

* **Error Margins:** While the validation metrics look promising on baseline flow, prediction errors widen during sudden heavy rainfall events. Predicting river stage changes driven by localised precipitation remains complex.
* **Feature Scope:** The pipeline relies on a limited set of gauge readings. Accuracy could be improved by incorporating catchment soil moisture levels, evapotranspiration rates, and multi-station radar precipitation grids.
* **Model Exploration:** Moving beyond standard gradient boosting to other architectures like Temporal Fusion Transformers or Graph-Based Models could prove beneficial for long-term hydrological memory and pattern detection.
