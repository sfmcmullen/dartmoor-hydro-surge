"""
main.py
-------
FastAPI application serving live hydrological predictions for the River Dart
at Austin's Bridge using real-time Environment Agency telemetry.
"""

# ==================================================
# Imports and Configuration
# ==================================================


import os
import logging
import traceback
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBRegressor

from ml.fetch_data import download_and_combine_data
from ml.prepare_features import load_and_clean_data, generate_features

logging.basicConfig(level=logging.INFO)

# FastAPI Application Setup
app = FastAPI(
    title="Dartmoor Hydro Surge API",
    description="Real-time river level forecasting using EA telemetry and XGBoost models.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Loading and Global Variables
BASE_DIR = os.path.dirname(__file__)
MODEL_1H_PATH = os.path.join(BASE_DIR, "models", "xgboost_1h.json")
MODEL_3H_PATH = os.path.join(BASE_DIR, "models", "xgboost_3h.json")

model_1h = None
model_3h = None


# ==================================================
# API Endpoints
# ==================================================


@app.on_event("startup")
def load_models():
    global model_1h, model_3h
    if not os.path.exists(MODEL_1H_PATH) or not os.path.exists(MODEL_3H_PATH):
        logging.error("Model files missing. Train models before running API.")
        return

    model_1h = XGBRegressor()
    model_1h.load_model(MODEL_1H_PATH)

    model_3h = XGBRegressor()
    model_3h.load_model(MODEL_3H_PATH)
    logging.info("XGBoost forecasting models loaded successfully.")


@app.get("/")
def read_root():
    return {
        "service": "Dartmoor Hydro Surge API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/api/predict",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": model_1h is not None and model_3h is not None,
    }


@app.get("/api/predict")
def get_latest_prediction():
    if model_1h is None or model_3h is None:
        raise HTTPException(status_code=500, detail="Models not loaded on server.")

    try:
        # 1. Fetch past 7 days of telemetry
        df_raw = download_and_combine_data(days_back=7, chunk_days=7)
        if df_raw.empty:
            raise HTTPException(
                status_code=502, detail="Failed to retrieve telemetry from EA API."
            )

        temp_raw_path = os.path.join(BASE_DIR, "data", "temp_live.csv")
        df_raw.to_csv(temp_raw_path, index=False)

        # 2. Extract seasonal and lag features
        df_clean = load_and_clean_data(temp_raw_path)
        df_feat = generate_features(df_clean)

        if df_feat.empty:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data points to compute feature lags.",
            )

        if "timestamp" not in df_feat.columns and df_feat.index.name == "timestamp":
            df_feat = df_feat.reset_index()

        latest_row = df_feat.iloc[-1:]

        # Filter out all target columns (both old stage targets and new delta targets)
        ignore_cols = [
            "timestamp",
            "target_stage_1h",
            "target_stage_3h",
            "target_delta_1h",
            "target_delta_3h",
        ]
        feature_cols = [c for c in latest_row.columns if c not in ignore_cols]

        X_latest = latest_row[feature_cols]

        # 3. Predict deltas
        delta_1h = float(model_1h.predict(X_latest)[0])
        delta_3h = float(model_3h.predict(X_latest)[0])

        current_level = float(latest_row["river_level"].values[0])

        # Calculate expected absolute river stages
        pred_1h = current_level + delta_1h
        pred_3h = current_level + delta_3h

        if "timestamp" in latest_row.columns:
            ts_val = latest_row["timestamp"].values[0]
        else:
            ts_val = latest_row.index[0]

        current_time = pd.to_datetime(ts_val).isoformat()

        return {
            "timestamp": current_time,
            "current_stage_m": round(current_level, 3),
            "forecast_1h_m": round(pred_1h, 3),
            "forecast_3h_m": round(pred_3h, 3),
            "delta_1h_m": round(delta_1h, 3),
            "delta_3h_m": round(delta_3h, 3),
        }
    except Exception as e:
        logging.error(f"Prediction endpoint error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
