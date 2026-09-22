"""
train_model.py
--------------
Trains XGBoost regression models for 1-hour and 3-hour stage deltas
using seasonal and baseline features with temporal validation.
"""

# ==================================================
# Imports and Configuration
# ==================================================


import os
import logging
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ==================================================
# Helper Functions
# ==================================================


def load_processed_data(file_path: str) -> pd.DataFrame:
    """Loads the processed feature dataset for model training."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature dataset not found at: {file_path}")

    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def train_eval_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    current_stage_val: pd.Series,
    target_name: str,
) -> XGBRegressor:
    """Trains an XGBoost regressor and evaluates it on the validation set."""
    logging.info(f"Training XGBoost Regressor for target: {target_name}...")

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="rmse",
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate predictions converted back to absolute river stage height
    pred_deltas = model.predict(X_val)
    pred_abs_stage = current_stage_val + pred_deltas
    actual_abs_stage = current_stage_val + y_val

    rmse = np.sqrt(mean_squared_error(actual_abs_stage, pred_abs_stage))
    mae = mean_absolute_error(actual_abs_stage, pred_abs_stage)
    r2 = r2_score(actual_abs_stage, pred_abs_stage)

    logging.info(f"[{target_name}] Validation RMSE: {rmse:.4f} m")
    logging.info(f"[{target_name}] Validation MAE:  {mae:.4f} m")
    logging.info(f"[{target_name}] Validation R^2:   {r2:.4f}")

    return model


# ==================================================
# Main Execution - Train and Save Models
# ==================================================


def main():
    """Main function to train and save XGBoost models for 1-hour and 3-hour river stage forecasts."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_processed_data(DATA_PATH)

    ignore_cols = ["timestamp", "target_delta_1h", "target_delta_3h"]
    feature_cols = [col for col in df.columns if col not in ignore_cols]

    # Temporal split (80% Train, 20% Validation)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    logging.info(f"Features count: {len(feature_cols)}")
    logging.info(
        f"Train set: {len(X_train)} samples | Validation set: {len(X_val)} samples"
    )

    # 1. Train 1-Hour Forecast Model
    y_train_1h = train_df["target_delta_1h"]
    y_val_1h = val_df["target_delta_1h"]
    model_1h = train_eval_model(
        X_train, y_train_1h, X_val, y_val_1h, val_df["river_level"], "1-Hour Forecast"
    )

    path_1h = os.path.join(MODEL_DIR, "xgboost_1h.json")
    model_1h.save_model(path_1h)
    logging.info(f"Saved 1-hour model to {path_1h}")

    # 2. Train 3-Hour Forecast Model
    df_3h = df.dropna(subset=["target_delta_3h"]).reset_index(drop=True)
    split_idx_3h = int(len(df_3h) * 0.80)

    X_train_3h = df_3h.iloc[:split_idx_3h][feature_cols]
    y_train_3h = df_3h.iloc[:split_idx_3h]["target_delta_3h"]
    X_val_3h = df_3h.iloc[split_idx_3h:][feature_cols]
    y_val_3h = df_3h.iloc[split_idx_3h:]["target_delta_3h"]
    val_stage_3h = df_3h.iloc[split_idx_3h:]["river_level"]

    model_3h = train_eval_model(
        X_train_3h, y_train_3h, X_val_3h, y_val_3h, val_stage_3h, "3-Hour Forecast"
    )

    path_3h = os.path.join(MODEL_DIR, "xgboost_3h.json")
    model_3h.save_model(path_3h)
    logging.info(f"Saved 3-hour model to {path_3h}")


if __name__ == "__main__":
    main()
