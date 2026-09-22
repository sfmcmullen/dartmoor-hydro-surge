"""
prepare_features.py
-------------------
Ingests raw_telemetry.csv, enforces a uniform 15-minute time grid,
and generates lag, rolling aggregate, seasonal, and target delta features.
"""

# ==================================================
# Imports and Configuration
# ==================================================


import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "raw_telemetry.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "processed_features.csv")


# ==================================================
# Load, Clean, and Feature Engineering Functions
# ==================================================


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Reads raw CSV, enforces 15-min index grid, and imputes missing data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")

    full_idx = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="15min",
        tz="UTC",
        name="timestamp",
    )
    df = df.reindex(full_idx)

    df["river_level"] = df["river_level"].ffill().bfill()
    df["rainfall"] = df["rainfall"].fillna(0.0)

    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates lag, rolling aggregate, seasonal, and target delta features."""
    df_feat = df.copy()

    # 1. Cyclical Calendar / Seasonal Features
    month = df_feat.index.month
    day_of_year = df_feat.index.dayofyear

    df_feat["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df_feat["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df_feat["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df_feat["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df_feat["hour"] = df_feat.index.hour

    # 2. Baseflow Context (7-day rolling stage mean)
    # 7 days * 24 hours * 4 steps = 672 steps
    df_feat["river_mean_7d"] = (
        df_feat["river_level"].rolling(window=672, min_periods=96).mean()
    )
    df_feat["river_stage_ratio_7d"] = df_feat["river_level"] - df_feat["river_mean_7d"]

    # 3. Short-term River Stage Lags (15m, 30m, 45m, 1h, 2h, 3h)
    lag_steps = [1, 2, 3, 4, 8, 12]
    for step in lag_steps:
        df_feat[f"river_lag_{step*15}m"] = df_feat["river_level"].shift(step)

    # 4. Historical Rainfall Lags (15m, 30m, 1h, 2h)
    rain_lags = [1, 2, 4, 8]
    for step in rain_lags:
        df_feat[f"rain_lag_{step*15}m"] = df_feat["rainfall"].shift(step)

    # 5. Cumulative Rainfall Totals (1h, 3h, 6h, 12h, 24h, 48h)
    rolling_windows = {"1h": 4, "3h": 12, "6h": 24, "12h": 48, "24h": 96, "48h": 192}
    for label, steps in rolling_windows.items():
        df_feat[f"rain_sum_{label}"] = (
            df_feat["rainfall"].rolling(window=steps, min_periods=1).sum()
        )

    # 6. Stage Derivatives (Rate of change)
    df_feat["river_diff_15m"] = df_feat["river_level"].diff(1)
    df_feat["river_diff_1h"] = df_feat["river_level"].diff(4)

    # 7. Delta Target Variables (Change in level relative to current stage)
    # Target = Future stage - Current stage
    future_1h = df_feat["river_level"].shift(-4)
    future_3h = df_feat["river_level"].shift(-12)

    df_feat["target_delta_1h"] = future_1h - df_feat["river_level"]
    df_feat["target_delta_3h"] = future_3h - df_feat["river_level"]

    # Drop initial setup rows containing NaN values from 7-day rolling window
    df_feat = df_feat.dropna(subset=["river_mean_7d"]).copy()

    return df_feat


# ==================================================
# Main Execution
# ==================================================


def main():
    """Main function to execute the feature engineering pipeline."""
    logging.info("Starting feature engineering pipeline...")

    df_clean = load_and_clean_data(INPUT_FILE)
    df_processed = generate_features(df_clean)

    # Drop rows where future target is missing
    df_final = df_processed.dropna(subset=["target_delta_1h"]).reset_index()

    df_final.to_csv(OUTPUT_FILE, index=False)

    logging.info(
        f"Feature processing complete. Saved {len(df_final)} rows to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
