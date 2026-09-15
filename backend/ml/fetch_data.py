"""
fetch_data.py
-------------
Downloads 15-minute telemetry data for:

    - River Dart at Austin's Bridge
      Flood Monitoring API
      Station: 46122

    - Princetown rainfall
      Hydrology API
      Station: 47166
      15-minute rainfall measure
"""

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone


# ==================================================
# Configuration
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data"
)


# ==================================================
# API URLs
# ==================================================

FLOOD_MONITORING_URL = (
    "https://environment.data.gov.uk/"
    "flood-monitoring/data/readings"
)

HYDROLOGY_URL = (
    "https://environment.data.gov.uk/"
    "hydrology/id/stations"
)


# ==================================================
# Fetch Austin's Bridge
# ==================================================

def fetch_austins_bridge(days_back: int = 7) -> pd.DataFrame:
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    measure_id = "46122-level-stage-i-15_min-m"

    url = (
        f"https://environment.data.gov.uk/flood-monitoring/"
        f"id/measures/{measure_id}/readings"
    )

    limit_param = days_back * 24 * 4  # 15-minute readings per day

    params = {
        "since": start_date.strftime("%Y-%m-%dT00:00:00Z"),
        "_limit": limit_param,
        # "_sorted": "true"
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    records = [
        {
            "timestamp": item["dateTime"],
            "river_level": item["value"]
        }
        for item in data.get("items", [])
    ]

    df = pd.DataFrame(records)

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    return df


# ==================================================
# Fetch Princetown
# ==================================================

def fetch_princetown(days_back: int = 7) -> pd.DataFrame:

    logging.info(
        f"Fetching Princetown rainfall "
        f"(Past {days_back} days)..."
    )

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    # Exact 15-minute Princetown rainfall measure
    measure_id = (
        "f0676423-f697-4dec-8958-440f335cce9a"
        "-rainfall-t-900-mm-qualified"
    )

    url = (
        f"https://environment.data.gov.uk/"
        f"hydrology/id/measures/{measure_id}/readings.json"
    )

    params = {
        "mineq-date": start_date.strftime("%Y-%m-%d"),
        "maxeq-date": end_date.strftime("%Y-%m-%d"),
        "_limit": 1000
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()
        items = data.get("items", [])

        if not items:
            logging.warning(
                "No Princetown rainfall readings returned."
            )
            return pd.DataFrame()

        records = [
            {
                "timestamp": item["dateTime"],
                "rainfall": item["value"]
            }
            for item in items
            if "dateTime" in item and "value" in item
        ]

        df = pd.DataFrame(records)

        if df.empty:
            logging.warning(
                "No valid Princetown rainfall readings returned."
            )
            return df

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True
        )

        df = (
            df
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
            .set_index("timestamp")
        )

        logging.info(
            f"Successfully downloaded "
            f"{len(df)} Princetown rainfall readings."
        )

        return df

    except requests.exceptions.RequestException as e:

        logging.error(
            f"Failed to fetch Princetown rainfall data: {e}"
        )

        return pd.DataFrame()


# ==================================================
# Download & Combine
# ==================================================

def download_and_combine_data(
    days_back: int = 7
) -> pd.DataFrame:

    # Fetch both datasets
    df_river = fetch_austins_bridge(
        days_back=days_back
    )

    time.sleep(1)

    df_rain = fetch_princetown(
        days_back=days_back
    )

    # Make sure both fetches succeeded
    if df_river.empty:
        logging.error("Could not fetch river level data.")
        return pd.DataFrame()

    if df_rain.empty:
        logging.error("Could not fetch rainfall data.")
        return pd.DataFrame()

    # Both DataFrames already have timestamp as their index.
    # Both are already 15-minute measurements.
    combined = pd.concat(
        [
            df_river[["river_level"]],
            df_rain[["rainfall"]]
        ],
        axis=1,
        sort=False
    )

    # Keep only timestamps where both datasets have data
    combined = combined.dropna()

    # Put timestamp back into a normal column
    combined = combined.reset_index()

    logging.info(
        f"Combined DataFrame shape: {combined.shape}"
    )

    return combined


# ==================================================
# Main
# ==================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "raw_telemetry.csv"
    )

    logging.info(
        "Starting EA telemetry ingestion pipeline..."
    )

    combined_data = download_and_combine_data(
        days_back=7
    )

    if combined_data.empty:
        logging.error(
            "No combined data available. CSV was not written."
        )
        return

    # Show a quick preview
    print("\nCombined Data:\n")
    print(combined_data.head())
    print()
    print(combined_data.shape)
    print()
    print(combined_data.tail())

    # Save CSV
    combined_data.to_csv(
        output_path,
        index=False
    )

    logging.info(
        f"Data ingestion complete!"
    )

    logging.info(
        f"Saved {len(combined_data)} rows to "
        f"{os.path.abspath(output_path)}"
    )

    print(combined_data["rainfall"].describe())
    print(combined_data["rainfall"].sum())

# def main():
#     df1 = fetch_austins_bridge(days_back=7)

#     print("\r\nAustin's Bridge Data:\r\n")
#     print(df1.head())
#     print(df1.shape)
#     print(df1.tail())

#     df2 = fetch_princetown(days_back=7)

#     print("\r\nPrincetown Data:\r\n")
#     print(df2.head())
#     print(df2.shape)
#     print(df2.tail())


if __name__ == "__main__":
    main()

