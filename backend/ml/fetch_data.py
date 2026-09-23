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

# ==================================================
# Imports and Configuration
# ==================================================


import os
import time
import logging
import requests
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta, timezone

# Configure logging to show timestamps and log levels
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ==================================================
# API URLs
# ==================================================

FLOOD_MONITORING_URL = (
    "https://environment.data.gov.uk/" "flood-monitoring/data/readings"
)

HYDROLOGY_URL = "https://environment.data.gov.uk/" "hydrology/id/stations"

ARCHIVE_URL = (
    "https://environment.data.gov.uk/" "flood-monitoring/archive/readings-{date}.csv"
)

# The real-time Flood Monitoring API provides readings for the
# recent period (up to about four weeks). Older river data is
# retrieved automatically from the daily archive CSVs.
REALTIME_HISTORY_DAYS = 28

# Filtered archive data is cached locally so old daily CSVs do not
# need to be downloaded again every time the script is run.
ARCHIVE_CACHE_DIR = os.path.join(OUTPUT_DIR, "archive_cache")


# ==================================================
# Fetch Austin's Bridge - real-time API
# ==================================================


def fetch_austins_bridge_realtime(
    start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """Fetches Austin's Bridge river-level readings from the Environment Agency's real-time Flood Monitoring API."""

    logging.info(
        f"Fetching Austin's Bridge from real-time API: {start_date.date()} -> {end_date.date()}"
    )

    # 46122 is the station ID for Austin's Bridge.
    measure_id = "46122-level-stage-i-15_min-m"

    url = f"https://environment.data.gov.uk/flood-monitoring/id/measures/{measure_id}/readings"

    params = {
        "startdate": start_date.strftime("%Y-%m-%d"),
        "enddate": end_date.strftime("%Y-%m-%d"),
        "_limit": 1000,  # 1 week of 15-minute readings is 672, so 1000 is safe for a week-long chunk
    }

    try:
        # Fetch data from the real-time API and convert to a DataFrame.
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        records = [
            {"timestamp": item["dateTime"], "river_level": item["value"]}
            for item in data.get("items", [])
            if "dateTime" in item and "value" in item
        ]

        df = pd.DataFrame(records)

        # Real-time API can occasionally return no readings for the requested window, or internal server errors.
        if df.empty:
            logging.warning(
                "No Austin's Bridge readings returned from the real-time API."
            )
            return df

        # Remove timestamps that exceed the requested window, as the API can return a few extra readings.
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        df = (
            df.sort_values("timestamp")
            .drop_duplicates("timestamp")
            .set_index("timestamp")
        )

        # Ensure we keep only the exact requested window.
        df = df[(df.index >= start_date) & (df.index < end_date)]

        logging.info(
            f"Downloaded {len(df)} Austin's Bridge readings from real-time API."
        )

        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch Austin's Bridge real-time data: {e}")

        return pd.DataFrame()


# ==================================================
# Fetch Austin's Bridge - historic archive
# ==================================================


def fetch_austins_bridge_archive(
    start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """Fetches Austin's Bridge river-level readings from the Environment Agency's daily archive CSVs."""

    # Ensure inputs are timezone-aware in UTC for consistent filtering.
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    logging.info(
        f"Fetching Austin's Bridge from archive: {start_date.date()} -> {end_date.date()}"
    )

    os.makedirs(ARCHIVE_CACHE_DIR, exist_ok=True)

    measure_id = "46122-level-stage-i-15_min-m"

    frames = []

    # Archive files are daily, so determine start_date (is inclusive).
    current_day = datetime(
        start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc
    )

    # end_date is exclusive so get the last instant of the previous day (inclusive).
    last_archive_instant = end_date - timedelta(microseconds=1)

    last_day = datetime(
        last_archive_instant.year,
        last_archive_instant.month,
        last_archive_instant.day,
        tzinfo=timezone.utc,
    )

    # Loop through each day in the requested window and fetch the corresponding archive CSV.
    while current_day <= last_day:
        day_string = current_day.strftime("%Y-%m-%d")

        cache_path = os.path.join(ARCHIVE_CACHE_DIR, f"austins_bridge_{day_string}.csv")

        try:
            # Reuse previously filtered data when available.
            if os.path.exists(cache_path):
                logging.info(f"Using cached Austin's Bridge archive: {day_string}")

                day_df = pd.read_csv(cache_path, parse_dates=["timestamp"])

                if not day_df.empty:
                    day_df["timestamp"] = pd.to_datetime(day_df["timestamp"], utc=True)
                    frames.append(day_df)

                current_day += timedelta(days=1)
                continue

            archive_url = ARCHIVE_URL.format(date=day_string)

            logging.info(f"Downloading archive file: {day_string}")

            response = requests.get(archive_url, timeout=120)

            response.raise_for_status()

            day_df = pd.read_csv(
                BytesIO(response.content),
                usecols=["dateTime", "measure", "value"],
                dtype={"measure": "string", "value": "string"},
            )

            # The archive contains readings for every measure.
            # Keep only Austin's Bridge's exact measure.
            day_df = day_df[day_df["measure"].str.endswith(measure_id, na=False)][
                ["dateTime", "value"]
            ].copy()

            if day_df.empty:

                logging.warning(
                    f"No Austin's Bridge readings found "
                    f"in archive for {day_string}."
                )

                current_day += timedelta(days=1)
                continue

            day_df = day_df.rename(
                columns={"dateTime": "timestamp", "value": "river_level"}
            )

            day_df["timestamp"] = pd.to_datetime(day_df["timestamp"], utc=True)

            # Archive CSV values can occasionally contain malformed
            # non-numeric values. Drop those rather than inventing
            # a replacement value.
            day_df["river_level"] = pd.to_numeric(
                day_df["river_level"], errors="coerce"
            )

            invalid_count = day_df["river_level"].isna().sum()

            if invalid_count:
                logging.warning(
                    f"Dropping {invalid_count} invalid river-level "
                    f"values from archive {day_string}."
                )

            day_df = day_df.dropna(subset=["river_level"])

            day_df = day_df.sort_values("timestamp").drop_duplicates("timestamp")

            # Cache only the filtered station/measure data, not the
            # much larger all-station archive CSV.
            day_df.to_csv(cache_path, index=False)

            frames.append(day_df)

        except (requests.exceptions.RequestException, ValueError) as e:

            logging.error(f"Failed to fetch archive for {day_string}: {e}")

        current_day += timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")

    # Keep only the exact requested window.
    df = df[(df.index >= start_date) & (df.index < end_date)]

    logging.info(f"Downloaded {len(df)} Austin's Bridge readings " f"from archive.")

    return df


# ==================================================
# Fetch Austin's Bridge - automatic source selection
# ==================================================


def fetch_austins_bridge(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetches Austin's Bridge river-level readings, automatically selecting between the real-time API and the historic archive based on the requested date range."""

    # The Environment Agency real-time API covers the recent
    # period. Use the archive automatically for older data.
    cutoff = datetime.now(timezone.utc) - timedelta(days=REALTIME_HISTORY_DAYS)

    # Entirely historical window -> archive only.
    if end_date <= cutoff:

        return fetch_austins_bridge_archive(start_date=start_date, end_date=end_date)

    # Entirely recent window -> real-time API only.
    if start_date >= cutoff:

        return fetch_austins_bridge_realtime(start_date=start_date, end_date=end_date)

    # This chunk crosses the boundary between the archive and
    # real-time API. Fetch both halves and combine them.
    logging.info("Chunk crosses the real-time/archive boundary; " "using both sources.")

    archive_df = fetch_austins_bridge_archive(start_date=start_date, end_date=cutoff)

    realtime_df = fetch_austins_bridge_realtime(start_date=cutoff, end_date=end_date)

    frames = [df for df in [archive_df, realtime_df] if not df.empty]

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()

    # Prefer the real-time value if an archive and real-time
    # reading happen to contain the same timestamp.
    df = df[~df.index.duplicated(keep="last")]

    return df


# ==================================================
# Fetch Princetown - one time window
# ==================================================


def fetch_princetown(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetches Princetown rainfall readings from the Environment Agency's Hydrology API."""

    logging.info(
        f"Fetching Princetown rainfall: " f"{start_date.date()} -> {end_date.date()}"
    )

    measure_id = "f0676423-f697-4dec-8958-440f335cce9a" "-rainfall-t-900-mm-qualified"

    url = (
        f"https://environment.data.gov.uk/"
        f"hydrology/id/measures/{measure_id}/readings.json"
    )

    params = {
        "mineq-date": start_date.strftime("%Y-%m-%d"),
        "maxeq-date": end_date.strftime("%Y-%m-%d"),
        "_limit": 1000,
    }

    try:

        response = requests.get(url, params=params, timeout=60)

        response.raise_for_status()

        data = response.json()

        records = [
            {"timestamp": item["dateTime"], "rainfall": item["value"]}
            for item in data.get("items", [])
            if "dateTime" in item and "value" in item
        ]

        df = pd.DataFrame(records)

        if df.empty:
            logging.warning("No Princetown rainfall readings returned.")
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        df = (
            df.sort_values("timestamp")
            .drop_duplicates("timestamp")
            .set_index("timestamp")
        )

        # Keep only the exact requested window.
        df = df[(df.index >= start_date) & (df.index < end_date)]

        logging.info(f"Downloaded {len(df)} Princetown readings.")

        return df

    except requests.exceptions.RequestException as e:

        logging.error(f"Failed to fetch Princetown rainfall: {e}")

        return pd.DataFrame()


# ==================================================
# Download & Combine in 7-day chunks
# ==================================================


def download_and_combine_data(days_back: int = 70, chunk_days: int = 7) -> pd.DataFrame:
    """Downloads and combines Austin's Bridge river-level and Princetown rainfall data in 7-day chunks, returning a single DataFrame with both datasets."""

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    logging.info(f"Requested data period: " f"{start_date} -> {end_date}")

    river_chunks = []
    rain_chunks = []

    current_start = start_date

    while current_start < end_date:

        current_end = min(current_start + timedelta(days=chunk_days), end_date)

        logging.info(f"----------------------------------------")

        logging.info(f"Fetching chunk: " f"{current_start} -> {current_end}")

        # ------------------------------------------
        # River
        # ------------------------------------------

        df_river = fetch_austins_bridge(start_date=current_start, end_date=current_end)

        time.sleep(1)

        # ------------------------------------------
        # Rainfall
        # ------------------------------------------

        df_rain = fetch_princetown(start_date=current_start, end_date=current_end)

        time.sleep(1)

        # ------------------------------------------
        # Store successful chunks
        # ------------------------------------------

        if not df_river.empty:
            river_chunks.append(df_river)
        else:
            logging.warning(
                f"No river data for chunk "
                f"{current_start.date()} -> "
                f"{current_end.date()}"
            )

        if not df_rain.empty:
            rain_chunks.append(df_rain)
        else:
            logging.warning(
                f"No rainfall data for chunk "
                f"{current_start.date()} -> "
                f"{current_end.date()}"
            )

        # Move to next chunk
        current_start = current_end

    # ==================================================
    # Combine all chunks
    # ==================================================

    if not river_chunks:
        logging.error("No Austin's Bridge data was downloaded.")
        return pd.DataFrame()

    if not rain_chunks:
        logging.error("No Princetown rainfall data was downloaded.")
        return pd.DataFrame()

    df_river = pd.concat(river_chunks).sort_index()
    df_river = df_river[~df_river.index.duplicated(keep="first")]

    df_rain = pd.concat(rain_chunks).sort_index()
    df_rain = df_rain[~df_rain.index.duplicated(keep="first")]

    # Combine the two datasets
    combined = pd.concat(
        [df_river[["river_level"]], df_rain[["rainfall"]]], axis=1, sort=False
    )

    # Only retain timestamps where both exist
    combined = combined.dropna()

    combined = combined.reset_index()

    combined = combined.sort_values("timestamp")

    logging.info(f"Final combined DataFrame shape: {combined.shape}")

    if not combined.empty:
        logging.info(
            f"Final date range: "
            f"{combined['timestamp'].min()} -> "
            f"{combined['timestamp'].max()}"
        )

    return combined


# ==================================================
# Main
# ==================================================


def main():
    """Main function to run the EA telemetry ingestion pipeline and save the combined data to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, "raw_telemetry.csv")

    logging.info("Starting EA telemetry ingestion pipeline...")

    combined_data = download_and_combine_data(days_back=365, chunk_days=7)

    if combined_data.empty:
        logging.error("No combined data available. CSV was not written.")
        return

    # Show a quick preview
    print("\nCombined Data:\n")
    print(combined_data.head())
    print()
    print(combined_data.shape)
    print()
    print(combined_data.tail())

    # Save CSV
    combined_data.to_csv(output_path, index=False)

    logging.info(f"Data ingestion complete!")

    logging.info(
        f"Saved {len(combined_data)} rows to " f"{os.path.abspath(output_path)}"
    )

    print(combined_data["rainfall"].describe())
    print(combined_data["rainfall"].sum())


if __name__ == "__main__":
    main()
