"""Build a 6-hour ETH + whale-feature dataset from 2017 to latest common timestamp."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_PATH = "data/raw/Binance_ETHUSDT_1h.csv"
WHALE_PATH = "data/raw/bigquery_whale_data.csv"
OUTPUT_PATH = "data/processed/eth_merged_6h_2017_to_latest.csv"

START_TS = pd.Timestamp("2017-08-17 06:00:00")


def build_dataset() -> pd.DataFrame:
    """Load, clean, aggregate, merge, and label ETH + whale data at 6-hour granularity."""

    # 1) Load raw files and parse timestamps first so we can compute common end time.
    price_df = pd.read_csv(PRICE_PATH)
    whale_df = pd.read_csv(WHALE_PATH)

    price_df["hour"] = pd.to_datetime(price_df["Date"], errors="coerce").dt.tz_localize(None)
    whale_df["hour"] = pd.to_datetime(whale_df["hour_timestamp"], errors="coerce", utc=True).dt.tz_convert(None)

    price_df = price_df.dropna(subset=["hour"])
    whale_df = whale_df.dropna(subset=["hour"])

    # End at the latest timestamp available in both sources.
    common_end_ts = min(price_df["hour"].max(), whale_df["hour"].max())
    print(f"Using time range: {START_TS} to {common_end_ts}")

    # 2) Clean and resample price data to 6-hour windows.
    if "Symbol" in price_df.columns:
        price_df = price_df[price_df["Symbol"].str.upper() == "ETHUSDT"]

    price_df = price_df.set_index("hour").sort_index()
    price_df = price_df.loc[(price_df.index >= START_TS) & (price_df.index <= common_end_ts)]

    price_agg_map = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume ETH": "sum",
        "Volume USDT": "sum",
        "tradecount": "sum",
    }
    price_agg_map = {col: agg for col, agg in price_agg_map.items() if col in price_df.columns}

    price_6h = price_df.resample("6h").agg(price_agg_map).dropna(how="all").reset_index()
    price_feature_cols = [c for c in price_6h.columns if c != "hour"]
    price_6h = price_6h[["hour", *price_feature_cols]]

    # 3) Clean and resample whale features to 6-hour windows.
    whale_df = whale_df.set_index("hour").sort_index()
    whale_df = whale_df.loc[(whale_df.index >= START_TS) & (whale_df.index <= common_end_ts)]

    whale_agg_map: dict[str, str] = {}
    for col in whale_df.columns:
        if col == "hour_timestamp":
            continue

        col_lower = col.lower()
        if ("avg_gas" in col_lower) or ("mean_gas" in col_lower):
            whale_agg_map[col] = "mean"
        elif "max_gas" in col_lower:
            whale_agg_map[col] = "max"
        elif ("unique_" in col_lower) and (
            ("sender" in col_lower) or ("receiver" in col_lower) or ("address" in col_lower) or ("entity" in col_lower)
        ):
            whale_agg_map[col] = "max"
        elif col_lower.endswith("_count") or col_lower.endswith("_volume"):
            whale_agg_map[col] = "sum"
        elif pd.api.types.is_numeric_dtype(whale_df[col]):
            whale_agg_map[col] = "sum"

    whale_6h = whale_df.resample("6h").agg(whale_agg_map).dropna(how="all").reset_index()
    if "hour_timestamp" in whale_6h.columns:
        whale_6h = whale_6h.drop(columns=["hour_timestamp"])

    # 4) Merge and drop rows with missing values.
    merged_df = pd.merge(price_6h, whale_6h, on="hour", how="inner").sort_values("hour").reset_index(drop=True)

    rows_before_dropna = len(merged_df)
    merged_df = merged_df.dropna().reset_index(drop=True)
    rows_after_dropna = len(merged_df)
    print(f"Rows before dropna: {rows_before_dropna}")
    print(f"Rows after dropna:  {rows_after_dropna}")
    print(f"Rows dropped:       {rows_before_dropna - rows_after_dropna}")

    # 5) Build label for next 6-hour horizon.
    # Use next 6h Low to capture downside risk inside the next interval.
    merged_df["future_price"] = merged_df["Low"].shift(-1)
    merged_df["future_drop_pct"] = (merged_df["future_price"] - merged_df["Close"]) / merged_df["Close"]
    merged_df["drawdown_6h_label"] = np.where(merged_df["future_drop_pct"] <= -0.03, 1, 0)

    # Drop final row because it has no future interval for target calculation.
    merged_df = merged_df.dropna(subset=["future_price"]).reset_index(drop=True)

    # 6) Final cleanup and save.
    merged_df = merged_df.drop(columns=["future_price", "future_drop_pct"])
    merged_df["drawdown_6h_label"] = merged_df["drawdown_6h_label"].astype(int)

    merged_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Final dataset shape: {merged_df.shape}")
    print(merged_df.head())

    return merged_df


if __name__ == "__main__":
    df = build_dataset()
