"""Build a 6-hour ETH + whale-feature dataset for calendar year 2022."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_PATH = "data/raw/Binance_ETHUSDT_1h.csv"
WHALE_PATH = "data/raw/bigquery_whale_data.csv"
OUTPUT_PATH = "data/processed/eth_merged_6h_2022.csv"

START_2022 = pd.Timestamp("2022-01-01 00:00:00")
END_2022 = pd.Timestamp("2022-12-31 23:59:59")


def build_dataset() -> pd.DataFrame:
    """Load, clean, aggregate, merge, and label ETH + whale data at 6-hour granularity."""

    # 1) Load and clean price data
    price_df = pd.read_csv(PRICE_PATH)
    price_df["hour"] = pd.to_datetime(price_df["Date"], errors="coerce").dt.tz_localize(None)
    price_df = price_df.dropna(subset=["hour"])

    if "Symbol" in price_df.columns:
        price_df = price_df[price_df["Symbol"].str.upper() == "ETHUSDT"]

    price_df = price_df.set_index("hour").sort_index()
    price_df = price_df.loc[(price_df.index >= START_2022) & (price_df.index <= END_2022)]

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

    # Keep only modeling-relevant numeric price features + hour.
    price_feature_cols = [c for c in price_6h.columns if c != "hour"]
    price_6h = price_6h[["hour", *price_feature_cols]]

    # 2) Load and clean whale data
    whale_df = pd.read_csv(WHALE_PATH)
    whale_df["hour"] = pd.to_datetime(whale_df["hour_timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    whale_df = whale_df.dropna(subset=["hour"])

    whale_df = whale_df.set_index("hour").sort_index()
    whale_df = whale_df.loc[(whale_df.index >= START_2022) & (whale_df.index <= END_2022)]

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
            # Numeric totals/default activity metrics -> sum by default.
            whale_agg_map[col] = "sum"

    whale_6h = whale_df.resample("6h").agg(whale_agg_map).dropna(how="all").reset_index()

    if "hour_timestamp" in whale_6h.columns:
        whale_6h = whale_6h.drop(columns=["hour_timestamp"])

    # 3) Merge on hour
    merged_df = pd.merge(price_6h, whale_6h, on="hour", how="inner").sort_values("hour").reset_index(drop=True)

    rows_before_dropna = len(merged_df)
    merged_df = merged_df.dropna().reset_index(drop=True)
    rows_after_dropna = len(merged_df)
    print(f"Rows before dropna: {rows_before_dropna}")
    print(f"Rows after dropna:  {rows_after_dropna}")
    print(f"Rows dropped:       {rows_before_dropna - rows_after_dropna}")

    # 4) Create 6h drawdown label for next 6h window.
    # We use next 6h Low as future_price because it captures intra-window downside risk
    # better than next Close.
    merged_df["future_price"] = merged_df["Low"].shift(-1)
    merged_df["future_drop_pct"] = (merged_df["future_price"] - merged_df["Close"]) / merged_df["Close"]
    merged_df["drawdown_6h_label"] = np.where(merged_df["future_drop_pct"] <= -0.05, 1, 0)

    # Last row has no future 6h row; we drop it to avoid training on unknown target.
    merged_df = merged_df.dropna(subset=["future_price"]).reset_index(drop=True)

    # 5) Final cleanup
    merged_df = merged_df.drop(columns=["future_price", "future_drop_pct"])
    merged_df["drawdown_6h_label"] = merged_df["drawdown_6h_label"].astype(int)

    merged_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Final dataset shape: {merged_df.shape}")
    print(merged_df.head())

    return merged_df


if __name__ == "__main__":
    df = build_dataset()
