"""Build a 6-hour ETH-only dataset from 2017 to latest available timestamp."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_PATH = "data/raw/Binance_ETHUSDT_1h.csv"
OUTPUT_PATH = "data/processed/eth_price_only_6h_2017_to_latest.csv"

START_TS = pd.Timestamp("2017-08-17 06:00:00")


def build_dataset() -> pd.DataFrame:
    """Load, clean, aggregate, and label ETH price data at 6-hour granularity."""
    price_df = pd.read_csv(PRICE_PATH)

    price_df["hour"] = pd.to_datetime(price_df["Date"], errors="coerce").dt.tz_localize(None)
    price_df = price_df.dropna(subset=["hour"])

    if "Symbol" in price_df.columns:
        price_df = price_df[price_df["Symbol"].str.upper() == "ETHUSDT"]

    end_ts = price_df["hour"].max()
    print(f"Using time range: {START_TS} to {end_ts}")

    price_df = price_df.set_index("hour").sort_index()
    price_df = price_df.loc[(price_df.index >= START_TS) & (price_df.index <= end_ts)]

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

    out = price_df.resample("6h").agg(price_agg_map).dropna(how="all").reset_index()

    rows_before_dropna = len(out)
    out = out.dropna().reset_index(drop=True)
    rows_after_dropna = len(out)
    print(f"Rows before dropna: {rows_before_dropna}")
    print(f"Rows after dropna:  {rows_after_dropna}")
    print(f"Rows dropped:       {rows_before_dropna - rows_after_dropna}")

    # Use next-interval low to represent downside risk in the next 6-hour window.
    out["future_price"] = out["Low"].shift(-1)
    out["future_drop_pct"] = (out["future_price"] - out["Close"]) / out["Close"]
    out["drawdown_6h_label"] = np.where(out["future_drop_pct"] <= -0.03, 1, 0)

    # Drop final row because there is no next 6-hour interval for target creation.
    out = out.dropna(subset=["future_price"]).reset_index(drop=True)

    out = out.drop(columns=["future_price", "future_drop_pct"])
    out["drawdown_6h_label"] = out["drawdown_6h_label"].astype(int)

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Final dataset shape: {out.shape}")
    print(out.head())

    return out


if __name__ == "__main__":
    df = build_dataset()
