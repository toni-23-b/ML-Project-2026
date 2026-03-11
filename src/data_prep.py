"""Data preparation helpers for ETH drawdown classification."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def _find_column(columns: Iterable[str], candidates: list[str]) -> str:
    """Return the first matching column name (case-insensitive)."""
    normalized = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ValueError(f"None of the candidate columns were found: {candidates}")


def load_eth_price_data(path: str) -> pd.DataFrame:
    """Load ETH market data, parse timestamps, and return hourly indexed data.

    Expected columns (flexible names): timestamp/date/time, open, high, low, close,
    and optionally volume + symbol/asset column.
    """
    df = pd.read_csv(path)

    timestamp_col = _find_column(df.columns, ["timestamp", "datetime", "date", "time"])
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

    symbol_candidates = ["symbol", "asset", "coin", "ticker", "pair"]
    symbol_col = next((c for c in symbol_candidates if c in map(str.lower, df.columns)), None)
    if symbol_col is not None:
        actual_symbol_col = _find_column(df.columns, symbol_candidates)
        symbol_values = df[actual_symbol_col].astype(str).str.upper()
        df = df[symbol_values.str.contains("ETH", na=False)]

    df = df.set_index(timestamp_col)

    # If intrahour rows exist, aggregate to one hourly row.
    if not (df.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        agg_map = {}
        for col in df.columns:
            col_l = col.lower()
            if col_l == "open":
                agg_map[col] = "first"
            elif col_l == "high":
                agg_map[col] = "max"
            elif col_l == "low":
                agg_map[col] = "min"
            elif col_l == "close":
                agg_map[col] = "last"
            elif "vol" in col_l:
                agg_map[col] = "sum"
            else:
                agg_map[col] = "last"
        df = df.resample("1h").agg(agg_map)

    return df.dropna(how="all")


def load_whale_data(path: str) -> pd.DataFrame:
    """Load whale feature data and return an hourly indexed DataFrame."""
    df = pd.read_csv(path)
    timestamp_col = _find_column(df.columns, ["timestamp", "datetime", "date", "time", "hour"])
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    df = df.set_index(timestamp_col)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    # Sum numeric activity metrics per hour and keep latest categorical/context columns.
    hourly_numeric = df[numeric_cols].resample("1h").sum(min_count=1) if len(numeric_cols) else pd.DataFrame(index=df.resample("1h").size().index)
    hourly_non_numeric = df[non_numeric_cols].resample("1h").last() if non_numeric_cols else pd.DataFrame(index=hourly_numeric.index)

    hourly_df = pd.concat([hourly_numeric, hourly_non_numeric], axis=1)
    return hourly_df.dropna(how="all")


def merge_price_and_whale(price_df: pd.DataFrame, whale_df: pd.DataFrame) -> pd.DataFrame:
    """Merge hourly price and whale features on timestamp index."""
    merged = price_df.join(whale_df, how="left")

    whale_cols = [c for c in whale_df.columns if c in merged.columns]
    numeric_whale_cols = [c for c in whale_cols if pd.api.types.is_numeric_dtype(merged[c])]
    if numeric_whale_cols:
        merged[numeric_whale_cols] = merged[numeric_whale_cols].fillna(0)

    return merged.sort_index()


def add_drawdown_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target: 1 if price drops >=5% within next 2 hours, else 0.

    Label definition at hour t:
    future_low = min(low[t+1], low[t+2])
    label = 1 if future_low <= close[t] * 0.95 else 0
    """
    out = df.copy()

    close_col = _find_column(out.columns, ["close"])
    low_col = _find_column(out.columns, ["low"])

    future_low = pd.concat([
        out[low_col].shift(-1),
        out[low_col].shift(-2),
    ], axis=1).min(axis=1)

    out["future_low_2h"] = future_low
    out["drawdown_label"] = (out["future_low_2h"] <= out[close_col] * 0.95).fillna(False).astype(int)

    return out
