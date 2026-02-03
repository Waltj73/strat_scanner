from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df


def _safe_series(x) -> pd.Series:
    """
    If yfinance returns a DataFrame for a column (dup cols / multiindex weirdness),
    take the first column and return a Series.
    """
    if isinstance(x, pd.DataFrame):
        # pick the first column safely
        if x.shape[1] == 0:
            return pd.Series(dtype="float64")
        return x.iloc[:, 0]
    if isinstance(x, pd.Series):
        return x
    # fallback
    return pd.Series(x)


def _flatten_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_datetime_index(df)
    if df.empty:
        return pd.DataFrame()

    # Handle MultiIndex returned by yfinance (common when group_by="column")
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)

        # Case A: OHLCV at level 0, tickers at level 1
        if set(REQUIRED_COLS).issubset(set(lvl0)):
            if ticker in set(lvl1):
                df = df.xs(ticker, axis=1, level=1, drop_level=True)
            else:
                df.columns = [c[0] for c in df.columns]

        # Case B: OHLCV at level 1, tickers at level 0
        elif set(REQUIRED_COLS).issubset(set(lvl1)):
            if ticker in set(lvl0):
                df = df.xs(ticker, axis=1, level=0, drop_level=True)
            else:
                df.columns = [c[1] for c in df.columns]

        # Case C: unknown MultiIndex format
        else:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Normalize names to standard OHLCV
    rename_map = {}
    for c in df.columns:
        if not isinstance(c, str):
            continue
        lc = c.lower()
        if lc == "open":
            rename_map[c] = "Open"
        elif lc == "high":
            rename_map[c] = "High"
        elif lc == "low":
            rename_map[c] = "Low"
        elif lc in ("close", "adj close", "adj_close", "adjclose"):
            # prefer Close if not already present
            rename_map[c] = "Close" if "Close" not in df.columns else c
        elif lc == "volume":
            rename_map[c] = "Volume"

    if rename_map:
        df = df.rename(columns=rename_map)

    # If Close still missing, try common alt label
    if "Close" not in df.columns:
        for alt in ["Adj Close", "adj close", "Adj_Close", "AdjClose"]:
            if alt in df.columns:
                df["Close"] = df[alt]
                break

    # If Volume missing (some indexes), create it
    if "Volume" not in df.columns:
        df["Volume"] = 0

    # Keep only needed cols if present
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(set(df.columns)):
        return pd.DataFrame()

    df = df[needed].copy()
    df = _dedupe_columns(df)

    # ✅ CRITICAL FIX:
    # Ensure each column is a Series BEFORE calling to_numeric
    for c in needed:
        df[c] = _safe_series(df[c])
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_hist(ticker: str, period: str = "3y") -> pd.DataFrame:
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    return _flatten_yf_columns(raw, ticker)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_datetime_index(df)
    df = _dedupe_columns(df)
    if df.empty:
        return pd.DataFrame()

    # make sure numeric + series
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = _safe_series(df[c])
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return pd.DataFrame()

    def safe_first(x):
        x = x.dropna()
        return x.iloc[0] if len(x) else np.nan

    def safe_last(x):
        x = x.dropna()
        return x.iloc[-1] if len(x) else np.nan

    g = df.resample(rule)
    out = pd.DataFrame({
        "Open": g["Open"].apply(safe_first),
        "High": g["High"].max(),
        "Low": g["Low"].min(),
        "Close": g["Close"].apply(safe_last),
        "Volume": g["Volume"].sum(),
    }).dropna(subset=["Open", "High", "Low", "Close"])

    return out
