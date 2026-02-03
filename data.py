# data.py — Data access & resampling layer
# Handles:
# - yfinance download
# - MultiIndex cleanup
# - datetime cleanup
# - OHLC standardization
# - caching
# - resampling (weekly/monthly)

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


# =========================
# INTERNAL HELPERS
# =========================
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure clean DatetimeIndex."""
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


def _flatten_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Flattens yfinance output so we always get:
    Open, High, Low, Close, Volume
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_datetime_index(df)

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)

        if ticker in set(lvl1):
            df = df.xs(ticker, axis=1, level=1, drop_level=True)
        elif ticker in set(lvl0):
            df = df.xs(ticker, axis=1, level=0, drop_level=True)
        else:
            df.columns = [c[0] for c in df.columns]

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
        elif lc in ("close", "adj close", "adj_close"):
            rename_map[c] = "Close"
        elif lc == "volume":
            rename_map[c] = "Volume"

    df = df.rename(columns=rename_map)

    # fallback for Close
    if "Close" not in df.columns:
        for alt in ["Adj Close", "adj close", "AdjClose"]:
            if alt in df.columns:
                df["Close"] = df[alt]

    if "Volume" not in df.columns:
        df["Volume"] = 0

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(df.columns):
        return pd.DataFrame()

    df = df[needed].copy()
    df = _dedupe_columns(df)

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


# =========================
# PUBLIC FUNCTIONS
# =========================
@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_hist(ticker: str, period: str = "3y") -> pd.DataFrame:
    """Download historical data safely."""
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
    """
    Resample daily OHLC to weekly/monthly.
    Example rules:
        "W-FRI"
        "M"
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_datetime_index(df)
    df = _dedupe_columns(df)

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return pd.DataFrame()

    g = df.resample(rule)

    out = pd.DataFrame({
        "Open": g["Open"].first(),
        "High": g["High"].max(),
        "Low": g["Low"].min(),
        "Close": g["Close"].last(),
        "Volume": g["Volume"].sum(),
    })

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out
