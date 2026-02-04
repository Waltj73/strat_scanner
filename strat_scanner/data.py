# strat_scanner/data.py — data fetch + resampling (cloud-safe)

from __future__ import annotations

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

def _flatten_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_datetime_index(df)
    if df.empty:
        return pd.DataFrame()

    # yfinance sometimes returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)

        if set(REQUIRED_COLS).issubset(set(lvl0)):
            # columns like ("Open", "AAPL")
            if ticker in set(lvl1):
                df = df.xs(ticker, axis=1, level=1, drop_level=True)
            else:
                df.columns = [c[0] for c in df.columns]
        elif set(REQUIRED_COLS).issubset(set(lvl1)):
            # columns like ("AAPL", "Open")
            if ticker in set(lvl0):
                df = df.xs(ticker, axis=1, level=0, drop_level=True)
            else:
                df.columns = [c[1] for c in df.columns]
        else:
            df.columns = [c[0] for c in df.columns]

    # normalize column names
    rename = {}
    for c in df.columns:
        if not isinstance(c, str):
            continue
        lc = c.lower()
        if lc == "open":
            rename[c] = "Open"
        elif lc == "high":
            rename[c] = "High"
        elif lc == "low":
            rename[c] = "Low"
        elif lc in ("close", "adj close", "adjclose", "adj_close"):
            rename[c] = "Close"
        elif lc == "volume":
            rename[c] = "Volume"
    if rename:
        df = df.rename(columns=rename)

    # ensure required cols exist
    if "Close" not in df.columns:
        return pd.DataFrame()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[[c for c in REQUIRED_COLS if c in df.columns]].copy()
    if not set(["Open", "High", "Low", "Close"]).issubset(df.columns):
        return pd.DataFrame()

    for c in REQUIRED_COLS:
        if c in df.columns:
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
    if df.empty:
        return pd.DataFrame()

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
