from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

REQUIRED = ["Open","High","Low","Close","Volume"]

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
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

def _flatten(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    raw = _ensure_dt_index(raw)

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance sometimes returns MultiIndex
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

    rename = {}
    for c in raw.columns:
        if not isinstance(c, str):
            continue
        lc = c.lower()
        if lc == "open": rename[c] = "Open"
        elif lc == "high": rename[c] = "High"
        elif lc == "low": rename[c] = "Low"
        elif lc in ("close","adj close","adjclose","adj_close"):
            rename[c] = "Close" if "Close" not in raw.columns else c
        elif lc == "volume": rename[c] = "Volume"

    if rename:
        raw = raw.rename(columns=rename)

    if "Close" not in raw.columns and "Adj Close" in raw.columns:
        raw["Close"] = raw["Adj Close"]

    if "Volume" not in raw.columns:
        raw["Volume"] = 0

    if not set(["Open","High","Low","Close"]).issubset(set(raw.columns)):
        return pd.DataFrame()

    df = raw[REQUIRED].copy()
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open","High","Low","Close"])
    return df

@st.cache_data(ttl=60*30, show_spinner=False)
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
    return _flatten(raw)

