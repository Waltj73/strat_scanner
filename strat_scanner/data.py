from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class HistParams:
    period: str = "2y"
    interval: str = "1d"
    auto_adjust: bool = False


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure output has standard columns: Open, High, Low, Close, Volume.
    Handles yfinance MultiIndex columns, weird naming, etc.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # If yfinance returns multiindex columns (e.g., ('Close', 'SPY'))
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]

    # Standardize column names
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "open":
            rename_map[c] = "Open"
        elif cl == "high":
            rename_map[c] = "High"
        elif cl == "low":
            rename_map[c] = "Low"
        elif cl == "close":
            rename_map[c] = "Close"
        elif cl == "adj close":
            rename_map[c] = "Close"
        elif cl == "volume":
            rename_map[c] = "Volume"

    df = df.rename(columns=rename_map)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            # allow missing volume sometimes (indices)
            if c == "Volume":
                df[c] = 0.0
            else:
                return pd.DataFrame()

    out = df[needed].copy()
    out = out.dropna(subset=["Close"])
    out.index = pd.to_datetime(out.index)
    return out


@lru_cache(maxsize=512)
def _download_hist(ticker: str, period: str, interval: str, auto_adjust: bool) -> bytes:
    """
    Cached raw download. We return bytes (parquet) to keep lru_cache stable.
    """
    t = ticker.strip().upper()
    raw = yf.download(
        t,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
        group_by="column",
    )

    raw = _normalize_ohlcv(raw)
    if raw.empty:
        return b""

    # serialize to parquet in-memory
    return raw.to_parquet(index=True)


def get_hist(ticker: str, params: Optional[HistParams] = None) -> pd.DataFrame:
    """
    Public function used everywhere.
    Returns a clean OHLCV DataFrame or empty DataFrame.
    """
    params = params or HistParams()
    blob = _download_hist(ticker.strip().upper(), params.period, params.interval, params.auto_adjust)
    if not blob:
        return pd.DataFrame()
    return pd.read_parquet(pd.io.common.BytesIO(blob))
