from __future__ import annotations
from typing import Optional
import pandas as pd
import yfinance as yf

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns multi-index columns (ex: ('Close','AAPL')).
    Normalize to single level columns: Open, High, Low, Close, Volume, Adj Close
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # MultiIndex -> flatten
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] for c in out.columns]

    # Ensure standard expected columns exist (yfinance casing varies)
    rename = {}
    for c in out.columns:
        cc = str(c).strip()
        if cc.lower() == "adj close":
            rename[c] = "Adj Close"
        elif cc.lower() == "close":
            rename[c] = "Close"
        elif cc.lower() == "open":
            rename[c] = "Open"
        elif cc.lower() == "high":
            rename[c] = "High"
        elif cc.lower() == "low":
            rename[c] = "Low"
        elif cc.lower() == "volume":
            rename[c] = "Volume"
    if rename:
        out = out.rename(columns=rename)

    # Convert numeric columns safely (only if they are Series)
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Close"])
    return out

def get_hist(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Returns dataframe with at least a 'Close' column.
    """
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = _flatten_yf_columns(raw)
        return df
    except Exception:
        return pd.DataFrame()
