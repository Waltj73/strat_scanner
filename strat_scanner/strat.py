# strat_scanner/strat.py
# STRAT logic helpers (no streamlit imports)

from __future__ import annotations

from typing import Optional
import pandas as pd


def _has_ohlc(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    cols = set(df.columns)
    return {"Open", "High", "Low", "Close"}.issubset(cols)


def _bar_type(df: pd.DataFrame) -> str:
    """
    Classify the most recent bar relative to the previous bar using High/Low.
    Returns one of: "1" (inside), "2U", "2D", "3" (outside), "NA"
    """
    if not _has_ohlc(df):
        return "NA"

    h = df["High"].dropna()
    l = df["Low"].dropna()
    if len(h) < 2 or len(l) < 2:
        return "NA"

    prev_high, prev_low = float(h.iloc[-2]), float(l.iloc[-2])
    last_high, last_low = float(h.iloc[-1]), float(l.iloc[-1])

    inside = (last_high <= prev_high) and (last_low >= prev_low)
    outside = (last_high > prev_high) and (last_low < prev_low)
    two_up = (last_high > prev_high) and (last_low >= prev_low)
    two_dn = (last_low < prev_low) and (last_high <= prev_high)

    if inside:
        return "1"
    if outside:
        return "3"
    if two_up:
        return "2U"
    if two_dn:
        return "2D"
    return "NA"


def best_trigger(df: pd.DataFrame, direction: Optional[str] = None) -> str:
    """
    Safe, compatible trigger function.

    ✅ Compatible with: best_trigger(df, direction=direction)
    ✅ df must be a DataFrame with OHLC (recommended).
    ✅ direction can be: "LONG", "SHORT", or None.
    """
    if df is None or df.empty:
        return "WAIT (No Data)"
    if not _has_ohlc(df):
        # Don't crash. Just return a stable label.
        return "WAIT (Missing OHLC)"

    bt = _bar_type(df)
    d = (direction or "").upper()
    if d not in ("LONG", "SHORT"):
        d = "NONE"

    # Inside bar logic: wait for break in direction
    if bt == "1":
        if d == "LONG":
            return "WAIT (Inside Bar) — break HIGH"
        if d == "SHORT":
            return "WAIT (Inside Bar) — break LOW"
        return "WAIT (Inside Bar)"

    # Outside bars are messy; require confirmation
    if bt == "3":
        return "WAIT (Outside Bar) — confirm"

    # Directional bar types
    if bt == "2U":
        return "READY (2U)"
    if bt == "2D":
        return "READY (2D)"

    # Fallback momentum read on Close (never crashes)
    c = df["Close"].dropna()
    if len(c) >= 3 and (c.iloc[-1] > c.iloc[-2] > c.iloc[-3]):
        return "READY (Momentum Up)"
    if len(c) >= 3 and (c.iloc[-1] < c.iloc[-2] < c.iloc[-3]):
        return "READY (Momentum Down)"

    return "WAIT"
