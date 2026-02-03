# strat_scanner/strat.py
from __future__ import annotations

from typing import Optional, Union
import pandas as pd


def best_trigger(data: Union[pd.DataFrame, pd.Series], direction: Optional[str] = None) -> str:
    """
    Robust STRAT trigger:
    - Accepts either OHLC DataFrame OR Close Series.
    - Accepts direction keyword safely ("LONG"/"SHORT"/None).
    - Never crashes: returns WAIT labels if data missing.
    """

    if data is None:
        return "WAIT (No Data)"

    d = (direction or "").upper().strip()
    if d not in ("LONG", "SHORT"):
        d = "NONE"

    # --- Case 1: Series (Close only) fallback ---
    if isinstance(data, pd.Series):
        close = data.dropna()
        if close.empty or len(close) < 3:
            return "WAIT (No Data)"
        if close.iloc[-1] > close.iloc[-2] > close.iloc[-3]:
            return "READY (Momentum)"
        return "WAIT"

    # --- Case 2: DataFrame (preferred) ---
    df = data
    if df.empty:
        return "WAIT (No Data)"

    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(df.columns):
        # If we only have Close, degrade gracefully
        if "Close" in df.columns:
            return best_trigger(df["Close"], direction=direction)
        return "WAIT (Missing OHLC)"

    if len(df) < 2:
        return "WAIT (Not Enough Bars)"

    prev = df.iloc[-2]
    cur = df.iloc[-1]

    prev_high, prev_low = float(prev["High"]), float(prev["Low"])
    cur_high, cur_low = float(cur["High"]), float(cur["Low"])

    inside = (cur_high <= prev_high) and (cur_low >= prev_low)   # 1
    outside = (cur_high > prev_high) and (cur_low < prev_low)    # 3
    two_up = (cur_high > prev_high) and (cur_low >= prev_low)    # 2U
    two_dn = (cur_low < prev_low) and (cur_high <= prev_high)    # 2D

    if inside:
        if d == "LONG":
            return "WAIT (Inside) — break HIGH"
        if d == "SHORT":
            return "WAIT (Inside) — break LOW"
        return "WAIT (Inside)"

    if outside:
        return "WAIT (Outside) — confirm"

    if two_up:
        return "READY (2U)"

    if two_dn:
        return "READY (2D)"

    return "WAIT"
