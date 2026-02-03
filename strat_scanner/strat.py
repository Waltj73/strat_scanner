# strat_scanner/strat.py
from __future__ import annotations
from typing import Optional
import pandas as pd


def best_trigger(df: pd.DataFrame, direction: Optional[str] = None) -> str:
    """
    STRAT-style trigger:
    - Uses last 2 bars to label Inside/2U/2D/3
    - Never crashes if OHLC missing (returns WAIT)
    """

    if df is None or df.empty:
        return "WAIT (No Data)"

    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(df.columns):
        return "WAIT (Missing OHLC)"

    if len(df) < 2:
        return "WAIT (Not Enough Bars)"

    d = (direction or "").upper()
    if d not in ("LONG", "SHORT"):
        d = "NONE"

    prev = df.iloc[-2]
    cur = df.iloc[-1]

    prev_high, prev_low = float(prev["High"]), float(prev["Low"])
    cur_high, cur_low = float(cur["High"]), float(cur["Low"])

    inside = (cur_high <= prev_high) and (cur_low >= prev_low)      # 1
    outside = (cur_high > prev_high) and (cur_low < prev_low)       # 3
    two_up = (cur_high > prev_high) and (cur_low >= prev_low)       # 2U
    two_dn = (cur_low < prev_low) and (cur_high <= prev_high)       # 2D

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
