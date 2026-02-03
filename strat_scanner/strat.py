# strat_scanner/strat.py
# STRAT trigger logic (safe + compatible)

from __future__ import annotations
from typing import Optional
import pandas as pd


def best_trigger(df: pd.DataFrame, direction: Optional[str] = None) -> str:
    """
    Compatible with engine.py calls like:
        best_trigger(df, direction=direction)

    Requires OHLC. If missing, returns a safe WAIT string (never crashes).
    """

    if df is None or df.empty:
        return "WAIT (No Data)"

    # Must have OHLC
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(df.columns)):
        return "WAIT (Missing OHLC)"

    # Normalize direction
    d = (direction or "").upper()
    if d not in ("LONG", "SHORT"):
        d = "NONE"

    # Need at least 2 bars
    if len(df) < 2:
        return "WAIT (Not Enough Bars)"

    prev = df.iloc[-2]
    cur = df.iloc[-1]

    prev_high, prev_low = float(prev["High"]), float(prev["Low"])
    cur_high, cur_low = float(cur["High"]), float(cur["Low"])

    # STRAT bar types
    inside = (cur_high <= prev_high) and (cur_low >= prev_low)     # 1
    outside = (cur_high > prev_high) and (cur_low < prev_low)      # 3
    two_up = (cur_high > prev_high) and (cur_low >= prev_low)      # 2U
    two_dn = (cur_low < prev_low) and (cur_high <= prev_high)      # 2D

    if inside:
        if d == "LONG":
            return "WAIT (Inside Bar) — break HIGH"
        if d == "SHORT":
            return "WAIT (Inside Bar) — break LOW"
        return "WAIT (Inside Bar)"

    if outside:
        return "WAIT (Outside Bar) — confirm"

    if two_up:
        return "READY (2U)"

    if two_dn:
        return "READY (2D)"

    # Fallback (should rarely hit)
    return "WAIT"
