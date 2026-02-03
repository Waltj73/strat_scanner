# strat_scanner/strat.py
# STRAT candle classification + setups + triggers
# NO streamlit imports. Engine/UI consume best_trigger() output.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class StratTrigger:
    status: str            # "READY" | "WAIT" | "AVOID"
    setup: str             # e.g. "Inside Bar", "2-1-2", "3 breakout", "No Setup"
    direction: str         # "LONG" | "SHORT" | "NONE"
    entry: Optional[float] # suggested trigger entry (break level)
    stop: Optional[float]  # suggested stop (other side / pattern invalidation)
    notes: str             # short explanation


def _safe_ohlc(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return cleaned df with required OHLC columns or None."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    needed = ["Open", "High", "Low", "Close"]
    for c in needed:
        if c not in df.columns:
            return None
    out = df[needed].copy()
    out = out.dropna()
    if len(out) < 3:
        return None
    return out


def strat_candle_type(prev_high: float, prev_low: float, high: float, low: float) -> str:
    """
    The STRAT:
    1  = inside bar  (high <= prev_high AND low >= prev_low)
    2U = higher high only (high > prev_high AND low >= prev_low)
    2D = lower low only  (low < prev_low AND high <= prev_high)
    3  = outside bar (high > prev_high AND low < prev_low)
    """
    if high <= prev_high and low >= prev_low:
        return "1"
    if high > prev_high and low >= prev_low:
        return "2U"
    if low < prev_low and high <= prev_high:
        return "2D"
    if high > prev_high and low < prev_low:
        return "3"
    # fallback (shouldn't happen often)
    return "?"


def _infer_direction_from_trend(close: pd.Series) -> str:
    """
    Light directional bias helper when caller doesn't supply direction.
    """
    if close is None or len(close) < 5:
        return "NONE"
    # simple slope check
    if close.iloc[-1] > close.iloc[-3]:
        return "LONG"
    if close.iloc[-1] < close.iloc[-3]:
        return "SHORT"
    return "NONE"


def best_trigger(df: pd.DataFrame, direction: Optional[str] = None) -> Dict[str, Any]:
    """
    Core STRAT trigger generator.
    Returns a dict (stable keys) for UI/engine.

    direction: Optional "LONG"/"SHORT" bias. If None, inferred from recent price drift.
    """
    ohlc = _safe_ohlc(df)
    if ohlc is None:
        return StratTrigger(
            status="WAIT",
            setup="No Data",
            direction="NONE",
            entry=None,
            stop=None,
            notes="Not enough OHLC data to evaluate STRAT setups.",
        ).__dict__

    # Use last 3 bars to evaluate
    prev = ohlc.iloc[-2]
    last = ohlc.iloc[-1]

    prev_high, prev_low = float(prev["High"]), float(prev["Low"])
    last_high, last_low = float(last["High"]), float(last["Low"])

    last_type = strat_candle_type(prev_high, prev_low, last_high, last_low)

    close = ohlc["Close"]
    bias = (direction or _infer_direction_from_trend(close)).upper()
    if bias not in ("LONG", "SHORT", "NONE"):
        bias = "NONE"

    # --- Inside bar logic ---
    if last_type == "1":
        # Inside bar: breakout levels are prev high/low
        entry_long = prev_high
        entry_short = prev_low
        stop_long = prev_low
        stop_short = prev_high

        if bias == "LONG":
            return StratTrigger(
                status="WAIT",
                setup="Inside Bar (1) — wait for break",
                direction="LONG",
                entry=entry_long,
                stop=stop_long,
                notes="Bias LONG: trigger is break above prior bar high; stop below prior bar low.",
            ).__dict__
        if bias == "SHORT":
            return StratTrigger(
                status="WAIT",
                setup="Inside Bar (1) — wait for break",
                direction="SHORT",
                entry=entry_short,
                stop=stop_short,
                notes="Bias SHORT: trigger is break below prior bar low; stop above prior bar high.",
            ).__dict__

        return StratTrigger(
            status="WAIT",
            setup="Inside Bar (1) — wait for break",
            direction="NONE",
            entry=None,
            stop=None,
            notes="Inside bar detected. Choose LONG (above prior high) or SHORT (below prior low).",
        ).__dict__

    # --- Outside bar logic (3) ---
    if last_type == "3":
        # Outside bar is volatile; prefer WAIT unless bias is clear
        if bias == "LONG":
            return StratTrigger(
                status="WAIT",
                setup="Outside Bar (3) — volatile",
                direction="LONG",
                entry=last_high,
                stop=last_low,
                notes="Outside bar: volatility spike. Prefer confirmation. If long, trigger is break above this bar high.",
            ).__dict__
        if bias == "SHORT":
            return StratTrigger(
                status="WAIT",
                setup="Outside Bar (3) — volatile",
                direction="SHORT",
                entry=last_low,
                stop=last_high,
                notes="Outside bar: volatility spike. Prefer confirmation. If short, trigger is break below this bar low.",
            ).__dict__

        return StratTrigger(
            status="WAIT",
            setup="Outside Bar (3) — volatile",
            direction="NONE",
            entry=None,
            stop=None,
            notes="Outside bar detected. Wait for next candle to give direction.",
        ).__dict__

    # --- Directional continuation bars (2U / 2D) ---
    if last_type == "2U":
        # Bullish expansion
        return StratTrigger(
            status="READY" if bias in ("LONG", "NONE") else "WAIT",
            setup="2U (bullish expansion)",
            direction="LONG" if bias != "SHORT" else "NONE",
            entry=last_high,
            stop=prev_low,
            notes="2U suggests upside continuation. Trigger break above this bar high; invalidation below prior bar low.",
        ).__dict__

    if last_type == "2D":
        # Bearish expansion
        return StratTrigger(
            status="READY" if bias in ("SHORT", "NONE") else "WAIT",
            setup="2D (bearish expansion)",
            direction="SHORT" if bias != "LONG" else "NONE",
            entry=last_low,
            stop=prev_high,
            notes="2D suggests downside continuation. Trigger break below this bar low; invalidation above prior bar high.",
        ).__dict__

    # fallback
    return StratTrigger(
        status="WAIT",
        setup=f"Unknown ({last_type})",
        direction="NONE",
        entry=None,
        stop=None,
        notes="Could not classify last candle cleanly.",
    ).__dict__
