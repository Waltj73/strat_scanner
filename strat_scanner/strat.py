from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Tuple

import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _as_ohlc(df_or_close: Union[pd.DataFrame, pd.Series]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Accept either:
      - DataFrame with columns: Open, High, Low, Close (case-insensitive)
      - Series (treated as Close; High/Low approximated from Close)
    Returns: (o, h, l, c) as pd.Series
    """
    if isinstance(df_or_close, pd.DataFrame):
        cols = {c.lower(): c for c in df_or_close.columns}
        # Required close
        close_col = cols.get("close")
        if close_col is None:
            raise ValueError("DataFrame missing 'Close' column.")
        c = df_or_close[close_col].dropna()

        # Optional OHLC
        open_col = cols.get("open")
        high_col = cols.get("high")
        low_col = cols.get("low")

        if open_col is None:
            o = c.copy()
        else:
            o = df_or_close[open_col].reindex(c.index).dropna()

        if high_col is None or low_col is None:
            # Fallback: approximate (won't be perfect but prevents crashes)
            h = c.copy()
            l = c.copy()
        else:
            h = df_or_close[high_col].reindex(c.index).dropna()
            l = df_or_close[low_col].reindex(c.index).dropna()

        # Align all
        idx = c.index.intersection(o.index).intersection(h.index).intersection(l.index)
        return o.loc[idx], h.loc[idx], l.loc[idx], c.loc[idx]

    # Series path: treat as close-only
    c = pd.Series(df_or_close).dropna()
    # approximate OHLC to avoid KeyErrors
    o = c.copy()
    h = c.copy()
    l = c.copy()
    return o, h, l, c


def _bar_type(prev_h: float, prev_l: float, h: float, l: float) -> str:
    """
    Classic STRAT bar types:
      1  = inside
      2U = directional up
      2D = directional down
      3  = outside
    """
    inside = (h <= prev_h) and (l >= prev_l)
    outside = (h > prev_h) and (l < prev_l)

    if inside:
        return "1"
    if outside:
        return "3"
    # directional
    if h > prev_h and l >= prev_l:
        return "2U"
    if l < prev_l and h <= prev_h:
        return "2D"
    # weird overlap: pick based on which side broke
    if h > prev_h:
        return "2U"
    if l < prev_l:
        return "2D"
    return "1"


def _direction_from_bar(bt: str) -> Optional[str]:
    if bt == "2U":
        return "LONG"
    if bt == "2D":
        return "SHORT"
    return None


# -----------------------------
# Public API
# -----------------------------
def best_trigger(
    df_or_close: Union[pd.DataFrame, pd.Series],
    direction: Optional[str] = None,
) -> str:
    """
    Backward-compatible STRAT trigger function.

    Accepts:
      - DataFrame (preferred) with OHLC
      - Series (Close)

    direction:
      - None (no filter)
      - "LONG" or "SHORT" (filter triggers to that bias)

    Returns a single stable string you can display everywhere.
    """
    try:
        o, h, l, c = _as_ohlc(df_or_close)
    except Exception:
        return "WAIT (No Data)"

    if len(c) < 3:
        return "WAIT (Not enough bars)"

    # last two bars vs previous
    prev_h, prev_l = float(h.iloc[-2]), float(l.iloc[-2])
    curr_h, curr_l = float(h.iloc[-1]), float(l.iloc[-1])

    bt_prev = _bar_type(float(h.iloc[-3]), float(l.iloc[-3]), prev_h, prev_l)
    bt_curr = _bar_type(prev_h, prev_l, curr_h, curr_l)

    # Simple, practical triggers:
    # 2-1-2 continuation
    # Example: 2U-1-2U = bullish continuation (break of inside bar high in an up context)
    trigger = None
    bias = None

    if bt_prev in ("2U", "2D") and bt_curr == "1":
        # next bar break implied; we can still label "SETUP"
        if bt_prev == "2U":
            trigger = "2-1 (Bull) Setup"
            bias = "LONG"
        else:
            trigger = "2-1 (Bear) Setup"
            bias = "SHORT"

    # If current bar is 2U/2D after an inside bar, that's the "fire"
    # (This is a clean “take it when it goes” interpretation.)
    if bt_prev == "1" and bt_curr in ("2U", "2D"):
        bias = _direction_from_bar(bt_curr)
        trigger = f"1-2 Fire ({bt_curr})"

    # If we have 2-1-2 specifically over last 3 bars
    bt_1 = _bar_type(float(h.iloc[-4]), float(l.iloc[-4]), float(h.iloc[-3]), float(l.iloc[-3])) if len(c) >= 4 else None
    bt_2 = bt_prev
    bt_3 = bt_curr
    if bt_1 in ("2U", "2D") and bt_2 == "1" and bt_3 in ("2U", "2D"):
        if bt_3 == "2U":
            bias = "LONG"
            trigger = "2-1-2 Bull Continuation"
        elif bt_3 == "2D":
            bias = "SHORT"
            trigger = "2-1-2 Bear Continuation"

    # If nothing clean, use simple momentum fallback
    if trigger is None:
        if float(c.iloc[-1]) > float(c.iloc[-2]) > float(c.iloc[-3]):
            bias = "LONG"
            trigger = "Momentum Up (Fallback)"
        elif float(c.iloc[-1]) < float(c.iloc[-2]) < float(c.iloc[-3]):
            bias = "SHORT"
            trigger = "Momentum Down (Fallback)"
        else:
            return f"WAIT (Bars: {bt_prev},{bt_curr})"

    # Direction filter
    if direction in ("LONG", "SHORT") and bias is not None and bias != direction:
        return f"WAIT ({trigger} filtered)"

    return f"READY [{bias}] — {trigger} (Bars: {bt_prev},{bt_curr})"
