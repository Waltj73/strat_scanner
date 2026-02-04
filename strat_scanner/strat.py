from __future__ import annotations

from typing import Optional, Union, Tuple
import pandas as pd


def _as_ohlc(df_or_close: Union[pd.DataFrame, pd.Series]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Accept either:
      - DataFrame with Open/High/Low/Close (case-insensitive)
      - Series treated as Close (OHLC approximated to Close)
    Returns aligned (o, h, l, c)
    """
    if isinstance(df_or_close, pd.DataFrame):
        cols = {c.lower(): c for c in df_or_close.columns}
        ccol = cols.get("close")
        if ccol is None:
            raise ValueError("DataFrame missing Close")

        c = df_or_close[ccol].dropna()

        ocol = cols.get("open")
        hcol = cols.get("high")
        lcol = cols.get("low")

        o = df_or_close[ocol].reindex(c.index) if ocol else c.copy()
        h = df_or_close[hcol].reindex(c.index) if hcol else c.copy()
        l = df_or_close[lcol].reindex(c.index) if lcol else c.copy()

        o = o.dropna()
        h = h.dropna()
        l = l.dropna()

        idx = c.index.intersection(o.index).intersection(h.index).intersection(l.index)
        return o.loc[idx], h.loc[idx], l.loc[idx], c.loc[idx]

    c = pd.Series(df_or_close).dropna()
    o = c.copy()
    h = c.copy()
    l = c.copy()
    return o, h, l, c


def _bar_type(prev_h: float, prev_l: float, h: float, l: float) -> str:
    """
    STRAT bar types:
      1  inside
      2U directional up
      2D directional down
      3  outside
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

    # overlap fallback
    if h > prev_h:
        return "2U"
    if l < prev_l:
        return "2D"
    return "1"


def strat_bar_types(df_or_close: Union[pd.DataFrame, pd.Series]) -> Tuple[str, str]:
    """
    Returns (prev_bar_type, last_bar_type) using last 3 bars.
    Safe for DF or Series.
    """
    try:
        _, h, l, c = _as_ohlc(df_or_close)
    except Exception:
        return ("n/a", "n/a")

    if len(c) < 3:
        return ("n/a", "n/a")

    # bar[-2] relative to bar[-3]
    bt_prev = _bar_type(float(h.iloc[-3]), float(l.iloc[-3]), float(h.iloc[-2]), float(l.iloc[-2]))
    # bar[-1] relative to bar[-2]
    bt_last = _bar_type(float(h.iloc[-2]), float(l.iloc[-2]), float(h.iloc[-1]), float(l.iloc[-1]))
    return (bt_prev, bt_last)


def best_trigger(
    df_or_close: Union[pd.DataFrame, pd.Series],
    direction: Optional[str] = None,
) -> str:
    """
    Robust trigger:
      - Accepts DataFrame or Series
      - direction may be "LONG" or "SHORT" (optional)
    Returns a stable string.
    """
    try:
        _, h, l, c = _as_ohlc(df_or_close)
    except Exception:
        return "WAIT (No Data)"

    if len(c) < 3:
        return "WAIT (Not enough bars)"

    bt_prev, bt_last = strat_bar_types(df_or_close)

    # Basic "fire" logic
    bias = None
    label = None

    # 2-1 setup (needs confirmation next candle, but still useful)
    if bt_prev in ("2U", "2D") and bt_last == "1":
        bias = "LONG" if bt_prev == "2U" else "SHORT"
        label = f"2-1 Setup ({bias})"

    # 1-2 fire
    if bt_prev == "1" and bt_last in ("2U", "2D"):
        bias = "LONG" if bt_last == "2U" else "SHORT"
        label = f"1-2 Fire ({bias})"

    # Fallback momentum
    if label is None:
        if float(c.iloc[-1]) > float(c.iloc[-2]) > float(c.iloc[-3]):
            bias = "LONG"
            label = "Momentum Up (Fallback)"
        elif float(c.iloc[-1]) < float(c.iloc[-2]) < float(c.iloc[-3]):
            bias = "SHORT"
            label = "Momentum Down (Fallback)"
        else:
            return f"WAIT (Bars: {bt_prev},{bt_last})"

    # Direction filter
    if direction in ("LONG", "SHORT") and bias != direction:
        return f"WAIT ({label} filtered)"

    return f"READY [{bias}] — {label} (Bars: {bt_prev},{bt_last})"
