# strat_scanner/engine.py
# Core analysis engine (no page UI). Provides analyze_ticker() + writeup_block().

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from strat_scanner.data import get_hist
from strat_scanner.indicators import (
    rsi_wilder,
    rs_vs_spy,
    trend_label,
    strength_meter,
    strength_label,
    pullback_zone_ok,
)

# Optional: STRAT trigger logic (if present)
try:
    from strat_scanner.strat import best_trigger  # noqa
except Exception:
    best_trigger = None  # type: ignore


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _last(series: pd.Series, default: float = float("nan")) -> float:
    try:
        if series is None or len(series) == 0:
            return default
        return _safe_float(series.iloc[-1], default=default)
    except Exception:
        return default


def _calc_entry_stop(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Simple, robust defaults:
    - Entry = last close
    - Stop  = recent swing low (last 10 bars) minus a tiny buffer
    """
    try:
        c = df["Close"].dropna()
        l = df["Low"].dropna() if "Low" in df.columns else None
        if c.empty:
            return None, None

        entry = float(c.iloc[-1])
        if l is None or l.empty:
            return entry, None

        lookback = 10 if len(l) >= 10 else len(l)
        swing_low = float(l.iloc[-lookback:].min())
        stop = swing_low * 0.995  # small buffer
        return entry, stop
    except Exception:
        return None, None


def _trigger_status(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Returns (TriggerStatus, TF). This will not crash even if strat.py changes.
    """
    if best_trigger is None:
        return "n/a", "n/a"

    try:
        # We don't assume a strict signature—handle common patterns safely.
        # If your best_trigger expects just df:
        out = best_trigger(df)  # type: ignore
        if isinstance(out, tuple) and len(out) >= 2:
            return str(out[0]), str(out[1])
        return str(out), "n/a"
    except TypeError:
        # Some versions might want (df, tf) etc. Fallback gracefully.
        try:
            out = best_trigger(df, "W")  # type: ignore
            if isinstance(out, tuple) and len(out) >= 2:
                return str(out[0]), str(out[1])
            return str(out), "W"
        except Exception:
            return "n/a", "n/a"
    except Exception:
        return "n/a", "n/a"


def analyze_ticker(
    ticker: str,
    spy_close: pd.Series,
    rs_short: int,
    rs_long: int,
    ema_trend_len: int,
    rsi_len: int,
) -> Optional[Dict[str, Any]]:
    """
    Main unified output used across Scanner/Dashboard/Analyzer.

    Returns a dict with keys your pages expect:
    Ticker, Strength, Meter, Trend, RSI, RS_short, RS_long, Rotation,
    TriggerStatus, TF, Entry, Stop
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return None

    df = get_hist(ticker)
    if df is None or df.empty:
        return None

    if "Close" not in df.columns:
        return None

    close = df["Close"].dropna()
    if close.empty or len(close) < max(rs_long, ema_trend_len, rsi_len) + 5:
        return None

    # Align spy series to close index if needed
    spy = spy_close.dropna()
    if spy.empty or len(spy) < rs_long + 5:
        return None

    # RS metrics
    rs_s = _last(rs_vs_spy(close, spy, int(rs_short)))
    rs_l = _last(rs_vs_spy(close, spy, int(rs_long)))
    rot = rs_s - rs_l

    # Trend + RSI
    tr = trend_label(close, int(ema_trend_len))
    rsi_val = _last(rsi_wilder(close, int(rsi_len)), default=50.0)

    # Strength score
    score = int(strength_meter(rs_s, rot, tr))
    meter = strength_label(score)

    # Trigger + risk
    trig, tf = _trigger_status(df)
    entry, stop = _calc_entry_stop(df)

    return {
        "Ticker": ticker,
        "Strength": score,
        "Meter": meter,
        "Trend": tr,
        "RSI": float(rsi_val),
        "RS_short": float(rs_s),
        "RS_long": float(rs_l),
        "Rotation": float(rot),
        "TriggerStatus": trig,
        "TF": tf,
        "Entry": entry,
        "Stop": stop,
        "Last": float(close.iloc[-1]),
    }


def writeup_block(info: Dict[str, Any], pb_low: float = 40, pb_high: float = 55) -> None:
    """
    Small write-up renderer used by Dashboard/Analyzer.
    This function writes directly to Streamlit, so pages can call it.
    """
    import streamlit as st  # local import to avoid non-UI contexts breaking

    ticker = info.get("Ticker", "n/a")
    strength = info.get("Strength", "n/a")
    meter = info.get("Meter", "n/a")
    trend = info.get("Trend", "n/a")
    rsi = info.get("RSI", np.nan)
    rs_s = info.get("RS_short", np.nan)
    rot = info.get("Rotation", np.nan)
    trig = info.get("TriggerStatus", "n/a")
    tf = info.get("TF", "n/a")
    entry = info.get("Entry", None)
    stop = info.get("Stop", None)

    st.markdown(f"### {ticker} — **{meter} ({strength}/100)**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trend", str(trend))
    c2.metric("RSI", f"{_safe_float(rsi, 50.0):.1f}")
    c3.metric("RS vs SPY (short)", f"{_safe_float(rs_s, 0.0)*100:.2f}%")
    c4.metric("Rotation", f"{_safe_float(rot, 0.0)*100:.2f}%")

    st.write(f"**Trigger:** {trig}  |  **TF:** {tf}")

    in_zone = pullback_zone_ok(str(trend), _safe_float(rsi, 50.0), pb_low, pb_high)
    st.write(f"**Pullback Zone ({pb_low}–{pb_high}) in UP trend:** {'✅ YES' if in_zone else '❌ NO'}")

    if entry is not None:
        st.write(f"**Entry (default):** {float(entry):.2f}")
    if stop is not None:
        st.write(f"**Stop (default):** {float(stop):.2f}")

    st.caption("Note: Entry/Stop are default, robust placeholders. You can refine later.")
