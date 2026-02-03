# strat_scanner/engine.py
# Analysis engine that merges: data + indicators + STRAT trigger
# Keeps UI thin and stable.

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from strat_scanner.data import get_hist
from strat_scanner.indicators import (
    rsi_wilder,
    rs_vs_spy,
    trend_label,
    strength_meter,
    strength_label,
)
from strat_scanner.strat import best_trigger


def _safe_close(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if "Close" not in df.columns:
        return None
    close = df["Close"].dropna()
    if close.empty:
        return None
    return close


def analyze_ticker(
    ticker: str,
    spy_close: pd.Series,
    rs_short: int,
    rs_long: int,
    ema_trend_len: int,
    rsi_len: int,
    direction: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Returns a stable dict used by Scanner/Dashboard/Analyzer pages.
    """
    df = get_hist(ticker)
    close = _safe_close(df)
    if df is None or close is None:
        return None

    # Need enough history for RS lookbacks
    if spy_close is None or len(spy_close) < (rs_long + 10) or len(close) < (rs_long + 10):
        return None

    # Relative strength + rotation
    rs_s = float(rs_vs_spy(close, spy_close, int(rs_short)).iloc[-1])
    rs_l = float(rs_vs_spy(close, spy_close, int(rs_long)).iloc[-1])
    rot = rs_s - rs_l

    # Trend + RSI
    tr = trend_label(close, int(ema_trend_len))
    rsi = float(rsi_wilder(close, int(rsi_len)).iloc[-1])

    # Strength score
    strength = int(strength_meter(rs_s, rot, tr))
    meter = strength_label(strength)

    # Direction bias (optional)
    bias = (direction or ("LONG" if tr == "UP" else "SHORT")).upper()
    if bias not in ("LONG", "SHORT", "NONE"):
        bias = "NONE"

    # STRAT trigger (robust output dict)
    trig = best_trigger(df, direction=bias)

    # Entry / Stop: prefer STRAT values if present; fallback to basic
    last_price = float(close.iloc[-1])
    fallback_stop = float(close.rolling(20).min().iloc[-1]) if len(close) >= 20 else float(close.min())

    entry = trig.get("entry", None)
    stop = trig.get("stop", None)

    if entry is None or not np.isfinite(entry):
        entry = last_price
    if stop is None or not np.isfinite(stop):
        stop = fallback_stop

    trigger_status = f"{trig.get('status','WAIT')} — {trig.get('setup','No Setup')}"
    if trig.get("direction") in ("LONG", "SHORT"):
        trigger_status += f" ({trig['direction']})"

    return {
        "Ticker": ticker.upper(),
        "Strength": strength,
        "Meter": meter,
        "Trend": tr,
        "RSI": rsi,
        "RS_short": rs_s,
        "RS_long": rs_l,
        "Rotation": rot,
        "TriggerStatus": trigger_status,
        "TriggerNotes": trig.get("notes", ""),
        "TF": "D",
        "Entry": float(entry),
        "Stop": float(stop),
    }


def writeup_block(info: Dict[str, Any], pb_low: float, pb_high: float):
    """
    Streamlit rendering helper used by pages.
    (kept in engine so pages stay thin)
    """
    import streamlit as st

    st.markdown(f"### {info['Ticker']} — {info['Meter']} ({info['Strength']}/100)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trend", info.get("Trend", "n/a"))
    c2.metric("RSI", f"{info.get('RSI', 0):.1f}" if isinstance(info.get("RSI", None), (int, float)) else "n/a")
    c3.metric("RS short", f"{info.get('RS_short', 0):.2%}" if isinstance(info.get("RS_short", None), (int, float)) else "n/a")
    c4.metric("Rotation", f"{info.get('Rotation', 0):.2%}" if isinstance(info.get("Rotation", None), (int, float)) else "n/a")

    st.write(f"Trigger: **{info.get('TriggerStatus','WAIT')}**")
    notes = info.get("TriggerNotes", "")
    if notes:
        st.caption(notes)

    st.write(f"Entry (guide): **{info.get('Entry', float('nan')):.2f}**")
    st.write(f"Stop (guide): **{info.get('Stop', float('nan')):.2f}**")

    # Pullback filter helper
    try:
        rsi_val = float(info.get("RSI", np.nan))
        trend = info.get("Trend", "")
        if trend == "UP" and pb_low <= rsi_val <= pb_high:
            st.success(f"Pullback zone OK (RSI between {pb_low}–{pb_high})")
        else:
            st.info("Pullback zone not confirmed (or trend not UP).")
    except Exception:
        st.info("Pullback zone not confirmed (or missing RSI).")
