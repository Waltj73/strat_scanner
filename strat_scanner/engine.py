# strat_scanner/engine.py
# Core analysis engine (NO streamlit imports here except inside writeup_block)

from __future__ import annotations
from typing import Optional, Dict

import math
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


def analyze_ticker(
    ticker: str,
    spy_close: pd.Series,
    rs_short: int,
    rs_long: int,
    ema_trend_len: int,
    rsi_len: int,
    direction: str = "LONG",
) -> Optional[Dict]:
    """
    Returns a stable dict used by Dashboard / Scanner / Analyzer.
    direction: "LONG" or "SHORT"
    """
    df = get_hist(ticker)
    if df is None or df.empty:
        return None

    # Require OHLC
    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            return None

    close = df["Close"].dropna()
    if close.empty:
        return None

    # Need enough history for RS lookbacks
    if len(close) < (rs_long + 10) or len(spy_close) < (rs_long + 10):
        return None

    # --- Core metrics ---
    rs_s = float(rs_vs_spy(close, spy_close, int(rs_short)).iloc[-1])
    rs_l = float(rs_vs_spy(close, spy_close, int(rs_long)).iloc[-1])
    rot = rs_s - rs_l

    # ✅ FIX: define trend BEFORE using it
    tr = trend_label(close, int(ema_trend_len))
    rsi = float(rsi_wilder(close, int(rsi_len)).iloc[-1])

    strength = int(strength_meter(rs_s, rot, tr))
    meter = strength_label(strength)

    # --- STRAT trigger (expects DataFrame) ---
  # --- STRAT trigger (handle older/newer versions safely) ---
try:
    trigger = best_trigger(df, direction=direction)   # NEW signature
except TypeError:
    try:
        trigger = best_trigger(df)                    # older: best_trigger(df)
    except TypeError:
        trigger = best_trigger(close)                 # oldest: best_trigger(close)


    # Optional: provide basic entry/stop if best_trigger returns levels
    # We support either:
    # - a string status ("READY"/"WAIT ...")
    # - a dict like {"Status": "...", "TF": "D/W", "Entry": x, "Stop": y}
    tf = None
    entry = None
    stop = None
    trigger_status = None

    if isinstance(trigger, dict):
        trigger_status = trigger.get("Status") or trigger.get("TriggerStatus") or "WAIT"
        tf = trigger.get("TF")
        entry = trigger.get("Entry")
        stop = trigger.get("Stop")
    else:
        trigger_status = str(trigger)
        tf = None
        entry = None
        stop = None

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
        "TF": tf,
        "Entry": entry,
        "Stop": stop,
        "DailyDF": df,
    }


def writeup_block(info: Dict, pb_low: float, pb_high: float) -> None:
    """
    Streamlit UI helper (kept here so pages stay thinner).
    If you moved this to ui.py, that's fine too—just import from the right place.
    """
    import streamlit as st
    from strat_scanner.indicators import pullback_zone_ok

    st.markdown(f"### {info['Ticker']} — {info['Meter']} ({info['Strength']}/100)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trend", info.get("Trend", "n/a"))
    c2.metric("RSI", f"{float(info.get('RSI', 50)):.1f}")
    c3.metric("RS short", f"{float(info.get('RS_short', 0.0)):.2%}")
    c4.metric("Rotation", f"{float(info.get('Rotation', 0.0)):.2%}")

    pb_ok = pullback_zone_ok(info.get("Trend", ""), float(info.get("RSI", 50)), pb_low, pb_high)
    st.write(f"Pullback zone ({pb_low}–{pb_high}) OK? **{'YES ✅' if pb_ok else 'NO ❌'}**")

    st.write(f"Trigger: **{info.get('TriggerStatus','WAIT')}**"
             + (f" | TF: **{info.get('TF')}**" if info.get("TF") else ""))

    if info.get("Entry") is not None and info.get("Stop") is not None:
        st.write(f"Entry: **{float(info['Entry']):.2f}**  | Stop: **{float(info['Stop']):.2f}**")
    else:
        st.caption("No Inside Bar levels printed yet (WAIT).")
