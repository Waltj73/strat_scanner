from __future__ import annotations

from typing import Optional, Dict

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
) -> Optional[Dict]:
    """
    Single source of truth used by:
      - scanner page
      - dashboard page
      - analyzer page
    Must remain stable.
    """
    df = get_hist(ticker)
    if df is None or df.empty or "Close" not in df.columns:
        return None

    close = df["Close"].dropna()
    if close.empty:
        return None

    # Ensure enough history for RS calcs
    if len(close) < (rs_long + 10) or len(spy_close) < (rs_long + 10):
        return None

    rs_s = float(rs_vs_spy(close, spy_close, int(rs_short)).iloc[-1])
    rs_l = float(rs_vs_spy(close, spy_close, int(rs_long)).iloc[-1])
    rot = rs_s - rs_l

    tr = trend_label(close, int(ema_trend_len))  # <-- fixes NameError: tr
    rsi = float(rsi_wilder(close, int(rsi_len)).iloc[-1])

    strength = int(strength_meter(rs_s, rot, tr))
    meter = strength_label(strength)

    # Strat trigger (direction filter optional)
    direction = "LONG" if tr == "UP" else ("SHORT" if tr == "DOWN" else None)
    trigger_status = best_trigger(df, direction=direction)  # accepts df + direction safely now

    # Basic entry/stop scaffolding (non-magical, stable)
    entry = float(close.iloc[-1])
    if len(close) >= 20:
        stop = float(close.rolling(20).min().iloc[-1])
    else:
        stop = float(close.min())

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
        "TF": "D",
        "Entry": entry,
        "Stop": stop,
    }


def writeup_block(info: Dict, pb_low: float = 40.0, pb_high: float = 60.0):
    """
    Used by pages/analyzer.py (your logs show analyzer imports this). :contentReference[oaicite:2]{index=2}
    Kept here so pages stay thin and imports stay consistent.
    """
    import streamlit as st

    st.markdown(f"### {info['Ticker']} — {info['Meter']} ({info['Strength']}/100)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trend", info["Trend"])
    c2.metric("RSI", f"{info['RSI']:.1f}")
    c3.metric("RS short", f"{info['RS_short']:.2%}")
    c4.metric("Rotation", f"{info['Rotation']:.2%}")

    st.write(f"Trigger: **{info['TriggerStatus']}**  | TF: **{info['TF']}**")
    st.write(f"Entry (guide): **{info['Entry']:.2f}**")
    st.write(f"Stop (guide): **{info['Stop']:.2f}**")

    # Simple pullback zone helper (optional)
    if info["Trend"] == "UP" and (pb_low <= info["RSI"] <= pb_high):
        st.success(f"Pullback zone OK (RSI between {pb_low}–{pb_high})")
    elif info["Trend"] == "DOWN" and (pb_low <= info["RSI"] <= pb_high):
        st.info("Downtrend + mid RSI (watch for bear continuation triggers).")
    else:
        st.info("Pullback zone not confirmed.")
