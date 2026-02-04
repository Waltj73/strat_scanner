from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from strat_scanner.data import get_hist
from strat_scanner.indicators import (
    rsi_wilder,
    rs_vs_spy,
    trend_label,
    strength_meter,
    strength_label,
)
from strat_scanner.strat import best_trigger, strat_flags


def analyze_ticker(
    ticker: str,
    spy_close: pd.Series,
    rs_short: int,
    rs_long: int,
    ema_trend_len: int,
    rsi_len: int,
) -> Optional[Dict]:
    df = get_hist(ticker)
    if df is None or df.empty or "Close" not in df.columns:
        return None

    close = df["Close"].dropna()
    if close.empty:
        return None

    # Ensure enough history for returns
    if len(close) < (rs_long + 10) or len(spy_close) < (rs_long + 10):
        return None

    rs_s = float(rs_vs_spy(close, spy_close, int(rs_short)).iloc[-1])
    rs_l = float(rs_vs_spy(close, spy_close, int(rs_long)).iloc[-1])
    rot = rs_s - rs_l

    tr = trend_label(close, int(ema_trend_len))
    rsi = float(rsi_wilder(close, int(rsi_len)).iloc[-1])

    strength = int(strength_meter(rs_s, rot, tr))
    meter = strength_label(strength)

    # Strat trigger + flags (uses full DF safely)
    flags = strat_flags(df)
    trigger = best_trigger(df)

    # simple entry/stop scaffolding
    entry = float(close.iloc[-1])
    stop = float(df["Low"].rolling(20).min().iloc[-1]) if "Low" in df.columns and len(df) >= 20 else float(close.min())

    direction = "LONG" if tr == "UP" else "SHORT"

    return {
        "Ticker": ticker.upper(),
        "Strength": strength,
        "Meter": meter,
        "Trend": tr,
        "RSI": rsi,
        "RS_short": rs_s,
        "RS_long": rs_l,
        "Rotation": rot,
        "Direction": direction,
        "TriggerStatus": trigger,
        "Strat_Last": flags.get("Last", "n/a"),
        "Strat_Prev": flags.get("Prev", "n/a"),
        "TF": "D",
        "Entry": entry,
        "Stop": stop,
    }


def writeup_block(info: Dict, pb_low: float, pb_high: float):
    """
    Streamlit rendering helper for consistent writeups across pages.
    """
    import streamlit as st

    st.markdown(f"## {info['Ticker']} — {info['Meter']} ({info['Strength']}/100)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trend", info["Trend"])
    c2.metric("RSI", f"{info['RSI']:.1f}")
    c3.metric("RS short", f"{info['RS_short']:.2%}")
    c4.metric("Rotation", f"{info['Rotation']:.2%}")
    c5.metric("Bias", info.get("Direction", "n/a"))

    st.write(f"**STRAT:** Prev **{info.get('Strat_Prev','n/a')}** → Last **{info.get('Strat_Last','n/a')}**")
    st.write(f"**Trigger:** {info['TriggerStatus']}  | TF: **{info['TF']}**")

    st.write(f"**Entry (guide):** {info['Entry']:.2f}")
    st.write(f"**Stop (guide):** {info['Stop']:.2f}")

    if info["Trend"] == "UP" and (pb_low <= info["RSI"] <= pb_high):
        st.success(f"Pullback zone OK (RSI between {pb_low}–{pb_high})")
    else:
        st.info("Pullback zone not confirmed (or trend not UP).")
