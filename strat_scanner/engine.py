# strat_scanner/engine.py
from __future__ import annotations
from typing import Optional, Dict

import numpy as np
import pandas as pd

from strat_scanner.data import get_hist
from strat_scanner.indicators import rsi_wilder, rs_vs_spy, trend_label, strength_meter, strength_label
from strat_scanner.strat import best_trigger, targets_from_entry, _norm_cols  # _norm_cols is safe internal helper


def analyze_ticker(
    ticker: str,
    spy_close: pd.Series,
    rs_short: int,
    rs_long: int,
    ema_trend_len: int,
    rsi_len: int,
) -> Optional[Dict]:
    df = get_hist(ticker)
    if df is None or df.empty:
        return None

    d = _norm_cols(df)  # guarantees Open/High/Low/Close
    if d.empty or len(d) < (rs_long + 10) or spy_close is None or len(spy_close) < (rs_long + 10):
        return None

    close = d["Close"]

    rs_s = float(rs_vs_spy(close, spy_close, int(rs_short)).iloc[-1])
    rs_l = float(rs_vs_spy(close, spy_close, int(rs_long)).iloc[-1])
    rot = rs_s - rs_l

    tr = trend_label(close, int(ema_trend_len))
    rsi = float(rsi_wilder(close, int(rsi_len)).iloc[-1])

    strength = int(strength_meter(rs_s, rot, tr))
    meter = strength_label(strength)

    trigger = best_trigger(df)

    setup = trig["Setup"]
    direction = trig["Direction"]
    status = trig["Status"]
    entry = float(trig["Entry"]) if np.isfinite(trig["Entry"]) else float(close.iloc[-1])

    # stop logic
    stop = float(trig["Stop"]) if np.isfinite(trig["Stop"]) else float(close.rolling(20).min().iloc[-1])

    # targets (ATR-based light guidance)
    atr = float((d["High"] - d["Low"]).rolling(14).mean().iloc[-1]) if len(d) >= 20 else float("nan")
    t1, t2 = targets_from_entry(entry, "LONG" if direction == "LONG" else "SHORT", atr)

    return {
        "Ticker": ticker.upper(),
        "Price": float(close.iloc[-1]),
        "Strength": strength,
        "Meter": meter,
        "Trend": tr,
        "RSI": rsi,
        "RS_short": rs_s,
        "RS_long": rs_l,
        "Rotation": rot,
        "Setup": setup,
        "Direction": direction,
        "TriggerStatus": status,
        "Entry": entry,
        "Stop": stop,
        "T1": t1,
        "T2": t2,
        "Note": trig.get("Note", ""),
        "TF": "D",
    }


def short_writeup(info: Dict) -> str:
    """
    One-liner for dashboard “write-up” lists.
    """
    s = f"{info['Ticker']} — {info['Meter']} {info['Strength']}/100 | {info['Trend']} | {info['Setup']} {info['TriggerStatus']}"
    if info.get("Direction") in ("LONG", "SHORT"):
        s += f" | {info['Direction']} idea"
    return s


def writeup_block(info: Dict, pb_low: float, pb_high: float):
    """
    Streamlit renderer used by multiple pages (keeps UI pages thin).
    """
    import streamlit as st

    st.markdown(f"### {info['Ticker']} — {info['Meter']} ({info['Strength']}/100)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trend", info["Trend"])
    c2.metric("RSI", f"{info['RSI']:.1f}")
    c3.metric("RS short", f"{info['RS_short']:.2%}")
    c4.metric("Rotation", f"{info['Rotation']:.2%}")

    st.write(f"**STRAT:** {info['Setup']} | **Status:** {info['TriggerStatus']} | **Direction:** {info['Direction']}")
    if info.get("Note"):
        st.caption(info["Note"])

    st.write(f"**Entry (guide):** {info['Entry']:.2f}")
    st.write(f"**Stop (guide):** {info['Stop']:.2f}")

    if np.isfinite(info.get("T1", np.nan)) and np.isfinite(info.get("T2", np.nan)):
        st.write(f"Targets (ATR guide): **T1 {info['T1']:.2f}**, **T2 {info['T2']:.2f}**")

    if info["Trend"] == "UP" and (pb_low <= info["RSI"] <= pb_high):
        st.success(f"Pullback zone OK (RSI {pb_low}–{pb_high})")
    else:
        st.info("Pullback zone not confirmed (or trend not UP).")
