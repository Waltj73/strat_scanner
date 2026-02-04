# pages/analyzer.py — Ticker Analyzer
# Deep dive: Trend + RSI + RS vs SPY + Rotation + STRAT trigger + trade plan notes

from typing import Dict

import streamlit as st
import pandas as pd

from data import get_hist
from indicators import (
    rsi_wilder,
    rs_vs_spy,
    trend_label,
    clamp_rs,
    strength_meter,
    strength_label,
    pullback_zone_ok,
    RS_CAP,
    ROT_CAP,
)
from strat import tf_frames, compute_flags, best_trigger


# -------------------------
# Analyzer core
# -------------------------
def analyze_ticker(
    ticker: str,
    spy_close: pd.Series,
    rs_short: int,
    rs_long: int,
    ema_trend_len: int,
    rsi_len: int,
) -> Dict | None:

    d = get_hist(ticker)
    if d.empty:
        return None

    close = d["Close"].dropna()
    if close.empty or len(close) < max(rs_long, 80) + 10:
        return None

    trend = trend_label(close, ema_trend_len)
    rsi_v = float(rsi_wilder(close, rsi_len).iloc[-1])

    rs_s = float(rs_vs_spy(close, spy_close, rs_short).iloc[-1])
    rs_l = float(rs_vs_spy(close, spy_close, rs_long).iloc[-1])

    rs_s_c = clamp_rs(rs_s, -RS_CAP, RS_CAP)
    rs_l_c = clamp_rs(rs_l, -RS_CAP, RS_CAP)

    rotation = rs_s - rs_l
    rot_c = clamp_rs(rotation, -ROT_CAP, ROT_CAP)

    strength = strength_meter(rs_s_c, rot_c, trend)
    meter = strength_label(strength)

    # STRAT context
    d_tf, w_tf, m_tf = tf_frames(d)
    flags = compute_flags(d_tf, w_tf, m_tf)

    tf, entry, stop = best_trigger("LONG", d_tf, w_tf)

    trigger_status = (
        "READY"
        if (flags.get("W_Inside") or flags.get("D_Inside"))
        else "WAIT (No Inside Bar)"
    )

    strat_notes = []
    if flags.get("M_Bull"):
        strat_notes.append("Monthly Bull")
    if flags.get("W_Bull"):
        strat_notes.append("Weekly Bull")
    if flags.get("D_Bull"):
        strat_notes.append("Daily Bull")

    if flags.get("W_Inside"):
        strat_notes.append("Weekly Inside Bar")
    if flags.get("D_Inside"):
        strat_notes.append("Daily Inside Bar")

    if flags.get("W_212Up"):
        strat_notes.append("Weekly 2-1-2 Up")
    if flags.get("D_212Up"):
        strat_notes.append("Daily 2-1-2 Up")

    if not strat_notes:
        strat_notes = ["No strong STRAT alignment"]

    explain = [
        f"Trend = {trend} (price vs EMA({ema_trend_len}) + EMA slope)",
        f"RSI({rsi_len}) = {rsi_v:.1f}",
        f"RS vs SPY short ({rs_short}) = {rs_s*100:.1f}% (capped {rs_s_c*100:.1f}%)",
        f"RS vs SPY long ({rs_long}) = {rs_l*100:.1f}% (capped {rs_l_c*100:.1f}%)",
        f"Rotation = short − long RS = {rotation*100:.1f}% (capped {rot_c*100:.1f}%)",
        f"Strength score = {strength}/100 ({meter})",
    ]

    return {
        "Ticker": ticker.upper(),
        "Trend": trend,
        "RSI": rsi_v,
        "RS_short": rs_s,
        "RS_long": rs_l,
        "Rotation": rotation,
        "Strength": strength,
        "Meter": meter,
        "TriggerStatus": trigger_status,
        "TF": tf,
        "Entry": None if entry is None else round(float(entry), 2),
        "Stop": None if stop is None else round(float(stop), 2),
        "Explain": explain,
        "STRAT_Notes": strat_notes,
        "DailyDF": d,
    }


# -------------------------
# UI Page
# -------------------------
def show_analyzer():
    st.title("🔎 Ticker Analyzer — Explain the Score + STRAT Context")
    st.caption("Type any ticker and get a swing-trader style breakdown.")

    # ---------- Settings ----------
    with st.expander("Analyzer Settings", expanded=True):
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 1.1])

        with c1:
            ticker = st.text_input("Ticker", value="AAPL")

        with c2:
            rs_short = st.selectbox("RS Lookback (short)", [21, 30, 42], index=0)

        with c3:
            rs_long = st.selectbox("RS Lookback (long)", [63, 90, 126], index=0)

        with c4:
            ema_trend_len = st.selectbox("Trend EMA", [50, 100, 200], index=0)

        rsi_len = st.selectbox("RSI Length", [7, 14, 21], index=1)

    # ---------- Pullback zone ----------
    pb1, pb2 = st.columns(2)
    with pb1:
        pb_low = st.slider("RSI Pullback Low (UP trend)", 25, 60, 40)
    with pb2:
        pb_high = st.slider("RSI Pullback High (UP trend)", 35, 75, 55)

    # ---------- SPY baseline ----------
    spy_df = get_hist("SPY")
    if spy_df.empty:
        st.warning("SPY data unavailable.")
        return

    spy = spy_df["Close"].dropna()
    if len(spy) < (rs_long + 10):
        st.warning("Not enough SPY history.")
        return

    # ---------- Analyzer output ----------
    if ticker:
        info = analyze_ticker(
            ticker.strip().upper(),
            spy,
            int(rs_short),
            int(rs_long),
            int(ema_trend_len),
            int(rsi_len),
        )

        if info is None:
            st.warning("No data returned.")
            return

        st.markdown(f"## {info['Ticker']} — {info['Meter']} ({info['Strength']}/100)")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.write(f"**Trend:** {info['Trend']}")
        with c2:
            st.write(f"**RSI:** {info['RSI']:.1f}")
        with c3:
            st.write(f"**RS short:** {info['RS_short']*100:.1f}%")
        with c4:
            st.write(f"**Rotation:** {info['Rotation']*100:.1f}%")

        pb_ok = pullback_zone_ok(info["Trend"], info["RSI"], pb_low, pb_high)
        st.write(f"Pullback zone valid? {'✅ YES' if pb_ok else '❌ NO'}")

        if info["Entry"]:
            st.write(
                f"Trigger: **{info['TriggerStatus']}** | "
                f"TF: **{info['TF']}** | "
                f"Entry: **{info['Entry']}** | "
                f"Stop: **{info['Stop']}**"
            )
        else:
            st.write(f"Trigger: **{info['TriggerStatus']}**")

        st.markdown("### Why this scores the way it does")
        for line in info["Explain"]:
            st.write(f"- {line}")

        st.markdown("### STRAT Context")
        for s in info["STRAT_Notes"]:
            st.write(f"- {s}")
