# pages/analyzer.py
import math
from typing import Dict, Optional, Tuple, List

import numpy as np
import streamlit as st
import pandas as pd

from data import get_hist
from indicators import rsi_wilder, trend_label, rs_vs_spy
from strat import tf_frames, compute_flags, best_trigger, atr14
from engine import (
    RS_CAP, ROT_CAP,
    clamp_rs,
    strength_meter,
    strength_label,
)

# -------------------------
# Helpers (Analyzer-specific)
# -------------------------
def pullback_zone_ok(trend: str, rsi_val: float, pb_low: float, pb_high: float) -> bool:
    return (trend == "UP") and (pb_low <= rsi_val <= pb_high)

def atrp_bucket(atrp: Optional[float]) -> str:
    if atrp is None:
        return "unknown"
    if atrp < 1.2:
        return "small"
    if atrp < 2.5:
        return "normal"
    return "aggressive"

def targets_from_range(d: pd.DataFrame, direction: str = "LONG") -> Tuple[Optional[float], Optional[float]]:
    if d is None or d.empty or len(d) < 70:
        return None, None
    hi20 = float(d["High"].rolling(20).max().iloc[-1])
    lo20 = float(d["Low"].rolling(20).min().iloc[-1])
    hi63 = float(d["High"].rolling(63).max().iloc[-1])
    lo63 = float(d["Low"].rolling(63).min().iloc[-1])
    return (hi20, hi63) if direction == "LONG" else (lo20, lo63)

def pick_grade(
    strength: int,
    trend: str,
    rotation: float,
    has_trigger: bool,
    rsi_val: float,
    pb_low: float,
    pb_high: float
) -> str:
    rot_c = clamp_rs(rotation, -ROT_CAP, ROT_CAP)
    pb_ok = pullback_zone_ok(trend, rsi_val, pb_low, pb_high)

    score = 0
    score += 2 if strength >= 75 else 1 if strength >= 65 else 0
    score += 1 if trend == "UP" else 0
    score += 1 if rot_c > 0 else 0
    score += 1 if has_trigger else 0
    score += 1 if pb_ok else 0

    if score >= 5:
        return "A"
    if score >= 3:
        return "B"
    return "C"

def trade_plan_notes(
    trend: str,
    rsi_val: float,
    strength: int,
    rotation: float,
    trigger_status: str,
    entry: Optional[float],
    stop: Optional[float],
    d_df: pd.DataFrame,
    pb_low: float,
    pb_high: float,
    direction: str = "LONG",
) -> Dict[str, str]:
    has_trigger = (entry is not None and stop is not None) and ("READY" in trigger_status)
    grade = pick_grade(strength, trend, rotation, has_trigger, rsi_val, pb_low, pb_high)

    if trend != "UP" and direction == "LONG":
        play = "AVOID (trend not UP). Focus names above the trend EMA with positive RS + rotation."
    else:
        if has_trigger:
            play = "BREAKOUT PLAY: place a stop order at Entry. Stop goes at Stop."
        else:
            pb_ok = pullback_zone_ok(trend, rsi_val, pb_low, pb_high)
            play = "WAIT: no Inside Bar trigger yet." if not pb_ok else "PULLBACK PLAY: strong name in pullback zone—wait for an Inside Bar."

    risk_unit = None
    if entry is not None and stop is not None:
        risk_unit = abs(entry - stop)

    atr = atr14(d_df)
    atrp = None
    if math.isfinite(atr) and atr > 0:
        close = float(d_df["Close"].iloc[-1])
        if close > 0:
            atrp = (atr / close) * 100.0

    sizing_hint = atrp_bucket(atrp)
    if sizing_hint == "small":
        sizing_note = "ATR% is small → can size a bit larger, expect slower movement."
    elif sizing_hint == "normal":
        sizing_note = "ATR% is normal → standard sizing."
    elif sizing_hint == "aggressive":
        sizing_note = "ATR% is high → size down, wider swings."
    else:
        sizing_note = "ATR% unavailable."

    t1, t2 = targets_from_range(d_df, direction)
    invalidation = "Invalidation = break below STOP (or close below if you trade close-based)."

    improve = []
    if trend != "UP":
        improve.append("Improve: reclaim and hold above the trend EMA + RS turns positive.")
    if clamp_rs(rotation, -ROT_CAP, ROT_CAP) <= 0:
        improve.append("Improve: rotation flips positive.")
    if not pullback_zone_ok(trend, rsi_val, pb_low, pb_high):
        improve.append(f"Improve: RSI pulls into {pb_low}-{pb_high} zone without breaking trend.")
    if not has_trigger:
        improve.append("Improve: print a Daily or Weekly Inside Bar for clean entry/stop.")

    rr_hint = "RR: n/a"
    if risk_unit is not None and t2 is not None and entry is not None:
        reward = max(0.0, t2 - entry) if direction == "LONG" else max(0.0, entry - t2)
        rr = reward / risk_unit if risk_unit > 0 else None
        if rr is not None and math.isfinite(rr):
            rr_hint = f"Approx RR to T2: ~{rr:.2f} (uses 63d extreme as T2)."

    return {
        "Grade": grade,
        "Play": play,
        "RiskUnit": f"{risk_unit:.2f}" if risk_unit is not None else "n/a",
        "Targets": f"T1: {t1:.2f} | T2: {t2:.2f}" if (t1 is not None and t2 is not None) else "Targets: n/a",
        "Invalidation": invalidation,
        "Improve": " | ".join(improve) if improve else "Improve: wait for trigger.",
        "Sizing": sizing_note,
        "RRHint": rr_hint,
    }

def analyze_ticker(
    ticker: str,
    spy_close: pd.Series,
    rs_short: int,
    rs_long: int,
    ema_trend_len: int,
    rsi_len: int,
) -> Optional[Dict]:
    d = get_hist(ticker)
    if d is None or d.empty:
        return None

    close = d["Close"].dropna()
    if close.empty or len(close) < max(rs_long, 80) + 5:
        return None

    tr = trend_label(close, ema_trend_len)
    rsi_v = float(rsi_wilder(close, rsi_len).iloc[-1])

    rs_s = float(rs_vs_spy(close, spy_close, rs_short).iloc[-1])
    rs_l = float(rs_vs_spy(close, spy_close, rs_long).iloc[-1])

    rs_s_c = clamp_rs(rs_s, -RS_CAP, RS_CAP)
    rs_l_c = clamp_rs(rs_l, -RS_CAP, RS_CAP)

    rot = rs_s - rs_l
    rot_c = clamp_rs(rot, -ROT_CAP, ROT_CAP)

    strength = strength_meter(rs_s_c, rot_c, tr)
    meter = strength_label(strength)

    d_tf, w_tf, m_tf = tf_frames(d)
    flags = compute_flags(d_tf, w_tf, m_tf)

    tf, entry, stop = best_trigger("LONG", d_tf, w_tf)
    trigger_status = "READY" if (flags.get("W_Inside") or flags.get("D_Inside")) else "WAIT (No Inside Bar)"

    entry_r = None if entry is None else round(float(entry), 2)
    stop_r  = None if stop  is None else round(float(stop), 2)

    explain = [
        f"Trend = {tr} (price vs {ema_trend_len} EMA + EMA slope)",
        f"RSI({rsi_len}) = {rsi_v:.1f}",
        f"RS vs SPY short ({rs_short}) = {rs_s*100:.1f}% (capped {rs_s_c*100:.1f}%)",
        f"RS vs SPY long  ({rs_long})  = {rs_l*100:.1f}% (capped {rs_l_c*100:.1f}%)",
        f"Rotation = RS(short) - RS(long) = {rot*100:.1f}% (capped {rot_c*100:.1f}%)",
        f"Strength Score = {strength}/100 ({meter})",
    ]

    strat_notes: List[str] = []
    if flags.get("M_Bull"): strat_notes.append("Monthly: Bull")
    if flags.get("W_Bull"): strat_notes.append("Weekly: Bull")
    if flags.get("D_Bull"): strat_notes.append("Daily: Bull")
    if flags.get("W_Inside"): strat_notes.append("Weekly: Inside Bar")
    if flags.get("D_Inside"): strat_notes.append("Daily: Inside Bar")
    if flags.get("W_212Up"): strat_notes.append("Weekly: 2-1-2 Up")
    if flags.get("D_212Up"): strat_notes.append("Daily: 2-1-2 Up")
    if not strat_notes:
        strat_notes = ["No STRAT alignment flags currently"]

    return {
        "Ticker": ticker.upper(),
        "Trend": tr,
        "RSI": rsi_v,
        "RS_short": rs_s,
        "RS_long": rs_l,
        "Rotation": rot,
        "Strength": int(strength),
        "Meter": meter,
        "TriggerStatus": trigger_status,
        "TF": tf,
        "Entry": entry_r,
        "Stop": stop_r,
        "Flags": flags,
        "Explain": explain,
        "STRAT_Notes": strat_notes,
        "DailyDF": d,
    }

def render_writeup(info: Dict, pb_low: float, pb_high: float) -> None:
    plan = trade_plan_notes(
        trend=info["Trend"],
        rsi_val=float(info["RSI"]),
        strength=int(info["Strength"]),
        rotation=float(info["Rotation"]),
        trigger_status=info["TriggerStatus"],
        entry=info["Entry"],
        stop=info["Stop"],
        d_df=info["DailyDF"],
        pb_low=pb_low,
        pb_high=pb_high,
        direction="LONG",
    )

    st.markdown(f"### {info['Ticker']} — {info['Meter']} ({info['Strength']}/100) | **Grade: {plan['Grade']}**")

    a, b, c, d = st.columns(4)
    with a: st.metric("Trend", info["Trend"])
    with b: st.metric("RSI", f"{info['RSI']:.1f}")
    with c: st.metric("RS short vs SPY", f"{info['RS_short']*100:.1f}%")
    with d: st.metric("Rotation", f"{info['Rotation']*100:.1f}%")

    pb_ok = pullback_zone_ok(info["Trend"], info["RSI"], pb_low, pb_high)
    st.write(f"**Pullback Zone ({pb_low}-{pb_high}) OK?** {'✅ YES' if pb_ok else '❌ NO'}")

    st.write(
        f"**Trigger:** {info['TriggerStatus']}"
        + (f" | TF: **{info['TF']}** | Entry: **{info['Entry']}** | Stop: **{info['Stop']}**" if info["Entry"] else "")
    )

    st.markdown("#### 🧠 Trade Plan Notes")
    x1, x2, x3 = st.columns([1, 2, 2])
    with x1:
        st.metric("Grade", plan["Grade"])
    with x2:
        st.write(f"**Play:** {plan['Play']}")
    with x3:
        st.write(f"**Risk Unit:** {plan['RiskUnit']} | **{plan['RRHint']}**")

    st.write(f"**Targets:** {plan['Targets']}")
    st.write(f"**Invalidation:** {plan['Invalidation']}")
    st.write(f"**Sizing Hint:** {plan['Sizing']}")
    st.write(f"**What makes it better:** {plan['Improve']}")

    with st.expander("Why this scores the way it does"):
        for line in info["Explain"]:
            st.write(f"- {line}")
        st.write("**STRAT context:**")
        for s in info["STRAT_Notes"]:
            st.write(f"- {s}")

# -------------------------
# Page entrypoint
# -------------------------
def render():
    st.title("🔎 Ticker Analyzer — Score + Grade + STRAT Context + Trade Plan")
    st.caption("Type any ticker and get a complete swing-trader write-up (Strength, Grade, triggers, and plan).")

    with st.expander("Analyzer Settings", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([1.2, 1.1, 1.1, 1.1, 1.1])
        with c1:
            ticker = st.text_input("Ticker", value="AAPL")
        with c2:
            rs_short = st.selectbox("RS Lookback (short)", [21, 30, 42], index=0)
        with c3:
            rs_long = st.selectbox("RS Lookback (long)", [63, 90, 126], index=0)
        with c4:
            ema_trend_len = st.selectbox("Trend EMA", [50, 100, 200], index=0)
        with c5:
            rsi_len = st.selectbox("RSI Length", [7, 14, 21], index=1)

    pb1, pb2 = st.columns(2)
    with pb1:
        pb_low = st.slider("RSI Pullback Low (UP trend)", 25, 60, 40)
    with pb2:
        pb_high = st.slider("RSI Pullback High (UP trend)", 35, 75, 55)

    spy_df = get_hist("SPY")
    if spy_df is None or spy_df.empty:
        st.error("SPY data unavailable; cannot compute RS vs SPY. Hit Refresh or try later.")
        return

    spy_close = spy_df["Close"].dropna()
    if spy_close.empty or len(spy_close) < (rs_long + 10):
        st.error("Not enough SPY history for these lookbacks. Lower RS long lookback or refresh.")
        return

    if not ticker:
        st.info("Type a ticker to analyze.")
        return

    info = analyze_ticker(ticker.strip().upper(), spy_close, int(rs_short), int(rs_long), int(ema_trend_len), int(rsi_len))
    if info is None:
        st.warning("No data returned (bad ticker, not enough history, or yfinance empty). Try another symbol.")
        return

    render_writeup(info, pb_low, pb_high)
