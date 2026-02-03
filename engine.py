# engine.py — Scanner & analysis engine
# Purpose:
# - ticker analysis
# - RR / magnitude calculations
# - scoring logic
# - writeup generation

import math
import pandas as pd

from data import get_hist
from indicators import (
    rsi_wilder,
    atr14,
    trend_label,
    strength_meter,
)
from strat import (
    tf_frames,
    compute_flags,
    best_trigger,
)


# =========================
# MAGNITUDE / RR METRICS
# =========================
def magnitude_metrics(bias, d, entry, stop):
    if d is None or d.empty or len(d) < 80:
        return None, None, None, None

    close = float(d["Close"].iloc[-1])
    atr = atr14(d)

    if not math.isfinite(atr) or atr <= 0:
        return None, None, None, None

    atrp = (atr / close) * 100

    if entry is None or stop is None:
        return None, atrp, None, None

    hi63 = float(d["High"].rolling(63).max().iloc[-1])
    lo63 = float(d["Low"].rolling(63).min().iloc[-1])

    if bias == "LONG":
        target = hi63
        room = max(0.0, target - entry)
        risk = entry - stop
        reward = target - entry
    else:
        target = lo63
        room = max(0.0, entry - target)
        risk = stop - entry
        reward = entry - target

    rr = None if risk <= 0 else (reward / risk if reward > 0 else 0.0)

    compression = None
    if len(d) >= 2:
        cur = d.iloc[-1]
        rng = float(cur["High"] - cur["Low"])
        compression = rng / atr if atr > 0 else None

    return rr, atrp, room, compression


# =========================
# SETUP + MAG SCORING
# =========================
def calc_scores(bias, flags, rr, atrp, compression, entry, stop):
    setup = 0
    mag = 0

    if bias == "LONG":
        setup += 30 if flags["M_Bull"] else 0
        setup += 20 if flags["W_Bull"] else 0
        setup += 10 if flags["D_Bull"] else 0
        setup += 20 if flags["W_212Up"] else 0
        setup += 10 if flags["D_212Up"] else 0
    else:
        setup += 30 if flags["M_Bear"] else 0
        setup += 20 if flags["W_Bear"] else 0
        setup += 10 if flags["D_Bear"] else 0
        setup += 20 if flags["W_212Dn"] else 0
        setup += 10 if flags["D_212Dn"] else 0

    setup += 10 if flags["W_Inside"] else 0
    setup += 5 if flags["D_Inside"] else 0

    # Magnitude scoring
    if rr is not None:
        if rr >= 3:
            mag += 35
        elif rr >= 2:
            mag += 25
        elif rr >= 1.5:
            mag += 10

    if atrp is not None:
        if atrp >= 3:
            mag += 20
        elif atrp >= 2:
            mag += 10
        elif atrp >= 1:
            mag += 5

    if compression is not None:
        if compression <= 0.6:
            mag += 15
        elif compression <= 0.9:
            mag += 8
        elif compression <= 1.2:
            mag += 3

    if entry is not None and stop is not None:
        mag += 5

    total = setup + mag
    return setup, mag, total


# =========================
# ANALYZE TICKER
# =========================
def analyze_ticker(ticker, bias="LONG"):
    d = get_hist(ticker)
    if d.empty:
        return None

    d_tf, w_tf, m_tf = tf_frames(d)
    flags = compute_flags(d_tf, w_tf, m_tf)

    tf, entry, stop = best_trigger(bias, d_tf, w_tf)

    rr, atrp, room, compression = magnitude_metrics(
        bias, d_tf, entry, stop
    )

    setup_score, mag_score, total_score = calc_scores(
        bias,
        flags,
        rr,
        atrp,
        compression,
        entry,
        stop,
    )

    close = float(d_tf["Close"].iloc[-1])
    rsi = float(rsi_wilder(d_tf["Close"]).iloc[-1])

    trend = trend_label(d_tf["Close"], 50)

    strength = strength_meter(0, 0, trend)

    return {
        "Ticker": ticker,
        "Close": round(close, 2),
        "RSI": round(rsi, 1),
        "Trend": trend,
        "Strength": strength,
        "SetupScore": setup_score,
        "MagScore": mag_score,
        "TotalScore": total_score,
        "TF": tf,
        "Entry": None if entry is None else round(entry, 2),
        "Stop": None if stop is None else round(stop, 2),
        "RR": None if rr is None else round(rr, 2),
        "ATR%": None if atrp is None else round(atrp, 2),
        "Room": None if room is None else round(room, 2),
        "Flags": flags,
    }


# =========================
# TRADE NOTES GENERATOR
# =========================
def trade_plan_notes(data):
    if data is None:
        return "No data available."

    notes = []

    notes.append(f"Trend: {data['Trend']}")
    notes.append(f"RSI: {data['RSI']}")

    if data["Entry"]:
        notes.append(
            f"Trigger above {data['Entry']} with stop {data['Stop']}."
        )
    else:
        notes.append("No Inside Bar trigger yet.")

    if data["RR"]:
        notes.append(f"Risk/Reward ≈ {data['RR']}")

    notes.append(
        f"Setup score {data['SetupScore']} + magnitude {data['MagScore']}."
    )

    return "\n".join(notes)


# =========================
# WRITEUP BLOCK
# =========================
def writeup_block(data):
    if data is None:
        return "Ticker not found."

    lines = [
        f"### {data['Ticker']} Trade Summary",
        f"- Trend: **{data['Trend']}**",
        f"- Strength Score: **{data['Strength']}**",
        f"- RSI: **{data['RSI']}**",
        f"- Setup Score: **{data['SetupScore']}**",
        f"- Magnitude Score: **{data['MagScore']}**",
        f"- Total Score: **{data['TotalScore']}**",
    ]

    if data["Entry"]:
        lines.append(
            f"- Trigger: Break **{data['Entry']}**, stop **{data['Stop']}**"
        )
        lines.append(f"- RR: **{data['RR']}**")

    return "\n".join(lines)

