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

    df = get_hist(ticker)
    if df is None or df.empty:
        return None

    if "Close" not in df.columns:
        return None

    close = df["Close"].dropna()
    if len(close) < (rs_long + 10) or len(spy_close) < (rs_long + 10):
        return None

    # --- Rotation / strength ---
    rs_s = float(rs_vs_spy(close, spy_close, int(rs_short)).iloc[-1])
    rs_l = float(rs_vs_spy(close, spy_close, int(rs_long)).iloc[-1])
    rot = rs_s - rs_l

    trend = trend_label(close, int(ema_trend_len))
    rsi = float(rsi_wilder(close, int(rsi_len)).iloc[-1])

    strength = int(strength_meter(rs_s, rot, trend))
    meter = strength_label(strength)

    # --- REAL Strat trigger ---
    direction = "LONG" if trend == "UP" else "SHORT"
    trigger = best_trigger(df, direction=direction)

    return {
        "Ticker": ticker.upper(),
        "Strength": strength,
        "Meter": meter,
        "Trend": trend,
        "RSI": rsi,
        "RS_short": rs_s,
        "RS_long": rs_l,
        "Rotation": rot,
        "TriggerStatus": trigger.status,
        "TF": trigger.tf,
        "Entry": trigger.entry,
        "Stop": trigger.stop,
    }
