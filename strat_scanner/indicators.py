from __future__ import annotations
import numpy as np
import pandas as pd

RS_CAP = 0.10
ROT_CAP = 0.08

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def total_return(series: pd.Series, lookback: int) -> pd.Series:
    return series / series.shift(lookback) - 1

def rs_vs_spy(series: pd.Series, spy_series: pd.Series, lookback: int) -> pd.Series:
    return total_return(series, lookback) - total_return(spy_series, lookback)

def clamp_rs(x, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0

def trend_label(series: pd.Series, ema_len: int) -> str:
    if series is None or len(series) < ema_len + 3:
        return "DOWN/CHOP"
    e = ema(series, ema_len)
    up = bool(series.iloc[-1] > e.iloc[-1] and e.iloc[-1] >= e.iloc[-2])
    return "UP" if up else "DOWN/CHOP"

def pullback_zone_ok(trend: str, rsi_val: float, pb_low: float, pb_high: float) -> bool:
    if trend != "UP":
        return False
    try:
        r = float(rsi_val)
    except Exception:
        return False
    return pb_low <= r <= pb_high

def strength_meter(rs_short_v: float, rotation_v: float, trend: str) -> int:
    rs_short_v = clamp_rs(rs_short_v, -RS_CAP, RS_CAP)
    rotation_v = clamp_rs(rotation_v, -ROT_CAP, ROT_CAP)

    rs_score = np.clip(50 + (rs_short_v * 100.0) * 6.0, 0, 100)
    rot_score = np.clip(50 + (rotation_v * 100.0) * 8.0, 0, 100)
    trend_bonus = 10 if trend == "UP" else -10

    score = 0.50 * rs_score + 0.35 * rot_score + 0.15 * 50 + trend_bonus
    return int(np.clip(score, 0, 100))

def strength_label(score: int) -> str:
    if score >= 70:
        return "STRONG"
    if score >= 45:
        return "NEUTRAL"
    return "WEAK"
