from __future__ import annotations
import numpy as np
import pandas as pd

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def total_return(close: pd.Series, lookback: int) -> float:
    if close is None or close.empty or len(close) <= lookback:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-1 - lookback] - 1)

def rs_vs_spy(close: pd.Series, spy_close: pd.Series, lookback: int) -> float:
    return total_return(close, lookback) - total_return(spy_close, lookback)

def trend_label(close: pd.Series, ema_len: int = 50) -> str:
    if close is None or close.empty or len(close) < ema_len + 3:
        return "n/a"
    e = ema(close, ema_len)
    up = bool(close.iloc[-1] > e.iloc[-1] and e.iloc[-1] > e.iloc[-2])
    return "UP" if up else "DOWN/CHOP"

def clamp(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(lo, min(hi, x))

def strength_score(rs_short: float, rs_long: float, rsi: float, trend: str) -> int:
    """
    0–100 score.
    - RS short & long are returns relative to SPY.
    - Rotation is (RS short - RS long).
    """
    RS_CAP = 0.10   # +/-10%
    ROT_CAP = 0.08  # +/-8%

    rs_s = clamp(rs_short, -RS_CAP, RS_CAP)
    rs_l = clamp(rs_long, -RS_CAP, RS_CAP)
    rot  = clamp(rs_short - rs_long, -ROT_CAP, ROT_CAP)

    rs_part  = np.clip(50 + rs_s*100*6.0, 0, 100)
    rot_part = np.clip(50 + rot*100*8.0, 0, 100)
    rsi_part = np.clip(float(rsi), 0, 100)

    trend_bonus = 8 if trend == "UP" else -8

    score = 0.45*rs_part + 0.35*rot_part + 0.20*rsi_part + trend_bonus
    return int(np.clip(score, 0, 100))

def strength_trend(
    close: pd.Series,
    spy_close: pd.Series,
    rs_short: int,
    rs_long: int,
    ema_len: int,
    rsi_len: int,
    lookback_bars: int = 5,   # ~1 trading week
) -> float:
    """
    Strength trend = strength(today) - strength(N bars ago).
    Uses same Strength formula, computed at two different timestamps.
    """
    if close is None or close.empty or spy_close is None or spy_close.empty:
        return float("nan")
    if len(close) < (max(rs_long, ema_len, rsi_len) + lookback_bars + 10):
        return float("nan")
    if len(spy_close) < (max(rs_long, ema_len, rsi_len) + lookback_bars + 10):
        return float("nan")

    # today
    rsi_today = float(rsi_wilder(close, rsi_len).iloc[-1])
    tr_today = trend_label(close, ema_len)
    rs_s_today = rs_vs_spy(close, spy_close, rs_short)
    rs_l_today = rs_vs_spy(close, spy_close, rs_long)
    s_today = strength_score(rs_s_today, rs_l_today, rsi_today, tr_today)

    # N bars ago
    close_prev = close.iloc[:-lookback_bars]
    spy_prev = spy_close.iloc[:-lookback_bars]
    if close_prev.empty or spy_prev.empty:
        return float("nan")

    rsi_prev = float(rsi_wilder(close_prev, rsi_len).iloc[-1])
    tr_prev = trend_label(close_prev, ema_len)
    rs_s_prev = rs_vs_spy(close_prev, spy_prev, rs_short)
    rs_l_prev = rs_vs_spy(close_prev, spy_prev, rs_long)
    s_prev = strength_score(rs_s_prev, rs_l_prev, rsi_prev, tr_prev)

    return float(s_today - s_prev)
