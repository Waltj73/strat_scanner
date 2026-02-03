# indicators.py — Technical indicators & strength calculations
# Used by dashboard, scanner, and analyzer

import numpy as np
import pandas as pd


# =========================
# BASIC INDICATORS
# =========================
def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=length, adjust=False).mean()


def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    """Wilder RSI implementation."""
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


# =========================
# RETURN & RELATIVE STRENGTH
# =========================
def total_return(series: pd.Series, lookback: int) -> pd.Series:
    """Percent return over lookback."""
    return series / series.shift(lookback) - 1


def rs_vs_spy(series: pd.Series, spy: pd.Series, lookback: int) -> pd.Series:
    """Relative strength vs SPY."""
    return total_return(series, lookback) - total_return(spy, lookback)


# =========================
# CAPS TO AVOID EXTREME DISTORTION
# =========================
RS_CAP = 0.10   # ±10%
ROT_CAP = 0.08  # ±8%


def clamp(value, lo, hi):
    try:
        return max(lo, min(hi, float(value)))
    except Exception:
        return 0.0


def clamp_rs(x, lo=-RS_CAP, hi=RS_CAP):
    return clamp(x, lo, hi)


# =========================
# TREND DETECTION
# =========================
def trend_label(series: pd.Series, ema_length: int) -> str:
    """
    Trend is UP when:
    - price above EMA
    - EMA rising
    """
    e = ema(series, ema_length)

    if len(e) < 2:
        return "DOWN/CHOP"

    up = bool(
        series.iloc[-1] > e.iloc[-1]
        and e.iloc[-1] > e.iloc[-2]
    )

    return "UP" if up else "DOWN/CHOP"


# =========================
# STRENGTH METER (0–100)
# =========================
def strength_meter(rs_short, rotation, trend: str) -> int:
    """
    Combines:
    - RS short term
    - Rotation
    - Trend bonus
    """

    rs_short = clamp_rs(rs_short)
    rotation = clamp(rotation, -ROT_CAP, ROT_CAP)

    rs_score = np.clip(50 + rs_short * 600, 0, 100)
    rot_score = np.clip(50 + rotation * 800, 0, 100)

    trend_bonus = 10 if trend == "UP" else -10

    score = (
        0.50 * rs_score +
        0.35 * rot_score +
        0.15 * 50 +
        trend_bonus
    )

    return int(np.clip(score, 0, 100))


def strength_label(score: int) -> str:
    if score >= 70:
        return "STRONG"
    if score >= 45:
        return "NEUTRAL"
    return "WEAK"


# =========================
# RSI PULLBACK FILTER
# =========================
def pullback_zone_ok(trend: str, rsi_val: float, low: float, high: float) -> bool:
    """
    Pullback allowed only in UP trends.
    """
    if trend != "UP":
        return False
    return low <= rsi_val <= high


# =========================
# ATR (volatility)
# =========================
def atr14(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 20:
        return float("nan")

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return float(tr.rolling(14).mean().iloc[-1])

