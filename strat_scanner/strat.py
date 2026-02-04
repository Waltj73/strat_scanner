# strat.py — STRAT pattern logic & regime scoring
# Handles:
# - Inside bars
# - 2u / 2d
# - 2-1-2 patterns
# - timeframe helpers
# - regime scoring
# - trigger detection

import pandas as pd
from typing import Dict, Optional, Tuple

from data import resample_ohlc


# =========================
# BASIC BAR TYPES
# =========================
def is_inside_bar(cur: pd.Series, prev: pd.Series) -> bool:
    return (cur["High"] <= prev["High"]) and (cur["Low"] >= prev["Low"])


def is_2up(cur: pd.Series, prev: pd.Series) -> bool:
    return (cur["High"] > prev["High"]) and (cur["Low"] >= prev["Low"])


def is_2dn(cur: pd.Series, prev: pd.Series) -> bool:
    return (cur["Low"] < prev["Low"]) and (cur["High"] <= prev["High"])


def is_green(cur: pd.Series) -> bool:
    return cur["Close"] > cur["Open"]


def is_red(cur: pd.Series) -> bool:
    return cur["Close"] < cur["Open"]


# =========================
# HELPER: last two bars
# =========================
def last_two(df: pd.DataFrame):
    if df is None or df.empty or len(df) < 2:
        return None
    return df.iloc[-1], df.iloc[-2]


# =========================
# STRAT CONDITIONS
# =========================
def strat_bull(df: pd.DataFrame) -> bool:
    pair = last_two(df)
    if not pair:
        return False

    cur, prev = pair
    return is_2up(cur, prev) and is_green(cur)


def strat_bear(df: pd.DataFrame) -> bool:
    pair = last_two(df)
    if not pair:
        return False

    cur, prev = pair
    return is_2dn(cur, prev) and is_red(cur)


def strat_inside(df: pd.DataFrame) -> bool:
    pair = last_two(df)
    if not pair:
        return False

    cur, prev = pair
    return is_inside_bar(cur, prev)


# =========================
# 2-1-2 PATTERNS
# =========================
def strat_212_up(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 4:
        return False

    a = df.iloc[-3]
    b = df.iloc[-2]
    c = df.iloc[-1]
    prev_a = df.iloc[-4]

    return (
        is_2up(a, prev_a)
        and is_inside_bar(b, a)
        and is_2up(c, b)
    )


def strat_212_dn(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 4:
        return False

    a = df.iloc[-3]
    b = df.iloc[-2]
    c = df.iloc[-1]
    prev_a = df.iloc[-4]

    return (
        is_2dn(a, prev_a)
        and is_inside_bar(b, a)
        and is_2dn(c, b)
    )


# =========================
# TIMEFRAME BUILDERS
# =========================
def tf_frames(daily: pd.DataFrame):
    """Return daily, weekly, monthly frames."""
    d = daily.copy()
    w = resample_ohlc(daily, "W-FRI")
    m = resample_ohlc(daily, "M")

    return d, w, m


# =========================
# FLAG BUILDING
# =========================
def compute_flags(
    d: pd.DataFrame,
    w: pd.DataFrame,
    m: pd.DataFrame
) -> Dict[str, bool]:

    return {
        "D_Bull": strat_bull(d),
        "W_Bull": strat_bull(w),
        "M_Bull": strat_bull(m),

        "D_Bear": strat_bear(d),
        "W_Bear": strat_bear(w),
        "M_Bear": strat_bear(m),

        "D_Inside": strat_inside(d),
        "W_Inside": strat_inside(w),
        "M_Inside": strat_inside(m),

        "D_212Up": strat_212_up(d),
        "W_212Up": strat_212_up(w),

        "D_212Dn": strat_212_dn(d),
        "W_212Dn": strat_212_dn(w),
    }


# =========================
# REGIME SCORING
# =========================
def score_regime(flags: Dict[str, bool]):
    bull = 0
    bear = 0

    bull += 3 if flags["M_Bull"] else 0
    bull += 2 if flags["W_Bull"] else 0
    bull += 1 if flags["D_Bull"] else 0

    bear += 3 if flags["M_Bear"] else 0
    bear += 2 if flags["W_Bear"] else 0
    bear += 1 if flags["D_Bear"] else 0

    bull += 2 if flags["W_212Up"] else 0
    bull += 1 if flags["D_212Up"] else 0

    bear += 2 if flags["W_212Dn"] else 0
    bear += 1 if flags["D_212Dn"] else 0

    return bull, bear


def market_bias_and_strength(rows):
    bull_total = sum(r["BullScore"] for r in rows)
    bear_total = sum(r["BearScore"] for r in rows)

    diff = bull_total - bear_total
    strength = int(max(0, min(100, 50 + diff * 5)))

    if diff >= 3:
        bias = "LONG"
    elif diff <= -3:
        bias = "SHORT"
    else:
        bias = "MIXED"

    return bias, strength, diff


# =========================
# TRIGGER LEVELS
# =========================
def best_trigger(bias, d, w):
    """
    Returns timeframe + entry + stop.
    Weekly preferred.
    """

    if strat_inside(w) and len(w) >= 2:
        cur = w.iloc[-1]
        hi, lo = float(cur["High"]), float(cur["Low"])

        if bias == "LONG":
            return "W", hi, lo
        else:
            return "W", lo, hi

    if strat_inside(d) and len(d) >= 2:
        cur = d.iloc[-1]
        hi, lo = float(cur["High"]), float(cur["Low"])

        if bias == "LONG":
            return "D", hi, lo
        else:
            return "D", lo, hi

    return None, None, None
