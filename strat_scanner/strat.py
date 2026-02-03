# strat_scanner/strat.py — STRAT candle logic + regime scoring

from __future__ import annotations
from typing import Dict, Optional, Tuple

import pandas as pd
from strat_scanner.data import resample_ohlc

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

def last_two(df: pd.DataFrame) -> Optional[Tuple[pd.Series, pd.Series]]:
    if df is None or df.empty or len(df) < 2:
        return None
    return df.iloc[-1], df.iloc[-2]

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

def strat_212_up(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 4:
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    prev_a = df.iloc[-4]
    return is_2up(a, prev_a) and is_inside_bar(b, a) and is_2up(c, b)

def strat_212_dn(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 4:
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    prev_a = df.iloc[-4]
    return is_2dn(a, prev_a) and is_inside_bar(b, a) and is_2dn(c, b)

def tf_frames(daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = daily.copy()
    w = resample_ohlc(daily, "W-FRI")
    m = resample_ohlc(daily, "M")
    return d, w, m

def compute_flags(d: pd.DataFrame, w: pd.DataFrame, m: pd.DataFrame) -> Dict[str, bool]:
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

def score_regime(flags: Dict[str, bool]) -> Tuple[int, int]:
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

def market_bias_and_strength(market_rows) -> Tuple[str, int, int]:
    bull_total = sum(r["BullScore"] for r in market_rows)
    bear_total = sum(r["BearScore"] for r in market_rows)
    diff = bull_total - bear_total

    strength = int(max(0, min(100, 50 + diff * 5)))

    if diff >= 3:
        bias = "LONG"
    elif diff <= -3:
        bias = "SHORT"
    else:
        bias = "MIXED"

    return bias, strength, diff

def alignment_ok(bias: str, flags: Dict[str, bool]) -> bool:
    if bias == "LONG":
        return flags["M_Bull"] or flags["W_Bull"]
    if bias == "SHORT":
        return flags["M_Bear"] or flags["W_Bear"]
    return False

def best_trigger(bias: str, d: pd.DataFrame, w: pd.DataFrame):
    # prefer weekly inside bar
    if strat_inside(w) and len(w) >= 2:
        cur = w.iloc[-1]
        hi, lo = float(cur["High"]), float(cur["Low"])
        return ("W", hi, lo) if bias == "LONG" else ("W", lo, hi)

    if strat_inside(d) and len(d) >= 2:
        cur = d.iloc[-1]
        hi, lo = float(cur["High"]), float(cur["Low"])
        return ("D", hi, lo) if bias == "LONG" else ("D", lo, hi)

    return (None, None, None)

