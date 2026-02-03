# strat_scanner/strat.py
# STRAT candle logic + trigger selection (Daily + Weekly)
#
# This module is pure logic (no streamlit imports).
# It provides:
# - Resample daily OHLC to weekly
# - Candle classification (1 / 2U / 2D / 3)
# - Pattern detection (Inside, 2-1-2 up/down, basic continuations)
# - A "best_trigger" that returns TF + Entry/Stop + Status

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd


# -------------------------
# Data structures
# -------------------------
@dataclass
class Trigger:
    tf: str                   # "D" or "W"
    pattern: str              # e.g. "Inside", "2-1-2 Up"
    status: str               # "READY" or "WAIT (No Trigger)"
    direction: str            # "LONG" or "SHORT"
    entry: Optional[float]    # entry price
    stop: Optional[float]     # stop price


# -------------------------
# Resampling
# -------------------------
def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def resample_ohlc(df: pd.DataFrame, rule: str = "W-FRI") -> pd.DataFrame:
    """
    Resample OHLCV using safe first/last for Open/Close, max/min for High/Low, sum for Volume.
    """
    df = _ensure_dt_index(df)
    if df.empty:
        return pd.DataFrame()

    needed = ["Open", "High", "Low", "Close"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    def safe_first(x):
        x = x.dropna()
        return x.iloc[0] if len(x) else np.nan

    def safe_last(x):
        x = x.dropna()
        return x.iloc[-1] if len(x) else np.nan

    g = df.resample(rule)
    out = pd.DataFrame({
        "Open": g["Open"].apply(safe_first),
        "High": g["High"].max(),
        "Low": g["Low"].min(),
        "Close": g["Close"].apply(safe_last),
    })

    if "Volume" in df.columns:
        out["Volume"] = g["Volume"].sum()

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out


# -------------------------
# Candle classification
# -------------------------
def candle_type(cur: pd.Series, prev: pd.Series) -> str:
    """
    Strat candle types:
    1  = inside bar
    2U = directional up (takes prev high, doesn't take prev low)
    2D = directional down (takes prev low, doesn't take prev high)
    3  = outside bar (takes both high and low)
    """
    ch, cl = float(cur["High"]), float(cur["Low"])
    ph, pl = float(prev["High"]), float(prev["Low"])

    inside = (ch <= ph) and (cl >= pl)
    outside = (ch > ph) and (cl < pl)
    two_up = (ch > ph) and (cl >= pl)
    two_dn = (cl < pl) and (ch <= ph)

    if inside:
        return "1"
    if outside:
        return "3"
    if two_up:
        return "2U"
    if two_dn:
        return "2D"
    # fallback (rare equality edge cases)
    return "1"


def is_green(cur: pd.Series) -> bool:
    return float(cur["Close"]) > float(cur["Open"])

def is_red(cur: pd.Series) -> bool:
    return float(cur["Close"]) < float(cur["Open"])

def _last_n(df: pd.DataFrame, n: int) -> Optional[pd.DataFrame]:
    if df is None or df.empty or len(df) < n:
        return None
    return df.iloc[-n:].copy()


# -------------------------
# Pattern detection
# -------------------------
def detect_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Returns booleans for key patterns.
    """
    out = {
        "Inside": False,
        "2U": False,
        "2D": False,
        "212Up": False,
        "212Dn": False,
    }

    df2 = _last_n(df, 2)
    if df2 is None:
        return out

    cur, prev = df2.iloc[-1], df2.iloc[-2]
    ct = candle_type(cur, prev)

    out["Inside"] = (ct == "1")
    out["2U"] = (ct == "2U")
    out["2D"] = (ct == "2D")

    # 2-1-2 needs last 3 bars + one bar before them to define first "2" vs its prev
    df4 = _last_n(df, 4)
    if df4 is not None:
        a = df4.iloc[-3]   # first "2"
        b = df4.iloc[-2]   # "1"
        c = df4.iloc[-1]   # second "2"
        p = df4.iloc[-4]   # bar before a

        a_type = candle_type(a, p)
        b_type = candle_type(b, a)
        c_type = candle_type(c, b)

        out["212Up"] = (a_type == "2U" and b_type == "1" and c_type == "2U")
        out["212Dn"] = (a_type == "2D" and b_type == "1" and c_type == "2D")

    return out


# -------------------------
# Trigger selection (the part your app needs)
# -------------------------
def _inside_trigger(df: pd.DataFrame, tf: str, direction: str) -> Optional[Trigger]:
    """
    If latest bar is inside bar, return entry/stop for the break.
    LONG: entry = inside high, stop = inside low
    SHORT: entry = inside low, stop = inside high
    """
    df2 = _last_n(df, 2)
    if df2 is None:
        return None
    cur, prev = df2.iloc[-1], df2.iloc[-2]
    if candle_type(cur, prev) != "1":
        return None

    hi, lo = float(cur["High"]), float(cur["Low"])
    if direction == "LONG":
        return Trigger(tf=tf, pattern="Inside", status="READY", direction="LONG", entry=hi, stop=lo)
    else:
        return Trigger(tf=tf, pattern="Inside", status="READY", direction="SHORT", entry=lo, stop=hi)


def _continuation_trigger(df: pd.DataFrame, tf: str, direction: str) -> Optional[Trigger]:
    """
    Basic continuation:
    If last bar is 2U (for LONG) or 2D (for SHORT), suggest trigger on break of that bar.
    This is less "clean" than an inside bar trigger, so we label it as WAIT unless you want it READY.
    """
    df2 = _last_n(df, 2)
    if df2 is None:
        return None
    cur, prev = df2.iloc[-1], df2.iloc[-2]
    ct = candle_type(cur, prev)
    hi, lo = float(cur["High"]), float(cur["Low"])

    if direction == "LONG" and ct == "2U":
        return Trigger(tf=tf, pattern="2U Continuation", status="WAIT (No Inside Bar)", direction="LONG", entry=hi, stop=lo)
    if direction == "SHORT" and ct == "2D":
        return Trigger(tf=tf, pattern="2D Continuation", status="WAIT (No Inside Bar)", direction="SHORT", entry=lo, stop=hi)

    return None


def best_trigger(
    daily_df: pd.DataFrame,
    direction: str = "LONG",
    prefer_weekly: bool = True
) -> Trigger:
    """
    Choose the best Strat trigger:
    1) Weekly inside bar (highest priority)
    2) Daily inside bar
    3) Otherwise return WAIT with best-effort context (like 2U/2D continuation)
    """
    direction = direction.upper().strip()
    if direction not in ("LONG", "SHORT"):
        direction = "LONG"

    d = _ensure_dt_index(daily_df)
    if d.empty or len(d) < 5:
        return Trigger(tf="D", pattern="None", status="WAIT (No Data)", direction=direction, entry=None, stop=None)

    w = resample_ohlc(d, "W-FRI")

    # Prefer weekly triggers
    if prefer_weekly and not w.empty and len(w) >= 3:
        trig = _inside_trigger(w, tf="W", direction=direction)
        if trig:
            return trig

    # Daily inside
    trig = _inside_trigger(d, tf="D", direction=direction)
    if trig:
        return trig

    # Fallback context (still WAIT)
    if prefer_weekly and not w.empty and len(w) >= 3:
        cont = _continuation_trigger(w, tf="W", direction=direction)
        if cont:
            return cont

    cont = _continuation_trigger(d, tf="D", direction=direction)
    if cont:
        return cont

    return Trigger(tf="D", pattern="None", status="WAIT (No Trigger)", direction=direction, entry=None, stop=None)
