# strat_scanner/strat.py
# STRAT candle typing + setups + actionable trigger logic
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure O/H/L/C columns exist and are floats."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # allow lowercase
    cols = {c.lower(): c for c in out.columns}
    for want in ["open", "high", "low", "close"]:
        if want not in cols:
            return pd.DataFrame()
    out = out[[cols["open"], cols["high"], cols["low"], cols["close"]]].copy()
    out.columns = ["Open", "High", "Low", "Close"]
    for c in ["Open", "High", "Low", "Close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()
    return out


def strat_type(df: pd.DataFrame) -> pd.Series:
    """
    STRAT candle types:
      1  = inside (lower high AND higher low)
      2U = higher high AND higher low
      2D = lower high AND lower low
      3  = higher high AND lower low (outside)
    """
    d = _norm_cols(df)
    if d.empty or len(d) < 2:
        return pd.Series(dtype="object")

    h = d["High"]
    l = d["Low"]
    ph = h.shift(1)
    pl = l.shift(1)

    inside = (h <= ph) & (l >= pl)
    up = (h > ph) & (l >= pl)
    down = (h <= ph) & (l < pl)
    outside = (h > ph) & (l < pl)

    t = pd.Series(index=d.index, dtype="object")
    t[inside] = "1"
    t[up] = "2U"
    t[down] = "2D"
    t[outside] = "3"
    t = t.fillna("—")
    return t


def _atr(d: pd.DataFrame, n: int = 14) -> float:
    """Simple ATR for target guidance."""
    if d is None or d.empty or len(d) < n + 2:
        return float("nan")
    high = d["High"]
    low = d["Low"]
    close = d["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean().iloc[-1]
    return float(atr) if np.isfinite(atr) else float("nan")


def detect_setups(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Detect core STRAT patterns on the last 3 bars:
      - inside_bar: last bar is a 1
      - two_one_two_up: 2U-1-2U (or 2U-1 and current is 2U)
      - two_one_two_down: 2D-1-2D
      - three_one_two_up: 3-1-2U
      - three_one_two_down: 3-1-2D
    """
    d = _norm_cols(df)
    if d.empty or len(d) < 3:
        return {
            "inside_bar": False,
            "two_one_two_up": False,
            "two_one_two_down": False,
            "three_one_two_up": False,
            "three_one_two_down": False,
        }

    t = strat_type(d)
    last3 = list(t.iloc[-3:].values)
    a, b, c = last3[0], last3[1], last3[2]

    inside_bar = (c == "1")
    two_one_two_up = (a == "2U" and b == "1" and c == "2U")
    two_one_two_down = (a == "2D" and b == "1" and c == "2D")
    three_one_two_up = (a == "3" and b == "1" and c == "2U")
    three_one_two_down = (a == "3" and b == "1" and c == "2D")

    return {
        "inside_bar": bool(inside_bar),
        "two_one_two_up": bool(two_one_two_up),
        "two_one_two_down": bool(two_one_two_down),
        "three_one_two_up": bool(three_one_two_up),
        "three_one_two_down": bool(three_one_two_down),
    }


def best_trigger(df: pd.DataFrame) -> Dict[str, object]:
    """
    Produce a single “best” actionable STRAT idea:
      - Direction: LONG / SHORT / NONE
      - Setup: IB / 2-1-2 / 3-1-2 / NONE
      - Status: WAIT / READY
      - Entry/Stop: based on last bar range
    """
    d = _norm_cols(df)
    if d.empty or len(d) < 3:
        return {
            "Setup": "NONE",
            "Direction": "NONE",
            "Status": "WAIT",
            "Entry": float("nan"),
            "Stop": float("nan"),
            "Note": "No data",
        }

    setups = detect_setups(d)
    last = d.iloc[-1]
    prev = d.iloc[-2]

    # Baseline levels (used for triggers)
    long_entry = float(last["High"])   # break last high
    long_stop = float(last["Low"])     # invalidate below last low
    short_entry = float(last["Low"])   # break last low
    short_stop = float(last["High"])   # invalidate above last high

    # Priority: 3-1-2 > 2-1-2 > inside bar
    if setups["three_one_two_up"]:
        return {"Setup": "3-1-2", "Direction": "LONG", "Status": "READY", "Entry": long_entry, "Stop": long_stop,
                "Note": "3-1-2U breakout idea (break last high)."}
    if setups["three_one_two_down"]:
        return {"Setup": "3-1-2", "Direction": "SHORT", "Status": "READY", "Entry": short_entry, "Stop": short_stop,
                "Note": "3-1-2D breakdown idea (break last low)."}

    if setups["two_one_two_up"]:
        return {"Setup": "2-1-2", "Direction": "LONG", "Status": "READY", "Entry": long_entry, "Stop": long_stop,
                "Note": "2-1-2U continuation idea (break last high)."}
    if setups["two_one_two_down"]:
        return {"Setup": "2-1-2", "Direction": "SHORT", "Status": "READY", "Entry": short_entry, "Stop": short_stop,
                "Note": "2-1-2D continuation idea (break last low)."}

    # Inside bar -> “ready” only if compression (range smaller than prior)
    is_inside = strat_type(d).iloc[-1] == "1"
    if is_inside:
        rng = float(last["High"] - last["Low"])
        prng = float(prev["High"] - prev["Low"])
        status = "READY" if (rng <= prng) else "WAIT"
        return {"Setup": "IB", "Direction": "BOTH", "Status": status, "Entry": float(last["High"]), "Stop": float(last["Low"]),
                "Note": "Inside bar. Long above high / short below low."}

    return {"Setup": "NONE", "Direction": "NONE", "Status": "WAIT", "Entry": float("nan"), "Stop": float("nan"),
            "Note": "No clean STRAT trigger on last 3 bars."}


def targets_from_entry(entry: float, direction: str, atr: float) -> Tuple[float, float]:
    """
    Very light target guidance (not pretending to be perfect):
    uses ATR multiples if available.
    """
    if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
        return float("nan"), float("nan")

    if direction == "LONG":
        return float(entry + 1.0 * atr), float(entry + 2.0 * atr)
    if direction == "SHORT":
        return float(entry - 1.0 * atr), float(entry - 2.0 * atr)
    return float("nan"), float("nan")
