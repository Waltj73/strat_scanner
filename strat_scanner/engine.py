# strat_scanner/engine.py — magnitude scoring (RR / ATR% / compression)

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple

import pandas as pd
from strat_scanner.strat import strat_inside

def atr14(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 20:
        return float("nan")
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return float(tr.rolling(14).mean().iloc[-1])

def magnitude_metrics(
    bias: str,
    d: pd.DataFrame,
    entry: Optional[float],
    stop: Optional[float]
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if d is None or d.empty or len(d) < 80:
        return None, None, None, None

    close = float(d["Close"].iloc[-1])
    atr = atr14(d)
    if not math.isfinite(atr) or atr <= 0:
        return None, None, None, None

    atrp = (atr / close) * 100.0

    if entry is None or stop is None:
        return None, atrp, None, None

    hi63 = float(d["High"].rolling(63).max().iloc[-1])
    lo63 = float(d["Low"].rolling(63).min().iloc[-1])

    if bias == "LONG":
        target = hi63
        risk = entry - stop
        reward = target - entry
        room = max(0.0, target - entry)
    else:
        target = lo63
        risk = stop - entry
        reward = entry - target
        room = max(0.0, entry - target)

    rr = None if risk <= 0 else (reward / risk if reward > 0 else 0.0)

    compression = None
    if strat_inside(d) and len(d) >= 2:
        cur = d.iloc[-1]
        rng = float(cur["High"] - cur["Low"])
        compression = rng / atr if atr > 0 else None

    return rr, atrp, room, compression

def calc_scores(
    bias: str,
    flags: Dict[str, bool],
    rr: Optional[float],
    atrp: Optional[float],
    compression: Optional[float],
    entry: Optional[float],
    stop: Optional[float],
) -> Tuple[int, int, int]:
    setup = 0
    mag = 0

    if bias == "LONG":
        setup += 30 if flags["M_Bull"] else 0
        setup += 20 if flags["W_Bull"] else 0
        setup += 10 if flags["D_Bull"] else 0
        setup += 20 if flags["W_212Up"] else 0
        setup += 10 if flags["D_212Up"] else 0
    elif bias == "SHORT":
        setup += 30 if flags["M_Bear"] else 0
        setup += 20 if flags["W_Bear"] else 0
        setup += 10 if flags["D_Bear"] else 0
        setup += 20 if flags["W_212Dn"] else 0
        setup += 10 if flags["D_212Dn"] else 0

    setup += 10 if flags["W_Inside"] else 0
    setup += 5 if flags["D_Inside"] else 0

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
