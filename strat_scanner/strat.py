# strat_scanner/strat.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class StratInfo:
    bar: str                 # "1", "2U", "2D", "3"
    is_inside: bool
    is_outside: bool
    trigger: str             # user-facing text


def _bar_type(prev_high: float, prev_low: float, high: float, low: float) -> tuple[str, bool, bool]:
    inside = (high <= prev_high) and (low >= prev_low)
    outside = (high > prev_high) and (low < prev_low)

    if inside:
        return "1", True, False
    if outside:
        return "3", False, True

    # directional 2s
    if (high > prev_high) and (low >= prev_low):
        return "2U", False, False
    if (low < prev_low) and (high <= prev_high):
        return "2D", False, False

    # weird overlap edge case (should be rare)
    return "?", False, False


def strat_signal(df: pd.DataFrame, direction: Optional[str] = None) -> StratInfo:
    """
    STRAT bar classification + simple trigger message.
    direction: "LONG" | "SHORT" | None
    """
    if df is None or df.empty:
        return StratInfo(bar="?", is_inside=False, is_outside=False, trigger="WAIT (No Data)")

    needed = {"High", "Low", "Close", "Open"}
    if not needed.issubset(df.columns):
        return StratInfo(bar="?", is_inside=False, is_outside=False, trigger="WAIT (Missing OHLC)")

    if len(df) < 2:
        return StratInfo(bar="?", is_inside=False, is_outside=False, trigger="WAIT (Not Enough Bars)")

    d = (direction or "").upper().strip()
    if d not in ("LONG", "SHORT"):
        d = "NONE"

    prev = df.iloc[-2]
    cur = df.iloc[-1]

    bar, inside, outside = _bar_type(
        float(prev["High"]), float(prev["Low"]),
        float(cur["High"]), float(cur["Low"])
    )

    # Basic trigger language
    if inside:
        if d == "LONG":
            trig = "WAIT (Inside) — break HIGH"
        elif d == "SHORT":
            trig = "WAIT (Inside) — break LOW"
        else:
            trig = "WAIT (Inside) — wait for break"
        return StratInfo(bar=bar, is_inside=True, is_outside=False, trigger=trig)

    if outside:
        return StratInfo(bar=bar, is_inside=False, is_outside=True, trigger="WAIT (Outside) — confirm next bar")

    if bar == "2U":
        return StratInfo(bar=bar, is_inside=False, is_outside=False, trigger="READY (2U continuation)")
    if bar == "2D":
        return StratInfo(bar=bar, is_inside=False, is_outside=False, trigger="READY (2D continuation)")

    return StratInfo(bar=bar, is_inside=False, is_outside=False, trigger="WAIT")
