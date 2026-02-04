from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import pandas as pd


def _as_df(x: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Accept either OHLCV DF or Close series.
    If series, we can't do STRAT. We'll return minimal DF.
    """
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, pd.Series):
        # fake minimal, STRAT will degrade gracefully
        return pd.DataFrame({"Close": x})
    return pd.DataFrame()


def candle_type(prev_h: float, prev_l: float, h: float, l: float) -> str:
    """
    STRAT bar types relative to previous bar:
    1 = inside
    2U = breaks up only
    2D = breaks down only
    3 = outside
    """
    broke_up = h > prev_h
    broke_dn = l < prev_l

    if broke_up and broke_dn:
        return "3"
    if broke_up and not broke_dn:
        return "2U"
    if broke_dn and not broke_up:
        return "2D"
    return "1"


@dataclass
class StratRead:
    last_type: str
    prev_type: str
    trigger: str


def strat_read(df: pd.DataFrame) -> Optional[StratRead]:
    df = _as_df(df)
    if df.empty:
        return None

    # Need OHLC for real strat. If missing, degrade to "WAIT"
    required = {"High", "Low", "Close"}
    if not required.issubset(set(df.columns)):
        if "Close" in df.columns and len(df["Close"]) >= 3:
            c = df["Close"].dropna()
            if len(c) >= 3 and (c.iloc[-1] > c.iloc[-2] > c.iloc[-3]):
                return StratRead(last_type="n/a", prev_type="n/a", trigger="READY (momentum)")
        return StratRead(last_type="n/a", prev_type="n/a", trigger="WAIT")

    d = df.dropna(subset=["High", "Low", "Close"]).copy()
    if len(d) < 3:
        return StratRead(last_type="n/a", prev_type="n/a", trigger="WAIT")

    prev = d.iloc[-2]
    last = d.iloc[-1]
    prevprev = d.iloc[-3]

    prev_type = candle_type(prevprev["High"], prevprev["Low"], prev["High"], prev["Low"])
    last_type = candle_type(prev["High"], prev["Low"], last["High"], last["Low"])

    # Very practical trigger rules:
    # - Inside bar last -> WAIT (needs break)
    # - 2U last -> READY long
    # - 2D last -> READY short
    # - 3 last -> WAIT (wide, messy) unless closes strong
    trigger = "WAIT"
    if last_type == "1":
        trigger = "WAIT (inside bar)"
    elif last_type == "2U":
        trigger = "READY (2U)"
    elif last_type == "2D":
        trigger = "READY (2D)"
    elif last_type == "3":
        # outside bar: only "ready" if close is near extreme
        rng = float(last["High"] - last["Low"])
        if rng > 0:
            pos = float((last["Close"] - last["Low"]) / rng)
            if pos >= 0.8:
                trigger = "READY (3 close high)"
            elif pos <= 0.2:
                trigger = "READY (3 close low)"
            else:
                trigger = "WAIT (3 bar)"
        else:
            trigger = "WAIT (3 bar)"

    return StratRead(last_type=last_type, prev_type=prev_type, trigger=trigger)


def best_trigger(df_or_close: Union[pd.DataFrame, pd.Series], direction: Optional[str] = None) -> str:
    """
    Safe: accepts DF or Close Series.
    direction is optional; if given, we filter "READY" to match direction.
    """
    sr = strat_read(df_or_close)
    if sr is None:
        return "WAIT"

    trig = sr.trigger

    if direction in ("LONG", "SHORT"):
        if direction == "LONG" and "2D" in trig:
            return "WAIT (direction mismatch)"
        if direction == "SHORT" and "2U" in trig:
            return "WAIT (direction mismatch)"

    return trig


def strat_flags(df: pd.DataFrame) -> Dict[str, str]:
    """
    Handy flags for UI / writeups.
    """
    sr = strat_read(df)
    if sr is None:
        return {"Last": "n/a", "Prev": "n/a", "Trigger": "WAIT"}
    return {"Last": sr.last_type, "Prev": sr.prev_type, "Trigger": sr.trigger}
