# strat_scanner/strat.py
# STRAT trigger logic (NO streamlit)

from __future__ import annotations
from typing import Dict, Optional, Union

import pandas as pd


def _inside_bar(cur: pd.Series, prev: pd.Series) -> bool:
    return (cur["High"] <= prev["High"]) and (cur["Low"] >= prev["Low"])


def best_trigger(df: pd.DataFrame, direction: str = "LONG") -> Union[str, Dict]:
    """
    STRAT trigger:
    - If Weekly Inside Bar exists: use that (preferred)
    - Else if Daily Inside Bar exists: use that
    Returns either:
      - dict: {"Status","TF","Entry","Stop"}
      - or a WAIT string
    """
    if df is None or df.empty or len(df) < 5:
        return "WAIT (No Data)"

    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            return "WAIT (Missing OHLC)"

    d = df.dropna(subset=["High", "Low", "Close"]).copy()
    if len(d) < 3:
        return "WAIT (No Data)"

    # Weekly bars
    w = d.resample("W-FRI").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
    ).dropna()

    # 1) Weekly Inside Bar
    if len(w) >= 2:
        cur, prev = w.iloc[-1], w.iloc[-2]
        if _inside_bar(cur, prev):
            hi = float(cur["High"])
            lo = float(cur["Low"])
            if direction.upper() == "SHORT":
                return {"Status": "READY", "TF": "W", "Entry": lo, "Stop": hi}
            return {"Status": "READY", "TF": "W", "Entry": hi, "Stop": lo}

    # 2) Daily Inside Bar
    if len(d) >= 2:
        cur, prev = d.iloc[-1], d.iloc[-2]
        if _inside_bar(cur, prev):
            hi = float(cur["High"])
            lo = float(cur["Low"])
            if direction.upper() == "SHORT":
                return {"Status": "READY", "TF": "D", "Entry": lo, "Stop": hi}
            return {"Status": "READY", "TF": "D", "Entry": hi, "Stop": lo}

    return "WAIT (No Inside Bar)"
