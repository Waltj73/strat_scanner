# strat_scanner/strat.py
from __future__ import annotations
import pandas as pd

def best_trigger(df: pd.DataFrame) -> dict:
    """
    Minimal stable STRAT trigger logic.
    Returns a dict so engine can safely use it.
    """

    if df is None or df.empty or len(df) < 3:
        return {
            "setup": "None",
            "status": "WAIT",
            "direction": "NONE",
        }

    close = df["Close"]

    # simple momentum check (stable placeholder)
    if close.iloc[-1] > close.iloc[-2] > close.iloc[-3]:
        return {
            "setup": "Momentum",
            "status": "READY",
            "direction": "LONG",
        }

    return {
        "setup": "None",
        "status": "WAIT",
        "direction": "NONE",
    }
