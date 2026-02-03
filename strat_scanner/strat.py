from __future__ import annotations
import pandas as pd

def best_trigger(close: pd.Series) -> str:
    """
    Placeholder: you can expand this later.
    For now, return a stable label used everywhere.
    """
    if close is None or close.empty or len(close) < 3:
        return "WAIT (No Data)"
    # Simple momentum read
    if close.iloc[-1] > close.iloc[-2] > close.iloc[-3]:
        return "READY"
    return "WAIT"
