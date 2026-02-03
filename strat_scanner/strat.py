from __future__ import annotations
import pandas as pd


def best_trigger(close: pd.Series) -> str:
    """
    Stable STRAT trigger placeholder.

    Input:
        close price series

    Output:
        simple STRAT readiness label
    """

    if close is None or close.empty or len(close) < 3:
        return "WAIT"

    # simple momentum logic placeholder
    if close.iloc[-1] > close.iloc[-2] > close.iloc[-3]:
        return "READY"

    return "WAIT"
