# strat_scanner/strat.py
from __future__ import annotations

import pandas as pd


def best_trigger(close: pd.Series, *args, **kwargs) -> str:
    """
    STRAT trigger placeholder (stable).
    IMPORTANT: accepts *args/**kwargs so engine can never break it again.

    Returns a simple string status used by UI:
      - "READY"
      - "WAIT (No Inside Bar)"
      - "WAIT"
    """
    if close is None or not isinstance(close, pd.Series) or close.dropna().empty:
        return "WAIT (No Data)"

    c = close.dropna()
    if len(c) < 3:
        return "WAIT"

    # Simple “inside bar” style placeholder:
    # We don't have High/Low here, so we use Close compression as a proxy.
    # (You can upgrade later to true STRAT 1-2-3 logic when stable.)
    last = c.iloc[-1]
    prev = c.iloc[-2]
    prev2 = c.iloc[-3]

    # Momentum “ready” signal
    if last > prev > prev2:
        return "READY"

    return "WAIT (No Inside Bar)"
