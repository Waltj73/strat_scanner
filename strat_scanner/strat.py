# strat_scanner/strat.py
from __future__ import annotations

from typing import Any, Dict, Optional, Union
import pandas as pd


def best_trigger(
    data: Union[pd.DataFrame, pd.Series],
    direction: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, str]:
    """
    Backward-compatible STRAT trigger.

    Works with BOTH call styles:
      - best_trigger(df)
      - best_trigger(df, direction="LONG")   <-- your engine is doing this

    Also works if someone accidentally passes just a Close series.

    Returns a stable dict used by engine/pages.
    """

    # -------------------------
    # Normalize inputs
    # -------------------------
    if data is None:
        return {"setup": "None", "status": "WAIT", "direction": "NONE"}

    close = None

    # If DataFrame, use Close column
    if isinstance(data, pd.DataFrame):
        if data.empty or "Close" not in data.columns:
            return {"setup": "None", "status": "WAIT", "direction": "NONE"}
        close = data["Close"].dropna()

    # If Series, treat it as Close series
    elif isinstance(data, pd.Series):
        close = data.dropna()

    else:
        return {"setup": "None", "status": "WAIT", "direction": "NONE"}

    if close is None or close.empty or len(close) < 3:
        return {"setup": "None", "status": "WAIT", "direction": "NONE"}

    # -------------------------
    # Simple, stable trigger logic
    # (placeholder you can expand later)
    # -------------------------
    up3 = close.iloc[-1] > close.iloc[-2] > close.iloc[-3]
    down3 = close.iloc[-1] < close.iloc[-2] < close.iloc[-3]

    if up3:
        return {"setup": "Momentum", "status": "READY", "direction": "LONG"}
    if down3:
        return {"setup": "Momentum", "status": "READY", "direction": "SHORT"}

    return {"setup": "None", "status": "WAIT", "direction": "NONE"}
