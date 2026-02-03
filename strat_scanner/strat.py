import pandas as pd

def best_trigger(df: pd.DataFrame) -> str:
    """
    Simple STRAT candle logic (core engine)
    """
    if df is None or df.empty or len(df) < 3:
        return "WAIT"

    h = df["High"]
    l = df["Low"]

    def candle(i):
        if h.iloc[i] > h.iloc[i-1] and l.iloc[i] < l.iloc[i-1]:
            return "3"
        if h.iloc[i] > h.iloc[i-1] and l.iloc[i] >= l.iloc[i-1]:
            return "2U"
        if h.iloc[i] <= h.iloc[i-1] and l.iloc[i] < l.iloc[i-1]:
            return "2D"
        return "1"

    c1 = candle(-3)
    c2 = candle(-2)
    c3 = candle(-1)

    if c1 == "1" and c2 == "2U":
        return "1-2 Break (Long)"
    if c1 == "1" and c2 == "2D":
        return "1-2 Break (Short)"

    if c2 == "2U" and c3 == "2U":
        return "2-2 Continuation (Long)"
    if c2 == "2D" and c3 == "2D":
        return "2-2 Continuation (Short)"

    return f"{c1}-{c2}-{c3}"
