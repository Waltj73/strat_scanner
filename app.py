import pandas as pd
import yfinance as yf
import streamlit as st

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="STRAT Regime Scanner", layout="wide")

# -----------------------------
# UNIVERSE
# -----------------------------
MARKET_ETFS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Russell 2000": "IWM",
    "Dow Jones": "DIA",
}

SECTOR_ETFS = {
    "Materials": "XLB",
    "Comm Services": "XLC",
    "Energy": "XLE",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Technology": "XLK",
    "Staples": "XLP",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Health Care": "XLV",
    "Discretionary": "XLY",
}

# Stable drill-down lists: top liquid names per sector
SECTOR_TOP_TICKERS = {
    "XLK": ["AAPL","MSFT","NVDA","AVGO","AMD","ADBE","CRM","INTC","CSCO","QCOM","ORCL","NOW","TXN","AMAT","MU"],
    "XLF": ["JPM","BAC","WFC","GS","MS","C","SCHW","AXP","BLK","USB","PNC","TFC","CB","AIG","COF"],
    "XLE": ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","KMI","HES","DVN","BKR","HAL","PXD"],
    "XLV": ["LLY","UNH","JNJ","MRK","ABBV","PFE","TMO","DHR","ABT","BMY","AMGN","ISRG","GILD","VRTX","MDT"],
    "XLI": ["CAT","DE","GE","RTX","BA","HON","ETN","UPS","LMT","MMM","NOC","EMR","ITW","GD","CSX"],
    "XLY": ["AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","BKNG","TGT","F","GM","MAR","RCL","CMG","EBAY"],
    "XLP": ["PG","KO","PEP","WMT","COST","PM","MDLZ","CL","KMB","KR","MO","HSY","KHC","WBA","MNST"],
    "XLU": ["NEE","DUK","SO","D","AEP","EXC","SRE","PEG","XEL","ED","EIX","WEC","AWK","CMS","ES"],
    "XLB": ["LIN","APD","ECL","SHW","FCX","NEM","DOW","DD","VMC","MLM","NUE","ALB","IFF","PPG","STLD"],
    "XLRE":["PLD","AMT","EQIX","CCI","PSA","O","SPG","WELL","AVB","DLR","VTR","EQR","CBRE","IRM","ARE"],
    "XLC": ["GOOGL","META","NFLX","TMUS","VZ","T","DIS","CHTR","CMCSA","EA","ROKU","TTWO","WBD","FOXA","PARA"],
}

# -----------------------------
# HELPERS
# -----------------------------
def _scalar(x):
    try:
        return float(x.item())
    except Exception:
        return float(x)

def bool_to_int(b):
    return 1 if bool(b) else 0

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["Open"].resample(rule).first()
    h = df["High"].resample(rule).max()
    l = df["Low"].resample(rule).min()
    c = df["Close"].resample(rule).last()
    out = pd.concat([o, h, l, c], axis=1).dropna()
    out.columns = ["Open", "High", "Low", "Close"]
    return out

@st.cache_data(ttl=60 * 60)
def get_prices(ticker: str, period: str = "24mo") -> pd.DataFrame:
    d = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if d is None or d.empty:
        return pd.DataFrame()

    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

    d.columns = [str(c).title() for c in d.columns]
    keep = [c for c in ["Open", "High", "Low", "Close"] if c in d.columns]
    d = d[keep].dropna()

    for c in keep:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    return d.dropna()

def strat_bar_type_last(df: pd.DataFrame) -> str:
    """Return last bar type relative to previous: 1 / 2U / 2D / 3"""
    if df is None or df.empty or len(df) < 2:
        return ""
    cur = df.iloc[-1]
    prev = df.iloc[-2]

    curH, prevH = _scalar(cur["High"]), _scalar(prev["High"])
    curL, prevL = _scalar(cur["Low"]), _scalar(prev["Low"])

    inside = (curH <= prevH) and (curL >= prevL)
    outside = (curH > prevH) and (curL < prevL)
    two_up = (curH > prevH) and (curL >= prevL)
    two_dn = (curL < prevL) and (curH <= prevH)

    if inside: return "1"
    if outside: return "3"
    if two_up: return "2U"
    if two_dn: return "2D"
    return ""

def is_inside_last(df: pd.DataFrame) -> bool:
    return strat_bar_type_last(df) == "1"

def inside_trigger_levels(df: pd.DataFrame):
    """If LAST bar is inside bar (1), return (high, low). Else (None, None)."""
    if df is None or df.empty or len(df) < 2:
        return (None, None)
    if strat_bar_type_last(df) != "1":
        return (None, None)
    cur = df.iloc[-1]
    return (_scalar(cur["High"]), _scalar(cur["Low"]))

def green_2u_last(df: pd.DataFrame) -> bool:
    """Bull: Green 2Up on LAST bar."""
    if df is None or df.empty or len(df) < 2:
        return False
    cur = df.iloc[-1]
    prev = df.iloc[-2]

    curH = _scalar(cur["High"]);  prevH = _scalar(prev["High"])
    curL = _scalar(cur["Low"]);   prevL = _scalar(prev["Low"])
    curC = _scalar(cur["Close"]); curO = _scalar(cur["Open"])

    is_2u = (curH > prevH) and (curL >= prevL)
    is_green = (curC > curO)
    return bool(is_2u and is_green)

def red_2d_last(df: pd.DataFrame) -> bool:
    """Bear: Red 2Down on LAST bar."""
    if df is None or df.empty or len(df) < 2:
        return False
    cur = df.iloc[-1]
    prev = df.iloc[-2]

    curH = _scalar(cur["High"]);  prevH = _scalar(prev["High"])
    curL = _scalar(cur["Low"]);   prevL = _scalar(prev["Low"])
    curC = _scalar(cur["Close"]); curO = _scalar(cur["Open"])

    is_2d = (curL < prevL) and (curH <= prevH)
    is_red = (curC < curO)
    return bool(is_2d and is_red)

def setup_212_up_forming(df: pd.DataFrame) -> bool:
    """2-1-2 UP forming: previous = 2U, current = 1."""
    if df is None or df.empty or len(df) < 3:
        return False
    cur_type = strat_bar_type_last(df)
    prev_type = strat_bar_type_last(df.iloc[:-1])
    return (prev_type == "2U") and (cur_type == "1")

def setup_212_dn_forming(df: pd.DataFrame) -> bool:
    """2-1-2 DOWN forming: previous = 2D, current = 1."""
    if df is None or df.empty or len(df) < 3:
        return False
    cur_type = strat_bar_type_last(df)
    prev_type = strat_bar_type_last(df.iloc[:-1])
    return (prev_type == "2D") and (cur_type == "1")

def atr14(df: pd.DataFrame, length: int = 14):
    """ATR using True Range, returns last ATR value or None."""
    if df is None or df.empty or len(df) < length + 2:
        return None
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(length).mean()
    v = atr.iloc[-1]
    return None if pd.isna(v) else float(v)

def signals_all(ticker: str) -> dict:
    """Returns both bull + bear signals + inside triggers."""
    d = get_prices(ticker)
    if d.empty or len(d) < 60:
        return {
            "D_Bull": False, "W_Bull": False, "M_Bull": False,
            "D_Bear": False, "W_Bear": False, "M_Bear": False,
            "D_Inside": False, "W_Inside": False, "M_Inside": False,
            "D_212Up": False, "W_212Up": False,
            "D_212Dn": False, "W_212Dn": False,
            "D_TrigHigh": None, "D_TrigLow": None,
            "W_TrigHigh": None, "W_TrigLow": None,
        }

    w = resample_ohlc(d, "W-FRI")
    m = resample_ohlc(d, "M")

    D_bull = green_2u_last(d)
    W_bull = green_2u_last(w)
    M_bull = green_2u_last(m)

    D_bear = red_2d_last(d)
    W_bear = red_2d_last(w)
    M_bear = red_2d_last(m)

    D_inside = is_inside_last(d)
    W_inside = is_inside_last(w)
    M_inside = is_inside_last(m)

    D_212u = setup_212_up_forming(d)
    W_212u = setup_212_up_forming(w)

    D_212d = setup_212_dn_forming(d)
    W_212d = setup_212_dn_forming(w)

    d_hi, d_lo = inside_trigger_levels(d)
    w_hi, w_lo = inside_trigger_levels(w)

    return {
        "D_Bull": D_bull, "W_Bull": W_bull, "M_Bull": M_bull,
        "D_Bear": D_bear, "W_Bear": W_bear, "M_Bear": M_bear,
        "D_Inside": D_inside, "W_Inside": W_inside, "M_Inside": M_inside,
        "D_212Up": D_212u, "W_212Up": W_212u,
        "D_212Dn": D_212d, "W_212Dn": W_212d,
        "D_TrigHigh": d_hi, "D_TrigLow": d_lo,
        "W_TrigHigh": w_hi, "W_TrigLow": w_lo,
    }

def market_bias_from_df(market_df: pd.DataFrame):
    """Bias based on BullScore - BearScore across SPY/QQQ/IWM/DIA."""
    if market_df is None or market_df.empty:
        return ("MIXED", 50, 0)

    diffs = []
    mags = []
    for _, r in market_df.iterrows():
        bull = (
            35 * bool_to_int(r.get("M_Bull", False)) +
            30 * bool_to_int(r.get("W_Bull", False)) +
            15 * bool_to_int(r.get("D_Bull", False)) +
            10 * bool_to_int(r.get("W_212Up", False)) +
            5  * bool_to_int(r.get("D_212Up", False))
        )
        bear = (
            35 * bool_to_int(r.get("M_Bear", False)) +
            30 * bool_to_int(r.get("W_Bear", False)) +
            15 * bool_to_int(r.get("D_Bear", False)) +
            10 * bool_to_int(r.get("W_212Dn", False)) +
            5  * bool_to_int(r.get("D_212Dn", False))
        )
        diffs.append(bull - bear)
        mags.append(max(bull, bear))

    diff = sum(diffs) / max(1, len(diffs))
    strength = sum(mags) / max(1, len(mags))

    if diff >= 15:
        return ("LONG", int(round(strength)), diff)
    elif diff <= -15:
        return ("SHORT", int(round(strength)), diff)
    else:
        return ("MIXED", int(round(strength)), diff)

def rank_score(sig: dict, bias: str) -> int:
    """Setup/Direction score (not magnitude)."""
    score = 0
    if bias == "SHORT":
        if sig["W_212Dn"]: score += 100
        if sig["D_212Dn"]: score += 60
        if sig["M_Bear"]:  score += 30
        if sig["W_Bear"]:  score += 25
        if sig["D_Bear"]:  score += 10
        if sig["W_Inside"]: score += 20
        if sig["D_Inside"]: score += 10
        if sig["M_Inside"]: score += 10
    else:
        if sig["W_212Up"]: score += 100
        if sig["D_212Up"]: score += 60
        if sig["M_Bull"]:  score += 30
        if sig["W_Bull"]:  score += 25
        if sig["D_Bull"]:  score += 10
        if sig["W_Inside"]: score += 20
        if sig["D_Inside"]: score += 10
        if sig["M_Inside"]: score += 10
    return score

def pick_entry_stop(sig: dict, bias: str):
    """
    Uses inside bar triggers if present.
    LONG: entry = trig high, stop = trig low
    SHORT: entry = trig low, stop = trig high
    Prefer weekly inside triggers; else daily.
    """
    w_hi, w_lo = sig.get("W_TrigHigh"), sig.get("W_TrigLow")
    d_hi, d_lo = sig.get("D_TrigHigh"), sig.get("D_TrigLow")

    use_weekly = (w_hi is not None) and (w_lo is not None) and sig.get("W_Inside", False)
    use_daily = (d_hi is not None) and (d_lo is not None) and sig.get("D_Inside", False)

    if use_weekly:
        tf, hi, lo = "W", w_hi, w_lo
    elif use_daily:
        tf, hi, lo = "D", d_hi, d_lo
    else:
        return (None, None, None)

    if bias == "SHORT":
        return (tf, lo, hi)   # entry low, stop high
    return (tf, hi, lo)       # entry high, stop low

def magnitude_metrics(ticker: str, bias: str, entry: float, stop: float, tf: str, sig: dict):
    """
    Magnitude = room-to-run vs risk + does it move enough (ATR%).
    - Target uses recent 63 trading days (approx 3 months) high/low.
    """
    d = get_prices(ticker)
    if d is None or d.empty or entry is None or stop is None:
        return (None, None, None, None, 0)

    close = float(d["Close"].iloc[-1])
    atr = atr14(d, 14)
    atr_pct = (atr / close) * 100 if (atr is not None and close > 0) else None

    lookback = 63 if len(d) >= 63 else len(d)
    recent = d.tail(lookback)

    recent_high = float(recent["High"].max())
    recent_low = float(recent["Low"].min())

    risk = abs(entry - stop)
    if risk <= 0:
        return (atr, atr_pct, None, None, 0)

    if bias == "SHORT":
        room = entry - recent_low
    else:
        room = recent_high - entry

    rr = room / risk

    # Compression factor if we used inside bar triggers
    bar_range = None
    if tf == "W" and sig.get("W_TrigHigh") is not None and sig.get("W_TrigLow") is not None:
        bar_range = float(sig["W_TrigHigh"] - sig["W_TrigLow"])
    if tf == "D" and sig.get("D_TrigHigh") is not None and sig.get("D_TrigLow") is not None:
        bar_range = float(sig["D_TrigHigh"] - sig["D_TrigLow"])

    # Magnitude score (simple + practical)
    mag_score = 0

    # RR scoring
    if rr >= 3.0:
        mag_score += 40
    elif rr >= 2.0:
        mag_score += 25
    elif rr >= 1.5:
        mag_score += 15
    elif rr >= 1.0:
        mag_score += 5

    # ATR% scoring (options need movement)
    if atr_pct is not None:
        if atr_pct >= 3.0:
            mag_score += 10
        elif atr_pct >= 2.0:
            mag_score += 7
        elif atr_pct >= 1.0:
            mag_score += 4

    # Compression scoring (inside bar tightness vs ATR)
    if bar_range is not None and atr is not None and atr > 0:
        comp = bar_range / atr
        if comp <= 0.75:
            mag_score += 10
        elif comp <= 1.0:
            mag_score += 6
        elif comp <= 1.25:
            mag_score += 3

    return (atr, atr_pct, room, rr, mag_score)

def top_sectors_text(sector_df: pd.DataFrame, bias: str, n: int = 3) -> str:
    if sector_df is None or sector_df.empty:
        return "Strong Sectors: n/a"
    df = sector_df.copy()

    if bias == "SHORT":
        df["Strength"] = (
            3 * df["BearScore"] +
            4 * df["W_212Dn"].astype(int) +
            2 * df["D_212Dn"].astype(int) +
            2 * df["W_Inside"].astype(int) +
            1 * df["D_Inside"].astype(int)
        )
    else:
        df["Strength"] = (
            3 * df["BullScore"] +
            4 * df["W_212Up"].astype(int) +
            2 * df["D_212Up"].astype(int) +
            2 * df["W_Inside"].astype(int) +
            1 * df["D_Inside"].astype(int)
        )

    leaders = df.sort_values("Strength", ascending=False).head(n)
    txt = ", ".join([f"{r['Sector']}({r['ETF']})" for _, r in leaders.iterrows()])
    return f"Strong Sectors: {txt}"

def rotation_text(sector_df: pd.DataFrame, bias: str, n: int = 3) -> str:
    if sector_df is None or sector_df.empty:
        return "Rotation: n/a"
    df = sector_df.copy()

    if bias == "SHORT":
        df["FreshIn"] = (
            2 * df["W_Bear"].astype(int) +
            1 * df["D_Bear"].astype(int) +
            2 * df["W_212Dn"].astype(int) +
            1 * df["D_212Dn"].astype(int) -
            1 * df["M_Bear"].astype(int)
        )
        df["WeakOut"] = (
            2 * (1 - df["W_Bear"].astype(int)) +
            1 * (1 - df["D_Bear"].astype(int)) +
            2 * (1 - df["W_212Dn"].astype(int)) +
            1 * (1 - df["D_212Dn"].astype(int))
        ) + (df["M_Bear"].astype(int))
    else:
        df["FreshIn"] = (
            2 * df["W_Bull"].astype(int) +
            1 * df["D_Bull"].astype(int) +
            2 * df["W_212Up"].astype(int) +
            1 * df["D_212Up"].astype(int) -
            1 * df["M_Bull"].astype(int)
        )
        df["WeakOut"] = (
            2 * (1 - df["W_Bull"].astype(int)) +
            1 * (1 - df["D_Bull"].astype(int)) +
            2 * (1 - df["W_212Up"].astype(int)) +
            1 * (1 - df["D_212Up"].astype(int))
        ) + (df["M_Bull"].astype(int))

    rot_in = df.sort_values("FreshIn", ascending=False).head(n)
    rot_out = df.sort_values("WeakOut", ascending=False).head(n)

    in_txt = ", ".join([f"{r['Sector']}({r['ETF']})" for _, r in rot_in.iterrows()])
    out_txt = ", ".join([f"{r['Sector']}({r['ETF']})" for _, r in rot_out.iterrows()])

    return f"Rotation IN: {in_txt} | OUT: {out_txt}"

def get_holdings(etf_ticker: str) -> list[str]:
    return SECTOR_TOP_TICKERS.get(etf_ticker, [])

def best_trade_candidate(df: pd.DataFrame):
    """Best trade uses TotalScore and requires Entry/Stop."""
    if df is None or df.empty:
        return None
    valid = df.dropna(subset=["Entry", "Stop", "TotalScore"])
    if valid.empty:
        return None
    return valid.sort_values("TotalScore", ascending=False).iloc[0]

# -----------------------------
# UI
# -----------------------------
st.title("STRAT Regime Scanner (Auto LONG/SHORT + Magnitude)")
st.caption("Bias from market regime. Ranks tickers by setup quality AND magnitude (RR + ATR% + compression).")

with st.expander("Filters", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    only_inside = c1.checkbox("ONLY Inside Bars (D or W)", value=False)
    only_212 = c2.checkbox("ONLY 2-1-2 forming (bias direction)", value=False)
    require_bias = c3.checkbox("Require Monthly OR Weekly alignment (bias direction)", value=True)
    top_n = c4.slider("Top Picks count", 3, 8, 5, 1)

st.divider()

# -----------------------------
# MARKET TABLE
# -----------------------------
st.subheader("Market Regime (SPY / QQQ / IWM / DIA) — Bull vs Bear")
market_rows = []
for name, ticker in MARKET_ETFS.items():
    s = signals_all(ticker)
    market_rows.append({
        "Market": name, "ETF": ticker,
        "D_Bull": s["D_Bull"], "W_Bull": s["W_Bull"], "M_Bull": s["M_Bull"],
        "D_Bear": s["D_Bear"], "W_Bear": s["W_Bear"], "M_Bear": s["M_Bear"],
        "D_212Up": s["D_212Up"], "W_212Up": s["W_212Up"],
        "D_212Dn": s["D_212Dn"], "W_212Dn": s["W_212Dn"],
    })
market_df = pd.DataFrame(market_rows)
st.dataframe(market_df, use_container_width=True)

bias, strength, diff = market_bias_from_df(market_df)

st.divider()

# -----------------------------
# SECTOR TABLE
# -----------------------------
st.subheader("Sectors (SPDR) — ranked after bias is known")
sector_rows = []
for sector_name, etf in SECTOR_ETFS.items():
    s = signals_all(etf)
    bull_score = int(s["D_Bull"]) + int(s["W_Bull"]) + int(s["M_Bull"])
    bear_score = int(s["D_Bear"]) + int(s["W_Bear"]) + int(s["M_Bear"])
    sector_rows.append({
        "Sector": sector_name, "ETF": etf,
        "BullScore": bull_score, "BearScore": bear_score,
        "D_Bull": s["D_Bull"], "W_Bull": s["W_Bull"], "M_Bull": s["M_Bull"],
        "D_Bear": s["D_Bear"], "W_Bear": s["W_Bear"], "M_Bear": s["M_Bear"],
        "D_Inside": s["D_Inside"], "W_Inside": s["W_Inside"], "M_Inside": s["M_Inside"],
        "D_212Up": s["D_212Up"], "W_212Up": s["W_212Up"],
        "D_212Dn": s["D_212Dn"], "W_212Dn": s["W_212Dn"],
    })

sector_df = pd.DataFrame(sector_rows)
if bias == "SHORT":
    sector_df = sector_df.sort_values(["BearScore", "M_Bear", "W_Bear", "D_Bear"], ascending=False)
else:
    sector_df = sector_df.sort_values(["BullScore", "M_Bull", "W_Bull", "D_Bull"], ascending=False)

st.dataframe(sector_df, use_container_width=True)

st.divider()

# -----------------------------
# DRILLDOWN
# -----------------------------
st.subheader("Drill into a sector (ranks candidates in bias direction + magnitude)")
choice = st.selectbox("Choose a sector:", options=list(SECTOR_ETFS.keys()))
etf = SECTOR_ETFS[choice]
tickers = get_holdings(etf)

st.write(f"Bias: **{bias}** | Strength: **{strength}/100** | Bull-Bear diff: **{diff:.0f}**")
st.write(f"Selected: **{choice} ({etf})** — tickers in list: **{len(tickers)}**")

max_names = st.slider("How many tickers to scan", 5, len(tickers), min(15, len(tickers)), step=5)

with st.spinner("Scanning tickers..."):
    rows = []
    for t in tickers[:max_names]:
        sig = signals_all(t)
        setup_score = rank_score(sig, bias)

        if require_bias:
            if bias == "SHORT":
                if not (sig["M_Bear"] or sig["W_Bear"]):
                    continue
            else:
                if not (sig["M_Bull"] or sig["W_Bull"]):
                    continue

        if only_inside and not (sig["D_Inside"] or sig["W_Inside"]):
            continue

        if only_212:
            if bias == "SHORT" and not (sig["D_212Dn"] or sig["W_212Dn"]):
                continue
            if bias != "SHORT" and not (sig["D_212Up"] or sig["W_212Up"]):
                continue

        tf, entry, stop = pick_entry_stop(sig, bias)

        # Only keep candidates relevant to bias
        if bias == "SHORT":
            relevant = (
                sig["D_Bear"] or sig["W_Bear"] or sig["M_Bear"] or
                sig["D_212Dn"] or sig["W_212Dn"] or
                sig["D_Inside"] or sig["W_Inside"]
            )
        else:
            relevant = (
                sig["D_Bull"] or sig["W_Bull"] or sig["M_Bull"] or
                sig["D_212Up"] or sig["W_212Up"] or
                sig["D_Inside"] or sig["W_Inside"]
            )

        if not relevant:
            continue

        atr, atr_pct, room, rr, mag_score = (None, None, None, None, 0)
        if entry is not None and stop is not None and tf is not None:
            atr, atr_pct, room, rr, mag_score = magnitude_metrics(t, bias, entry, stop, tf, sig)

        total = setup_score + mag_score

        rows.append({
            "Ticker": t,
            "SetupScore": setup_score,
            "MagScore": mag_score,
            "TotalScore": total,
            "TF": tf,
            "Entry": entry,
            "Stop": stop,
            "Room": room,
            "RR": rr,
            "ATR%": atr_pct,

            # bias-relevant flags
            "W_Inside": sig["W_Inside"],
            "D_Inside": sig["D_Inside"],
            "W_212Up": sig["W_212Up"],
            "D_212Up": sig["D_212Up"],
            "W_212Dn": sig["W_212Dn"],
            "D_212Dn": sig["D_212Dn"],
            "M_Bull": sig["M_Bull"],
            "W_Bull": sig["W_Bull"],
            "D_Bull": sig["D_Bull"],
            "M_Bear": sig["M_Bear"],
            "W_Bear": sig["W_Bear"],
            "D_Bear": sig["D_Bear"],
        })

if not rows:
    st.info("No matches found with your filters. Try scanning more tickers or loosening alignment requirement.")
else:
    names_df = pd.DataFrame(rows).sort_values(["TotalScore"], ascending=False)

    st.subheader(f"Top Trade Ideas (best {top_n}) — Bias: {bias} (ranked by TotalScore)")
    top_df = names_df.head(top_n).copy()

    if bias == "SHORT":
        cols = ["Ticker","TotalScore","SetupScore","MagScore","TF","Entry","Stop","Room","RR","ATR%","W_212Dn","D_212Dn","M_Bear","W_Bear","D_Bear","W_Inside","D_Inside"]
    else:
        cols = ["Ticker","TotalScore","SetupScore","MagScore","TF","Entry","Stop","Room","RR","ATR%","W_212Up","D_212Up","M_Bull","W_Bull","D_Bull","W_Inside","D_Inside"]

    st.dataframe(top_df[cols], use_container_width=True)

    st.divider()
    st.subheader("🎯 Trade of the Day (best TotalScore + valid trigger)")

    best = best_trade_candidate(names_df)
    if best is None or pd.isna(best.get("Entry")) or pd.isna(best.get("Stop")):
        st.info("No valid trigger found (needs Inside Bar levels). Use Top Ideas and wait for an Inside Bar trigger.")
    else:
        direction = "SHORT" if bias == "SHORT" else "LONG"
        entry = float(best["Entry"]); stop = float(best["Stop"])
        rr = best["RR"]; atrp = best["ATR%"]; room = best["Room"]
        rr_txt = f"{rr:.2f}R" if pd.notna(rr) else "n/a"
        atr_txt = f"{atrp:.2f}%" if pd.notna(atrp) else "n/a"
        room_txt = f"{room:.2f}" if pd.notna(room) else "n/a"
        st.success(
            f"{direction} {best['Ticker']} | Entry: {entry:.2f} | Stop: {stop:.2f} | "
            f"TF: {best['TF']} | Room: {room_txt} | RR: {rr_txt} | ATR%: {atr_txt}"
        )

    st.subheader("All Matches (ranked by TotalScore)")
    st.dataframe(names_df, use_container_width=True)

# -----------------------------
# QUICK MARKET READ
# -----------------------------
st.divider()
st.subheader("Quick Market Read")

if bias == "LONG":
    st.write(f"Bias: **LONG ✅** | Strength: **{strength}/100**")
elif bias == "SHORT":
    st.write(f"Bias: **SHORT 🛑** | Strength: **{strength}/100**")
else:
    st.write(f"Bias: **MIXED ⚠️** | Strength: **{strength}/100**")

st.write(f"**{top_sectors_text(sector_df, bias=bias, n=3)}**")
st.write(f"**{rotation_text(sector_df, bias=bias, n=3)}**")

if bias == "LONG":
    st.success("Plan: LONG only. Prioritize Weekly 2-1-2 UP + Inside Bar breaks. Prefer higher RR + ATR%.")
elif bias == "SHORT":
    st.error("Plan: SHORT only. Prioritize Weekly 2-1-2 DOWN + Inside Bar breakdowns. Prefer higher RR + ATR%.")
else:
    st.warning("Plan: Mixed tape. Trade fewer setups or wait for a decisive LONG/SHORT bias.")

st.caption(
    "Magnitude = Room-to-run vs Risk (RR) + ATR% (movement) + Compression (inside bar tightness vs ATR). "
    "Targets use the last ~63 trading days high/low as a practical reference."
)
