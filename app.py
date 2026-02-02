# app.py
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =========================
# STREAMLIT CONFIG + NAV
# =========================
st.set_page_config(page_title="STRAT Regime Scanner", layout="wide")

page = st.sidebar.radio("Navigation", ["Scanner", "📘 User Guide"])

# =========================
# USER GUIDE PAGE
# =========================
def show_user_guide():
    st.title("📘 STRAT Regime Scanner — Complete User Guide")

    st.markdown("""
## What this scanner is for
This tool helps you **stop guessing** and trade with structure:

1) **Market direction** (risk-on vs risk-off)  
2) **Sector strength** (where money is rotating)  
3) **Stock selection** (best setups with triggers + RR + magnitude)

---

## The order matters
**Market → Sector → Stock → Setup → Trigger**

If you skip steps, you’ll end up taking random trades that don’t align with the environment.

---

## Section 1: Filters (how to use them)
**ONLY Inside Bars (D or W)**  
- Shows stocks that have a clean **breakout trigger**.
- Best when you want “set orders + walk away” trades.

**ONLY 2-1-2 forming (bias direction)**  
- Shows continuation setups forming.
- Great in trending conditions.

**Require Monthly OR Weekly alignment (bias direction)** *(recommended ON)*  
- Filters out weak names.
- Keeps you trading with the larger timeframe.

**Top Picks count**  
- Controls how many top candidates appear.

**Refresh data button**  
- Clears cache and pulls fresh data immediately.

---

## Section 2: Market Regime (your “trade direction”)
The Market Regime table checks:
- **SPY / QQQ / IWM / DIA**
- Daily / Weekly / Monthly bull vs bear
- 2-1-2 patterns

### How to read bias
If the market is:
- **LONG** → prioritize long setups only
- **SHORT** → prioritize short setups only
- **MIXED** → be selective or reduce size

Strength (0–100) is a simple “how clear is the bias?” gauge.

---

## Section 3: Sector Ranking (where the edge is)
Sectors are ranked **after bias is known**.

### How to use it
- In **LONG** bias: focus the **top 1–3 sectors**
- In **SHORT** bias: focus the **weakest sectors** (bear leadership)

This is how you catch “leaders” instead of random names.

---

## Section 4: Drilldown (how to pick actual trades)
Pick a sector → scan the tickers inside it.

The scanner ranks candidates by:
- **SetupScore** = alignment + STRAT structure + trigger readiness
- **MagScore** = movement potential (ATR%) + room-to-run + compression
- **TotalScore** = SetupScore + MagScore

---

## Section 5: Trigger Levels (Entry / Stop)
When the scanner shows TF/Entry/Stop, treat it like this:

**LONG**
- Entry = break above inside bar high
- Stop = below inside bar low

**SHORT**
- Entry = break below inside bar low
- Stop = above inside bar high

Weekly triggers are preferred (cleaner swings).

---

## Section 6: RR / Room / ATR%
These help you avoid “dead trades.”

- **RR**: estimated reward-to-risk (higher is better)
- **Room**: distance to a practical extreme (63-day high/low)
- **ATR%**: how much the ticker typically moves (too low = slow)

Good swing candidates usually have:
- RR **≥ 2**
- ATR% **≥ 1** (higher is fine as long as structure is clean)

---

## Trade of the Day
This is simply:
- best ranked setup
- that also has a valid trigger (Entry/Stop)
- and RR meets your threshold

If it says “none”, that’s a feature: **don’t force it.**

---

## Daily routine (2–5 minutes)
1) Open app  
2) Check **Bias + Strength**  
3) Note **top 1–3 sectors**  
4) Drill into strongest sector  
5) Pick **1–3 names** with Entry/Stop and RR ≥ 2  
6) Confirm chart sanity (wicks, gaps, earnings)  
7) Place trigger orders (or wait)

---

## When NOT to trade
- Bias is MIXED and strength is low
- No inside triggers exist
- The name is gapping wildly / news-driven
- Earnings within a few days (optional filter later)

---

## Bottom line
This scanner’s job is not to “predict” — it’s to:
- **filter noise**
- **keep you aligned**
- **give clean trigger levels**
- **keep you patient**
""")

# If guide selected, show it and stop
if page == "📘 User Guide":
    show_user_guide()
    st.stop()

# =========================
# UI CONTROLS
# =========================
st.title("STRAT Regime Scanner (Auto LONG/SHORT + Magnitude)")
st.caption("Bias from market regime. Ranks tickers by setup quality AND magnitude (RR + ATR% + compression).")

with st.expander("Filters", expanded=True):
    colA, colB, colC, colD = st.columns([1.1, 1.2, 1.6, 1.1])

    with colA:
        only_inside = st.checkbox("ONLY Inside Bars (D or W)", value=False)
    with colB:
        only_212 = st.checkbox("ONLY 2-1-2 forming (bias direction)", value=False)
    with colC:
        require_alignment = st.checkbox("Require Monthly OR Weekly alignment (bias direction)", value=True)
    with colD:
        top_k = st.slider("Top Picks count", min_value=3, max_value=8, value=5)

    colR1, colR2 = st.columns([1, 3])
    with colR1:
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# =========================
# UNIVERSE
# =========================
MARKET_ETFS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Russell 2000": "IWM",
    "Dow Jones": "DIA",
}

SECTOR_ETFS = {
    "Energy": "XLE",
    "Comm Services": "XLC",
    "Staples": "XLP",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Discretionary": "XLY",
    "Utilities": "XLU",
    "Financials": "XLF",
    "Technology": "XLK",
    "Health Care": "XLV",
}

SECTOR_TICKERS: Dict[str, List[str]] = {
    "Energy": ["XOM","CVX","COP","EOG","SLB","HAL","PSX","MPC","VLO","OXY","KMI","WMB","BKR","DVN","PXD"],
    "Comm Services": ["GOOGL","GOOG","META","NFLX","TMUS","VZ","T","DIS","CMCSA","CHTR","EA","TTWO","SPOT","ROKU","SNAP"],
    "Staples": ["PG","KO","PEP","WMT","COST","PM","MO","MDLZ","CL","KMB","GIS","KHC","SYY","HSY","EL"],
    "Materials": ["LIN","APD","SHW","NUE","DOW","PPG","ECL","FCX","NEM","IFF","MLM","VMC","ALB","MOS","DD"],
    "Industrials": ["CAT","DE","HON","GE","LMT","RTX","BA","UNP","UPS","FDX","ETN","EMR","CSX","NSC","WM"],
    "Real Estate": ["PLD","AMT","EQIX","PSA","O","WELL","DLR","SPG","CCI","VICI","AVB","EQR","IRM","SBAC","EXR"],
    "Discretionary": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","GM","F","MAR","ROST","ORLY","CMG"],
    "Utilities": ["NEE","DUK","SO","D","AEP","EXC","XEL","SRE","ED","PEG","EIX","PCG","WEC","ES","AWK"],
    "Financials": ["BRK-B","JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","SPGI","ICE","CME","PNC","TFC"],
    "Technology": ["AAPL","MSFT","NVDA","AVGO","CRM","ORCL","ADBE","AMD","CSCO","INTC","QCOM","TXN","NOW","AMAT","MU"],
    "Health Care": ["UNH","JNJ","LLY","PFE","MRK","ABBV","TMO","ABT","DHR","BMY","AMGN","GILD","ISRG","VRTX","MDT"],
}

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

# =========================
# DATA FETCH (CACHED) — HARDENED
# =========================
def _flatten_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)

        if set(REQUIRED_COLS).issubset(set(lvl0)):
            if ticker in set(lvl1):
                df = df.xs(ticker, axis=1, level=1, drop_level=True)
            else:
                df.columns = [c[0] for c in df.columns]
        elif set(REQUIRED_COLS).issubset(set(lvl1)):
            if ticker in set(lvl0):
                df = df.xs(ticker, axis=1, level=0, drop_level=True)
            else:
                df.columns = [c[1] for c in df.columns]
        else:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        rename_map = {}
        for c in df.columns:
            if isinstance(c, str):
                lc = c.lower()
                if lc == "open": rename_map[c] = "Open"
                elif lc == "high": rename_map[c] = "High"
                elif lc == "low": rename_map[c] = "Low"
                elif lc == "close": rename_map[c] = "Close"
                elif lc == "volume": rename_map[c] = "Volume"
        if rename_map:
            df = df.rename(columns=rename_map)

    if "Volume" not in df.columns:
        df["Volume"] = 0

    if not set(REQUIRED_COLS).issubset(set(df.columns)):
        return pd.DataFrame()

    df = df[REQUIRED_COLS].dropna(subset=["Open","High","Low","Close"])
    return df


@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_hist(ticker: str, period: str = "3y") -> pd.DataFrame:
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    return _flatten_yf_columns(raw, ticker)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not set(REQUIRED_COLS).issubset(df.columns):
        return pd.DataFrame()

    ohlc = df[["Open", "High", "Low", "Close"]].resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    })
    vol = df[["Volume"]].resample(rule).sum()
    out = pd.concat([ohlc, vol], axis=1).dropna(subset=["Open","High","Low","Close"])
    return out

# =========================
# STRAT HELPERS
# =========================
def is_inside_bar(cur: pd.Series, prev: pd.Series) -> bool:
    return (cur["High"] <= prev["High"]) and (cur["Low"] >= prev["Low"])

def is_2up(cur: pd.Series, prev: pd.Series) -> bool:
    return (cur["High"] > prev["High"]) and (cur["Low"] >= prev["Low"])

def is_2dn(cur: pd.Series, prev: pd.Series) -> bool:
    return (cur["Low"] < prev["Low"]) and (cur["High"] <= prev["High"])

def is_green(cur: pd.Series) -> bool:
    return cur["Close"] > cur["Open"]

def is_red(cur: pd.Series) -> bool:
    return cur["Close"] < cur["Open"]

def last_two(df: pd.DataFrame) -> Optional[Tuple[pd.Series, pd.Series]]:
    if df is None or df.empty or len(df) < 2:
        return None
    return df.iloc[-1], df.iloc[-2]

def strat_bull(df: pd.DataFrame) -> bool:
    pair = last_two(df)
    if not pair:
        return False
    cur, prev = pair
    return is_2up(cur, prev) and is_green(cur)

def strat_bear(df: pd.DataFrame) -> bool:
    pair = last_two(df)
    if not pair:
        return False
    cur, prev = pair
    return is_2dn(cur, prev) and is_red(cur)

def strat_inside(df: pd.DataFrame) -> bool:
    pair = last_two(df)
    if not pair:
        return False
    cur, prev = pair
    return is_inside_bar(cur, prev)

def strat_212_up(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 4:
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    prev_a = df.iloc[-4]
    return is_2up(a, prev_a) and is_inside_bar(b, a) and is_2up(c, b)

def strat_212_dn(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 4:
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    prev_a = df.iloc[-4]
    return is_2dn(a, prev_a) and is_inside_bar(b, a) and is_2dn(c, b)

def atr14(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 20:
        return float("nan")
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return float(tr.rolling(14).mean().iloc[-1])

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# =========================
# FEATURE BUILDERS
# =========================
def tf_frames(daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = daily.copy()
    w = resample_ohlc(daily, "W-FRI")
    m = resample_ohlc(daily, "M")
    return d, w, m

def compute_flags(d: pd.DataFrame, w: pd.DataFrame, m: pd.DataFrame) -> Dict[str, bool]:
    return {
        "D_Bull": strat_bull(d),
        "W_Bull": strat_bull(w),
        "M_Bull": strat_bull(m),
        "D_Bear": strat_bear(d),
        "W_Bear": strat_bear(w),
        "M_Bear": strat_bear(m),
        "D_Inside": strat_inside(d),
        "W_Inside": strat_inside(w),
        "M_Inside": strat_inside(m),
        "D_212Up": strat_212_up(d),
        "W_212Up": strat_212_up(w),
        "D_212Dn": strat_212_dn(d),
        "W_212Dn": strat_212_dn(w),
    }

def score_regime(flags: Dict[str, bool]) -> Tuple[int, int]:
    bull = 0
    bear = 0
    bull += 3 if flags["M_Bull"] else 0
    bull += 2 if flags["W_Bull"] else 0
    bull += 1 if flags["D_Bull"] else 0
    bear += 3 if flags["M_Bear"] else 0
    bear += 2 if flags["W_Bear"] else 0
    bear += 1 if flags["D_Bear"] else 0
    bull += 2 if flags["W_212Up"] else 0
    bull += 1 if flags["D_212Up"] else 0
    bear += 2 if flags["W_212Dn"] else 0
    bear += 1 if flags["D_212Dn"] else 0
    return bull, bear

def market_bias_and_strength(market_rows: List[Dict]) -> Tuple[str, int, int]:
    bull_total = sum(r["BullScore"] for r in market_rows)
    bear_total = sum(r["BearScore"] for r in market_rows)
    diff = bull_total - bear_total
    strength = int(clamp(50 + diff * 5, 0, 100))
    if diff >= 3:
        bias = "LONG"
    elif diff <= -3:
        bias = "SHORT"
    else:
        bias = "MIXED"
    return bias, strength, diff

def alignment_ok(bias: str, flags: Dict[str, bool]) -> bool:
    if bias == "LONG":
        return flags["M_Bull"] or flags["W_Bull"]
    if bias == "SHORT":
        return flags["M_Bear"] or flags["W_Bear"]
    return False

def best_trigger(bias: str, d: pd.DataFrame, w: pd.DataFrame) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    if strat_inside(w) and len(w) >= 2:
        cur = w.iloc[-1]
        hi, lo = float(cur["High"]), float(cur["Low"])
        if bias == "LONG":
            return "W", hi, lo
        if bias == "SHORT":
            return "W", lo, hi

    if strat_inside(d) and len(d) >= 2:
        cur = d.iloc[-1]
        hi, lo = float(cur["High"]), float(cur["Low"])
        if bias == "LONG":
            return "D", hi, lo
        if bias == "SHORT":
            return "D", lo, hi

    return None, None, None

def magnitude_metrics(bias: str, d: pd.DataFrame, entry: Optional[float], stop: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if entry is None or stop is None or d is None or d.empty or len(d) < 80:
        return None, None, None, None

    close = float(d["Close"].iloc[-1])
    atr = atr14(d)
    if not math.isfinite(atr) or atr <= 0:
        return None, None, None, None

    atrp = (atr / close) * 100.0
    hi63 = float(d["High"].rolling(63).max().iloc[-1])
    lo63 = float(d["Low"].rolling(63).min().iloc[-1])

    if bias == "LONG":
        target = hi63
        room = max(0.0, target - entry)
        risk = entry - stop
        reward = target - entry
    else:
        target = lo63
        room = max(0.0, entry - target)
        risk = stop - entry
        reward = entry - target

    rr = None if risk <= 0 else (reward / risk if reward > 0 else 0.0)

    compression = None
    if strat_inside(d) and len(d) >= 2:
        cur = d.iloc[-1]
        rng = float(cur["High"] - cur["Low"])
        compression = rng / atr if atr > 0 else None

    return rr, atrp, room, compression

def calc_scores(
    bias: str,
    flags: Dict[str, bool],
    rr: Optional[float],
    atrp: Optional[float],
    compression: Optional[float],
    entry: Optional[float],
    stop: Optional[float],
) -> Tuple[int, int, int]:
    setup = 0
    mag = 0

    if bias == "LONG":
        setup += 30 if flags["M_Bull"] else 0
        setup += 20 if flags["W_Bull"] else 0
        setup += 10 if flags["D_Bull"] else 0
        setup += 20 if flags["W_212Up"] else 0
        setup += 10 if flags["D_212Up"] else 0
    elif bias == "SHORT":
        setup += 30 if flags["M_Bear"] else 0
        setup += 20 if flags["W_Bear"] else 0
        setup += 10 if flags["D_Bear"] else 0
        setup += 20 if flags["W_212Dn"] else 0
        setup += 10 if flags["D_212Dn"] else 0

    setup += 10 if flags["W_Inside"] else 0
    setup += 5 if flags["D_Inside"] else 0

    if rr is not None:
        if rr >= 3:
            mag += 35
        elif rr >= 2:
            mag += 25
        elif rr >= 1.5:
            mag += 10

    if atrp is not None:
        if atrp >= 3:
            mag += 20
        elif atrp >= 2:
            mag += 10
        elif atrp >= 1:
            mag += 5

    if compression is not None:
        if compression <= 0.6:
            mag += 15
        elif compression <= 0.9:
            mag += 8
        elif compression <= 1.2:
            mag += 3

    if entry is not None and stop is not None:
        mag += 5

    total = setup + mag
    return setup, mag, total

# =========================
# BUILD MARKET REGIME
# =========================
market_rows: List[Dict] = []
for name, etf in MARKET_ETFS.items():
    d = get_hist(etf)
    if d.empty:
        flags = {k: False for k in [
            "D_Bull","W_Bull","M_Bull","D_Bear","W_Bear","M_Bear",
            "D_Inside","W_Inside","M_Inside","D_212Up","W_212Up","D_212Dn","W_212Dn"
        ]}
        bull, bear = 0, 0
    else:
        d_tf, w_tf, m_tf = tf_frames(d)
        flags = compute_flags(d_tf, w_tf, m_tf)
        bull, bear = score_regime(flags)

    row = {"Market": name, "ETF": etf, "BullScore": bull, "BearScore": bear}
    row.update(flags)
    market_rows.append(row)

bias, strength, bull_bear_diff = market_bias_and_strength(market_rows)

st.subheader("Market Regime (SPY / QQQ / IWM / DIA) — Bull vs Bear")
market_df = pd.DataFrame(market_rows)[[
    "Market","ETF",
    "D_Bull","W_Bull","M_Bull",
    "D_Bear","W_Bear","M_Bear",
    "D_212Up","W_212Up","D_212Dn","W_212Dn"
]]
st.dataframe(market_df, use_container_width=True, hide_index=True)

# =========================
# BUILD SECTOR TABLE
# =========================
sector_rows: List[Dict] = []
for sector, etf in SECTOR_ETFS.items():
    d = get_hist(etf)
    if d.empty:
        flags = {k: False for k in [
            "D_Bull","W_Bull","M_Bull","D_Bear","W_Bear","M_Bear",
            "D_Inside","W_Inside","M_Inside","D_212Up","W_212Up","D_212Dn","W_212Dn"
        ]}
        bull, bear = 0, 0
    else:
        d_tf, w_tf, m_tf = tf_frames(d)
        flags = compute_flags(d_tf, w_tf, m_tf)
        bull, bear = score_regime(flags)

    row = {"Sector": sector, "ETF": etf, "BullScore": bull, "BearScore": bear}
    row.update(flags)
    sector_rows.append(row)

sectors_df = pd.DataFrame(sector_rows)

if bias == "LONG":
    sectors_df = sectors_df.sort_values(["BullScore","BearScore"], ascending=[False, True])
elif bias == "SHORT":
    sectors_df = sectors_df.sort_values(["BearScore","BullScore"], ascending=[False, True])
else:
    sectors_df["Dominance"] = (sectors_df["BullScore"] - sectors_df["BearScore"]).abs()
    sectors_df = sectors_df.sort_values("Dominance", ascending=False)

st.subheader("Sectors (SPDR) — ranked after bias is known")
st.dataframe(
    sectors_df[[
        "Sector","ETF","BullScore","BearScore",
        "D_Bull","W_Bull","M_Bull","D_Bear","W_Bear","M_Bear",
        "D_Inside","W_Inside","M_Inside",
        "D_212Up","W_212Up","D_212Dn","W_212Dn"
    ]],
    use_container_width=True,
    hide_index=True
)

# =========================
# DRILLDOWN: TOP NAMES
# =========================
st.subheader("Drill into a sector (ranks candidates in bias direction + magnitude)")

sector_choice = st.selectbox("Choose a sector:", options=list(SECTOR_TICKERS.keys()), index=0)
tickers = SECTOR_TICKERS.get(sector_choice, [])
st.write(f"Selected: **{sector_choice}** ({SECTOR_ETFS.get(sector_choice,'')}) — tickers in list: **{len(tickers)}**")

scan_n = st.slider("How many tickers to scan", min_value=5, max_value=len(tickers), value=min(15, len(tickers)))
scan_list = tickers[:scan_n]

cand_rows: List[Dict] = []
for t in scan_list:
    d = get_hist(t)
    if d.empty:
        continue

    d_tf, w_tf, m_tf = tf_frames(d)
    flags = compute_flags(d_tf, w_tf, m_tf)

    if require_alignment and bias in ("LONG","SHORT") and not alignment_ok(bias, flags):
        continue

    if only_inside and not (flags["D_Inside"] or flags["W_Inside"]):
        continue

    if only_212:
        if bias == "LONG" and not (flags["D_212Up"] or flags["W_212Up"]):
            continue
        if bias == "SHORT" and not (flags["D_212Dn"] or flags["W_212Dn"]):
            continue

    eff_bias = bias if bias in ("LONG","SHORT") else "LONG"

    tf, entry, stop = best_trigger(eff_bias, d_tf, w_tf)
    rr, atrp, room, compression = magnitude_metrics(eff_bias, d_tf, entry, stop)

    setup_score, mag_score, total_score = calc_scores(eff_bias, flags, rr, atrp, compression, entry, stop)

    cand = {
        "Ticker": t,
        "SetupScore": setup_score,
        "MagScore": mag_score,
        "TotalScore": total_score,
        "TF": tf,
        "Entry": None if entry is None else round(float(entry), 2),
        "Stop": None if stop is None else round(float(stop), 2),
        "Room": None if room is None else round(float(room), 2),
        "RR": None if rr is None else round(float(rr), 2),
        "ATR%": None if atrp is None else round(float(atrp), 2),
    }
    cand.update(flags)
    cand_rows.append(cand)

cand_df = pd.DataFrame(cand_rows)
if cand_df.empty:
    st.info("No matches under current filters. Loosen filters (or market is in drift/chop).")
else:
    cand_df = cand_df.sort_values("TotalScore", ascending=False)

    st.markdown(f"### Top Trade Ideas (best {top_k}) — Bias: **{bias}** (ranked by TotalScore)")
    top_df = cand_df.head(top_k)[[
        "Ticker","TotalScore","SetupScore","MagScore","TF","Entry","Stop","Room","RR","ATR%",
        "W_212Up","D_212Up","M_Bull","W_Bull","D_Bull","W_Inside","D_Inside",
        "W_212Dn","D_212Dn","M_Bear","W_Bear","D_Bear"
    ]]
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    st.markdown("### 🎯 Trade of the Day (best TotalScore + valid trigger)")
    valid = cand_df.dropna(subset=["Entry","Stop","RR"]).copy()
    valid = valid[valid["RR"] >= 2.0]
    if valid.empty:
        st.warning("No valid trigger found (needs Inside Bar levels). Use Top Ideas and wait for an Inside Bar trigger.")
    else:
        best = valid.iloc[0]
        st.success(
            f"**{best['Ticker']}** | Bias: **{bias}** | TF: **{best['TF']}** | "
            f"Entry: **{best['Entry']}** | Stop: **{best['Stop']}** | "
            f"RR: **{best['RR']}** | ATR%: **{best['ATR%']}**"
        )

    st.markdown("### All Matches (ranked by TotalScore)")
    st.dataframe(
        cand_df[[
            "Ticker","SetupScore","MagScore","TotalScore","TF","Entry","Stop","Room","RR","ATR%",
            "W_Inside","D_Inside","W_212Up","D_212Up","W_212Dn","D_212Dn",
            "M_Bull","W_Bull","D_Bull","M_Bear","W_Bear","D_Bear"
        ]],
        use_container_width=True,
        hide_index=True
    )

# =========================
# QUICK MARKET READ
# =========================
st.subheader("Quick Market Read")

if bias == "LONG":
    rotation_in = [f"{r['Sector']}({r['ETF']})" for _, r in sectors_df.head(3).iterrows()]
    rotation_out = [f"{r['Sector']}({r['ETF']})" for _, r in sectors_df.tail(3).iterrows()]
elif bias == "SHORT":
    rotation_in = [f"{r['Sector']}({r['ETF']})" for _, r in sectors_df.head(3).iterrows()]
    rotation_out = [f"{r['Sector']}({r['ETF']})" for _, r in sectors_df.tail(3).iterrows()]
else:
    rotation_in = []
    rotation_out = []

if bias == "MIXED" or strength < 50:
    plan = "Plan: Defensive. Trade smaller, or wait for A+ triggers (inside bars / clean 2-1-2)."
    badge = "🟠"
elif bias == "LONG":
    plan = "Plan: LONG only. Prioritize Weekly 2-1-2 UP + Inside Bar breaks. Prefer higher RR + ATR%."
    badge = "🟢"
else:
    plan = "Plan: SHORT only. Prioritize Weekly 2-1-2 DN + Inside Bar breaks. Prefer higher RR + ATR%."
    badge = "🔴"

st.write(f"Bias: **{bias}** {badge} | Strength: **{strength}/100** | Bull–Bear diff: **{bull_bear_diff}**")

if rotation_in and rotation_out:
    st.write(f"Rotation IN: {', '.join(rotation_in)} | OUT: {', '.join(rotation_out)}")

st.success(plan)

st.caption(
    "Trigger logic: If Inside Bar exists, LONG = buy break of High / stop below Low. "
    "SHORT = sell break of Low / stop above High. Weekly triggers are preferred when available."
)
