import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="STRAT Regime Scanner", layout="wide")

APP_TITLE = "STRAT Regime Scanner (Auto LONG/SHORT + Magnitude)"
APP_SUB = "Bias from market regime. Ranks tickers by setup quality AND magnitude (RR + ATR% + compression)."

# =========================
# UNIVERSES
# =========================
MARKET_ETFS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Russell 2000": "IWM",
    "Dow Jones": "DIA",
}

SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# Hard-coded liquid lists (stable; no ETF-holdings scraping failures)
SECTOR_TICKERS = {
    "XLK": ["AAPL","MSFT","NVDA","AVGO","CRM","ADBE","ORCL","CSCO","AMD","QCOM","TXN","INTU","NOW","AMAT","MU"],
    "XLF": ["JPM","BAC","WFC","C","GS","MS","BLK","SCHW","AXP","SPGI","ICE","CME","CB","PNC","AIG"],
    "XLV": ["UNH","LLY","JNJ","ABBV","MRK","PFE","TMO","ABT","DHR","BMY","AMGN","ISRG","GILD","VRTX","MDT"],
    "XLE": ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","KMI","WMB","HAL","BKR","DVN","APA"],
    "XLY": ["AMZN","TSLA","HD","MCD","LOW","BKNG","NKE","SBUX","TJX","TGT","GM","F","ROST","MAR","CMG"],
    "XLP": ["PG","KO","PEP","WMT","COST","PM","MO","MDLZ","CL","KMB","GIS","KHC","SYY","HSY","MNST"],
    "XLI": ["CAT","BA","HON","GE","RTX","LMT","DE","UPS","UNP","ETN","WM","ADP","MMM","EMR","NOC"],
    "XLB": ["LIN","SHW","NUE","FCX","ECL","DOW","PPG","APD","MLM","VMC","IFF","NEM","ALB","DD","CTVA"],
    "XLU": ["NEE","DUK","SO","D","AEP","EXC","SRE","XEL","ED","PEG","WEC","EIX","ES","PPL","DTE"],
    "XLRE": ["PLD","AMT","EQIX","O","WELL","PSA","CCI","SPG","DLR","VICI","AVB","EQR","INVH","BXP","SBAC"],
    "XLC": ["GOOGL","GOOG","META","NFLX","DIS","TMUS","T","VZ","CMCSA","CHTR","TTWO","EA","WBD","PARA","SPOT"],
}

# =========================
# DATA HELPERS (ROBUST)
# =========================
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

def normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame()

    # MultiIndex handling (yfinance can return MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)

        if {"Open","High","Low","Close"}.issubset(set(lvl0)):
            # (Field, Ticker)
            if ticker in set(lvl1):
                df = df.xs(ticker, axis=1, level=1, drop_level=True)
            else:
                df.columns = [c[0] for c in df.columns]
        elif {"Open","High","Low","Close"}.issubset(set(lvl1)):
            # (Ticker, Field)
            if ticker in set(lvl0):
                df = df.xs(ticker, axis=1, level=0, drop_level=True)
            else:
                df.columns = [c[1] for c in df.columns]
        else:
            df.columns = ["_".join([str(x) for x in c if x is not None]) for c in df.columns]

    # Normalize case
    ren = {}
    for c in df.columns:
        if isinstance(c, str):
            lc = c.lower()
            if lc == "open": ren[c] = "Open"
            elif lc == "high": ren[c] = "High"
            elif lc == "low": ren[c] = "Low"
            elif lc == "close": ren[c] = "Close"
            elif lc == "volume": ren[c] = "Volume"
    if ren:
        df = df.rename(columns=ren)

    if "Volume" not in df.columns:
        df["Volume"] = 0

    if not set(["Open","High","Low","Close"]).issubset(df.columns):
        return pd.DataFrame()

    df = df[["Open","High","Low","Close","Volume"]].dropna(subset=["Open","High","Low","Close"])
    df = df.sort_index()
    return df


@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_hist(ticker: str, period: str = "5y") -> pd.DataFrame:
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
    return normalize_ohlcv(raw, ticker)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not set(REQUIRED_COLS).issubset(df.columns):
        return pd.DataFrame()

    ohlc = df[["Open","High","Low","Close"]].resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    })
    vol = df[["Volume"]].resample(rule).sum()
    out = pd.concat([ohlc, vol], axis=1).dropna(subset=["Open","High","Low","Close"])
    return out


def tf_frames(daily: pd.DataFrame):
    w = resample_ohlc(daily, "W-FRI")
    m = resample_ohlc(daily, "M")
    return daily, w, m


# =========================
# STRAT LOGIC
# =========================
def is_bull_bar(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 1:
        return False
    c = df.iloc[-1]
    return bool(c["Close"] > c["Open"])

def is_bear_bar(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 1:
        return False
    c = df.iloc[-1]
    return bool(c["Close"] < c["Open"])

def is_inside_bar(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 2:
        return False
    cur = df.iloc[-1]
    prev = df.iloc[-2]
    return bool((cur["High"] <= prev["High"]) and (cur["Low"] >= prev["Low"]))

def is_outside_bar(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 2:
        return False
    cur = df.iloc[-1]
    prev = df.iloc[-2]
    return bool((cur["High"] >= prev["High"]) and (cur["Low"] <= prev["Low"]))

def is_2u(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 2:
        return False
    cur = df.iloc[-1]
    prev = df.iloc[-2]
    return bool(cur["High"] > prev["High"])

def is_2d(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 2:
        return False
    cur = df.iloc[-1]
    prev = df.iloc[-2]
    return bool(cur["Low"] < prev["Low"])

def strat_green_2u(df: pd.DataFrame) -> bool:
    return bool(is_2u(df) and is_bull_bar(df))

def strat_red_2d(df: pd.DataFrame) -> bool:
    return bool(is_2d(df) and is_bear_bar(df))

def forming_212(df: pd.DataFrame) -> bool:
    # Simple + useful approximation:
    # Prior bar = outside (2), current bar = inside (1)
    if df is None or len(df) < 3:
        return False
    prev = df.iloc[:-1]
    return bool(is_outside_bar(prev) and is_inside_bar(df))

def atr(df: pd.DataFrame, length: int = 14) -> float:
    if df is None or len(df) < length + 2:
        return float("nan")
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    a = tr.rolling(length).mean().iloc[-1]
    return float(a)

def atr_pct(df: pd.DataFrame, length: int = 14) -> float:
    a = atr(df, length)
    if not np.isfinite(a):
        return float("nan")
    px = float(df["Close"].iloc[-1])
    if px <= 0:
        return float("nan")
    return 100.0 * (a / px)

def room_to_run(df: pd.DataFrame, direction: str, lookback: int = 63) -> float:
    if df is None or len(df) < lookback + 2:
        return float("nan")
    cur = float(df["Close"].iloc[-1])
    if direction == "LONG":
        recent_high = float(df["High"].iloc[-lookback:].max())
        return max(0.0, recent_high - cur)
    else:
        recent_low = float(df["Low"].iloc[-lookback:].min())
        return max(0.0, cur - recent_low)

def compression_score(df: pd.DataFrame) -> float:
    # tighter current bar range vs ATR => higher score
    a = atr(df, 14)
    if not np.isfinite(a) or a <= 0:
        return 0.0
    rng = float(df["High"].iloc[-1] - df["Low"].iloc[-1])
    ratio = rng / a
    # ratio small => good; clamp into 0..1
    return float(max(0.0, min(1.0, 1.0 - ratio)))

def trigger_levels(daily: pd.DataFrame, weekly: pd.DataFrame, direction: str):
    """
    Prefers weekly inside-bar triggers when present; otherwise daily inside-bar.
    Returns TF, entry, stop, room, rr, atr_pct
    """
    a_pct = atr_pct(daily, 14)
    a = atr(daily, 14)

    tf = None
    entry = None
    stop = None
    room = None
    rr = None

    if weekly is not None and len(weekly) >= 2 and is_inside_bar(weekly):
        tf = "W"
        cur = weekly.iloc[-1]
        if direction == "LONG":
            entry = float(cur["High"])
            stop = float(cur["Low"])
        else:
            entry = float(cur["Low"])
            stop = float(cur["High"])
    elif daily is not None and len(daily) >= 2 and is_inside_bar(daily):
        tf = "D"
        cur = daily.iloc[-1]
        if direction == "LONG":
            entry = float(cur["High"])
            stop = float(cur["Low"])
        else:
            entry = float(cur["Low"])
            stop = float(cur["High"])

    if tf is not None and entry is not None and stop is not None:
        risk = abs(entry - stop)
        room = room_to_run(daily, direction, 63)
        if risk > 0 and np.isfinite(room):
            rr = room / risk

    return tf, entry, stop, room, rr, a_pct


def setup_flags(d: pd.DataFrame, w: pd.DataFrame, m: pd.DataFrame):
    # Bull/Bear by candle color on each TF (simple regime)
    return {
        "D_Bull": is_bull_bar(d),
        "W_Bull": is_bull_bar(w),
        "M_Bull": is_bull_bar(m),
        "D_Bear": is_bear_bar(d),
        "W_Bear": is_bear_bar(w),
        "M_Bear": is_bear_bar(m),
        "D_Inside": is_inside_bar(d),
        "W_Inside": is_inside_bar(w),
        "M_Inside": is_inside_bar(m),
        "D_212Up": forming_212(d) and (not is_bear_bar(d)),
        "W_212Up": forming_212(w) and (not is_bear_bar(w)),
        "D_212Dn": forming_212(d) and (not is_bull_bar(d)),
        "W_212Dn": forming_212(w) and (not is_bull_bar(w)),
    }


def market_bias(market_rows: pd.DataFrame):
    """
    Returns bias direction and strength 0..100, plus bull-bear diff.
    """
    if market_rows is None or market_rows.empty:
        return "NEUTRAL", 0, 0

    bull = int(market_rows[["D_Bull","W_Bull","M_Bull"]].sum().sum())
    bear = int(market_rows[["D_Bear","W_Bear","M_Bear"]].sum().sum())
    diff = bull - bear

    # strength: normalize using total checks
    total_checks = 3 * len(market_rows)
    raw = (diff + total_checks) / (2 * total_checks)  # 0..1
    strength = int(round(100 * raw))

    if diff >= 2:
        return "LONG", strength, diff
    if diff <= -2:
        return "SHORT", strength, diff
    return "NEUTRAL", strength, diff


def score_ticker(flags: dict, direction: str, require_align: bool, only_212: bool, only_inside: bool):
    """
    SetupScore emphasizes:
    - Weekly trigger quality
    - Monthly/Weekly alignment
    - Inside-bar availability
    """
    # Filters
    if only_inside and not (flags["D_Inside"] or flags["W_Inside"]):
        return 0, False
    if only_212:
        if direction == "LONG" and not (flags["D_212Up"] or flags["W_212Up"]):
            return 0, False
        if direction == "SHORT" and not (flags["D_212Dn"] or flags["W_212Dn"]):
            return 0, False

    # Alignment filter (bias direction): require at least one of W or M aligns
    if require_align:
        if direction == "LONG" and not (flags["W_Bull"] or flags["M_Bull"]):
            return 0, False
        if direction == "SHORT" and not (flags["W_Bear"] or flags["M_Bear"]):
            return 0, False

    score = 0

    # Bias-direction regime points
    if direction == "LONG":
        score += 20 if flags["M_Bull"] else 0
        score += 20 if flags["W_Bull"] else 0
        score += 10 if flags["D_Bull"] else 0
        score += 15 if flags["W_212Up"] else 0
        score += 10 if flags["D_212Up"] else 0
    elif direction == "SHORT":
        score += 20 if flags["M_Bear"] else 0
        score += 20 if flags["W_Bear"] else 0
        score += 10 if flags["D_Bear"] else 0
        score += 15 if flags["W_212Dn"] else 0
        score += 10 if flags["D_212Dn"] else 0

    # Trigger availability (inside bars)
    score += 15 if flags["W_Inside"] else 0
    score += 10 if flags["D_Inside"] else 0

    return score, True


def magnitude_score(daily: pd.DataFrame, direction: str):
    """
    Magnitude score uses:
    - ATR% (prefer higher, up to a point)
    - Room-to-run in ATRs (prefer higher)
    - Compression (prefer tighter)
    """
    a_pct = atr_pct(daily, 14)
    a = atr(daily, 14)

    if not np.isfinite(a_pct) or not np.isfinite(a) or a <= 0:
        return 0

    # ATR% scoring (0..30)
    # sweet spot ~ 1% to 4% for swing; clamp
    atr_component = max(0.0, min(1.0, (a_pct - 1.0) / 3.0))
    atr_points = 30.0 * atr_component

    # Room in ATRs (0..40)
    r = room_to_run(daily, direction, 63)
    room_atr = (r / a) if np.isfinite(r) else 0.0
    room_component = max(0.0, min(1.0, room_atr / 6.0))  # 6 ATRs ~ strong
    room_points = 40.0 * room_component

    # Compression (0..30)
    comp = compression_score(daily)
    comp_points = 30.0 * comp

    return int(round(atr_points + room_points + comp_points))


# =========================
# UI
# =========================
st.title(APP_TITLE)
st.caption(APP_SUB)

with st.expander("Filters", expanded=True):
    c1, c2, c3, c4 = st.columns([1.2, 1.6, 2.2, 2.0])
    with c1:
        only_inside = st.checkbox("ONLY Inside Bars (D or W)", value=False)
    with c2:
        only_212 = st.checkbox("ONLY 2-1-2 forming (bias direction)", value=False)
    with c3:
        require_align = st.checkbox("Require Monthly OR Weekly alignment (bias direction)", value=True)
    with c4:
        top_n = st.slider("Top Picks count", min_value=3, max_value=8, value=5)

st.divider()

# =========================
# BUILD MARKET REGIME
# =========================
st.subheader("Market Regime (SPY / QQQ / IWM / DIA) — Bull vs Bear")

market_rows = []
for name, etf in MARKET_ETFS.items():
    d = get_hist(etf)
    d, w, m = tf_frames(d)
    if d.empty or w.empty or m.empty:
        continue
    flags = setup_flags(d, w, m)
    row = {"Market": name, "ETF": etf, **flags}
    market_rows.append(row)

market_df = pd.DataFrame(market_rows)
if market_df.empty:
    st.error("No market data loaded. Try again in a minute (rate-limit), or refresh.")
    st.stop()

# Show compact columns for market
market_show_cols = ["Market","ETF","D_Bull","W_Bull","M_Bull","D_Bear","W_Bear","M_Bear","D_212Up","W_212Up","D_212Dn","W_212Dn"]
st.dataframe(market_df[market_show_cols], use_container_width=True, hide_index=True)

bias, strength, diff = market_bias(market_df)

st.divider()

# =========================
# SECTOR RANKING
# =========================
st.subheader("Sectors (SPDR) — ranked after bias is known")

sector_rows = []
for sector, etf in SECTOR_ETFS.items():
    d = get_hist(etf)
    d, w, m = tf_frames(d)
    if d.empty or w.empty or m.empty:
        continue
    flags = setup_flags(d, w, m)

    # BullScore / BearScore depending on bias
    bullscore = int(flags["M_Bull"]) + int(flags["W_Bull"]) + int(flags["D_Bull"])
    bearscore = int(flags["M_Bear"]) + int(flags["W_Bear"]) + int(flags["D_Bear"])

    row = {
        "Sector": sector,
        "ETF": etf,
        "BullScore": bullscore,
        "BearScore": bearscore,
        **flags
    }
    sector_rows.append(row)

sector_df = pd.DataFrame(sector_rows)

# Rank based on bias
if bias == "LONG":
    sector_df = sector_df.sort_values(["BullScore","W_Inside","D_Inside"], ascending=[False, False, False])
elif bias == "SHORT":
    sector_df = sector_df.sort_values(["BearScore","W_Inside","D_Inside"], ascending=[False, False, False])
else:
    sector_df = sector_df.sort_values(["BullScore","BearScore"], ascending=[False, False])

sector_show_cols = [
    "Sector","ETF","BullScore","BearScore",
    "D_Bull","W_Bull","M_Bull","D_Bear","W_Bear","M_Bear",
    "D_Inside","W_Inside","M_Inside","D_212Up","W_212Up","D_212Dn","W_212Dn"
]
st.dataframe(sector_df[sector_show_cols], use_container_width=True, hide_index=True)

st.divider()

# =========================
# DRILL INTO SECTOR
# =========================
st.subheader("Drill into a sector (ranks candidates in bias direction + magnitude)")

sector_names = list(SECTOR_ETFS.keys())
default_sector = sector_names[0]
selected_sector = st.selectbox("Choose a sector:", sector_names, index=sector_names.index(default_sector))
selected_etf = SECTOR_ETFS[selected_sector]

st.caption(f"Bias: **{bias}** | Strength: **{strength}/100** | Bull-Bear diff: **{diff}**")
tickers = SECTOR_TICKERS.get(selected_etf, [])
st.write(f"Selected: **{selected_sector} ({selected_etf})** — tickers in list: **{len(tickers)}**")

scan_count = st.slider("How many tickers to scan", min_value=5, max_value=max(5, len(tickers)), value=min(15, len(tickers)))
scan_list = tickers[:scan_count]

# Build ticker table
rows = []
for t in scan_list:
    d = get_hist(t)
    if d.empty:
        continue
    d, w, m = tf_frames(d)
    if d.empty or w.empty or m.empty:
        continue

    flags = setup_flags(d, w, m)

    # If bias neutral, still score both and take best direction for display
    direction = bias
    if direction == "NEUTRAL":
        # pick direction with higher simple regime agreement
        long_agree = int(flags["W_Bull"]) + int(flags["M_Bull"])
        short_agree = int(flags["W_Bear"]) + int(flags["M_Bear"])
        direction = "LONG" if long_agree >= short_agree else "SHORT"

    setup_score, ok = score_ticker(flags, direction, require_align, only_212, only_inside)
    if not ok or setup_score <= 0:
        continue

    mag_score = magnitude_score(d, direction)
    total_score = setup_score + mag_score

    tf, entry, stop, room, rr, a_pct = trigger_levels(d, w, direction)

    rows.append({
        "Ticker": t,
        "TotalScore": total_score,
        "SetupScore": setup_score,
        "MagScore": mag_score,
        "TF": tf,
        "Entry": None if entry is None else round(entry, 2),
        "Stop": None if stop is None else round(stop, 2),
        "Room": None if room is None else round(room, 2),
        "RR": None if rr is None or not np.isfinite(rr) else round(rr, 2),
        "ATR%": None if a_pct is None or not np.isfinite(a_pct) else round(a_pct, 2),
        **flags
    })

cand_df = pd.DataFrame(rows)

if cand_df.empty:
    st.info("No matches found with current filters. Loosen filters (turn off Inside-only / 2-1-2-only / alignment) or try another sector.")
else:
    cand_df = cand_df.sort_values(["TotalScore","SetupScore","MagScore"], ascending=[False, False, False])

    st.subheader(f"Top Trade Ideas (best {top_n}) — Bias: {bias if bias!='NEUTRAL' else 'AUTO'}")
    top_df = cand_df.head(top_n).copy()
    show_cols = [
        "Ticker","TotalScore","SetupScore","MagScore","TF","Entry","Stop","Room","RR","ATR%",
        "W_212Up","D_212Up","W_212Dn","D_212Dn",
        "M_Bull","W_Bull","D_Bull","M_Bear","W_Bear","D_Bear",
        "W_Inside","D_Inside"
    ]
    # Keep only cols that exist
    show_cols = [c for c in show_cols if c in top_df.columns]
    st.dataframe(top_df[show_cols], use_container_width=True, hide_index=True)

    # Trade of the day: needs a valid trigger (Entry/Stop)
    trade_df = cand_df.dropna(subset=["Entry","Stop"]).copy()
    st.subheader("🎯 Trade of the Day (best TotalScore + valid trigger)")
    if trade_df.empty:
        st.info("No valid trigger found (needs Inside Bar levels). Use Top Ideas and wait for an Inside Bar trigger.")
    else:
        best = trade_df.iloc[0]
        st.success(
            f"{best['Ticker']} | TF: {best['TF']} | Entry: {best['Entry']} | Stop: {best['Stop']} | "
            f"RR: {best.get('RR','-')} | ATR%: {best.get('ATR%','-')} | TotalScore: {best['TotalScore']}"
        )

    with st.expander("All Matches (ranked by TotalScore)", expanded=True):
        st.dataframe(cand_df[show_cols], use_container_width=True, hide_index=True)

st.divider()

# =========================
# QUICK MARKET READ (Sentiment + Rotation)
# =========================
st.subheader("Quick Market Read")

# Strong sectors
if bias == "LONG":
    strong = sector_df.head(3)[["Sector","ETF"]]
    strong_txt = ", ".join([f"{r.Sector}({r.ETF})" for r in strong.itertuples(index=False)])
    in_sectors = sector_df[sector_df["BullScore"] >= 2][["Sector","ETF"]]
    out_sectors = sector_df[sector_df["BearScore"] >= 2][["Sector","ETF"]]
elif bias == "SHORT":
    strong = sector_df.head(3)[["Sector","ETF"]]
    strong_txt = ", ".join([f"{r.Sector}({r.ETF})" for r in strong.itertuples(index=False)])
    in_sectors = sector_df[sector_df["BearScore"] >= 2][["Sector","ETF"]]
    out_sectors = sector_df[sector_df["BullScore"] >= 2][["Sector","ETF"]]
else:
    strong = sector_df.head(3)[["Sector","ETF"]]
    strong_txt = ", ".join([f"{r.Sector}({r.ETF})" for r in strong.itertuples(index=False)])
    in_sectors = sector_df[sector_df["BullScore"] > sector_df["BearScore"]][["Sector","ETF"]]
    out_sectors = sector_df[sector_df["BearScore"] > sector_df["BullScore"]][["Sector","ETF"]]

in_txt = ", ".join([f"{r.Sector}({r.ETF})" for r in in_sectors.itertuples(index=False)]) if len(in_sectors) else "None"
out_txt = ", ".join([f"{r.Sector}({r.ETF})" for r in out_sectors.itertuples(index=False)]) if len(out_sectors) else "None"

# Plan text
if bias == "LONG":
    plan = "Plan: LONG only. Prioritize Weekly 2-1-2 UP + Inside Bar breaks in leading sectors. Prefer higher RR + ATR%."
    pill = "✅"
elif bias == "SHORT":
    plan = "Plan: SHORT only. Prioritize Weekly 2-1-2 DOWN + Inside Bar breaks in weak sectors. Prefer higher RR + ATR%."
    pill = "🟥"
else:
    plan = "Plan: Mixed/neutral. Reduce size. Wait for clear regime (W/M alignment) before leaning hard."
    pill = "⚪"

st.write(f"**Bias:** {bias} {pill} | **Strength:** {strength}/100")
st.write(f"**Strong Sectors:** {strong_txt}")
st.write(f"**Rotation IN:** {in_txt} | **OUT:** {out_txt}")

# Short + punchy message box
if bias == "LONG" and strength >= 55:
    st.success(plan)
elif bias == "SHORT" and strength >= 55:
    st.error(plan)
else:
    st.info(plan)

st.caption(
    "Trigger Levels: If Inside Bar exists, LONG = buy break of High, stop below Low. "
    "SHORT = sell break of Low, stop above High. Weekly triggers preferred when available. "
    "Magnitude = Room-to-run vs risk (RR) + ATR% (movement) + compression (tight bar range)."
)
