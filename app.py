# app.py — STRAT Regime Scanner V1.3.0 (Merged + Trade Plan Notes)
# Includes:
# - Scanner (STRAT regime + triggers + ranking)
# - Market Dashboard (sentiment + sector rotation + strength leaders)
# - Today Watchlist Builder (#5)
# - Ticker Analyzer (search + explain scoring + STRAT trigger context)
# - Trade Plan Notes (A/B/C grade, targets, invalidation, improve notes)

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="STRAT Regime Scanner V1.3.0", layout="wide")

# =========================
# UNIVERSE
# =========================
MARKET_ETFS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Russell 2000": "IWM",
    "Dow Jones": "DIA",
}

METALS_ETFS = {
    "Metals - Gold": "GLD",
    "Metals - Silver": "SLV",
    "Metals - Copper": "CPER",
    "Metals - Platinum": "PPLT",
    "Metals - Palladium": "PALL",
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
    **METALS_ETFS,
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

    "Metals - Gold": ["GLD"],
    "Metals - Silver": ["SLV"],
    "Metals - Copper": ["CPER"],
    "Metals - Platinum": ["PPLT"],
    "Metals - Palladium": ["PALL"],
}

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

# =========================
# DATA FETCH (CACHED) — HARDENED
# =========================
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df

def _flatten_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_datetime_index(df)
    if df.empty:
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

    rename_map = {}
    for c in df.columns:
        if not isinstance(c, str):
            continue
        lc = c.lower()
        if lc == "open": rename_map[c] = "Open"
        elif lc == "high": rename_map[c] = "High"
        elif lc == "low": rename_map[c] = "Low"
        elif lc in ("close", "adj close", "adj_close", "adjclose"):
            rename_map[c] = "Close" if "Close" not in df.columns else c
        elif lc == "volume": rename_map[c] = "Volume"

    if rename_map:
        df = df.rename(columns=rename_map)

    if "Close" not in df.columns:
        for alt in ["Adj Close", "adj close", "Adj_Close", "AdjClose"]:
            if alt in df.columns:
                df["Close"] = df[alt]
                break

    if "Volume" not in df.columns:
        df["Volume"] = 0

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(set(df.columns)):
        return pd.DataFrame()

    df = df[needed].copy()
    df = _dedupe_columns(df)

    for c in needed:
        if c in df.columns and isinstance(df[c], pd.DataFrame):
            df[c] = df[c].iloc[:, 0]
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
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

    df = _ensure_datetime_index(df)
    df = _dedupe_columns(df)
    if df.empty:
        return pd.DataFrame()

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            if isinstance(df[c], pd.DataFrame):
                df[c] = df[c].iloc[:, 0]
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return pd.DataFrame()

    def safe_first(x):
        x = x.dropna()
        return x.iloc[0] if len(x) else np.nan

    def safe_last(x):
        x = x.dropna()
        return x.iloc[-1] if len(x) else np.nan

    g = df.resample(rule)
    out = pd.DataFrame({
        "Open": g["Open"].apply(safe_first),
        "High": g["High"].max(),
        "Low": g["Low"].min(),
        "Close": g["Close"].apply(safe_last),
        "Volume": g["Volume"].sum(),
    }).dropna(subset=["Open", "High", "Low", "Close"])

    return out

# =========================
# DASHBOARD HELPERS (CAPPED)
# =========================
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def total_return(series: pd.Series, lookback: int) -> pd.Series:
    return series / series.shift(lookback) - 1

def rs_vs_spy(series: pd.Series, spy_series: pd.Series, lookback: int) -> pd.Series:
    return total_return(series, lookback) - total_return(spy_series, lookback)

def clamp_rs(x, lo, hi):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0

RS_CAP = 0.10      # ±10% cap for RS vs SPY
ROT_CAP = 0.08     # ±8% cap for Rotation

def strength_meter(rs_short_v: float, rotation_v: float, trend: str) -> int:
    rs_short_v = clamp_rs(rs_short_v, -RS_CAP, RS_CAP)
    rotation_v = clamp_rs(rotation_v, -ROT_CAP, ROT_CAP)

    rs_score = np.clip(50 + (rs_short_v * 100.0) * 6.0, 0, 100)
    rot_score = np.clip(50 + (rotation_v * 100.0) * 8.0, 0, 100)
    trend_bonus = 10 if trend == "UP" else -10

    score = 0.50 * rs_score + 0.35 * rot_score + 0.15 * 50 + trend_bonus
    return int(np.clip(score, 0, 100))

def strength_label(score: int) -> str:
    if score >= 70:
        return "STRONG"
    if score >= 45:
        return "NEUTRAL"
    return "WEAK"

def meter_style(val: str) -> str:
    if val == "STRONG":
        return "background-color: #114b2b; color: white;"
    if val == "NEUTRAL":
        return "background-color: #5a4b11; color: white;"
    return "background-color: #5a1111; color: white;"

def strength_style(v):
    try:
        x = float(v)
    except Exception:
        return ""
    x = max(0.0, min(100.0, x))
    if x < 50:
        t = x / 50.0
        r, g, b = 90, int(17 + (75 - 17) * t), 17
    else:
        t = (x - 50.0) / 50.0
        r, g, b = int(90 + (17 - 90) * t), 75, int(17 + (43 - 17) * t)
    return f"background-color: rgb({r},{g},{b}); color: white; font-weight: 600;"

def trend_label(series: pd.Series, ema_len: int) -> str:
    e = ema(series, ema_len)
    up = bool(series.iloc[-1] > e.iloc[-1] and e.iloc[-1] > e.iloc[-2])
    return "UP" if up else "DOWN/CHOP"

def pullback_zone_ok(trend: str, rsi_val: float, pb_low: float, pb_high: float) -> bool:
    if trend != "UP":
        return False
    return (pb_low <= rsi_val <= pb_high)

# =========================
# STRAT HELPERS (Scanner + Analyzer)
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
        return ("W", hi, lo) if bias == "LONG" else ("W", lo, hi)

    if strat_inside(d) and len(d) >= 2:
        cur = d.iloc[-1]
        hi, lo = float(cur["High"]), float(cur["Low"])
        return ("D", hi, lo) if bias == "LONG" else ("D", lo, hi)

    return None, None, None

# =========================
# TRADE PLAN NOTES (NEW)
# =========================
def pick_grade(strength: int, trend: str, rot: float, has_trigger: bool, rsi_val: float, pb_low: float, pb_high: float) -> str:
    rot_c = clamp_rs(rot, -ROT_CAP, ROT_CAP)
    pb_ok = pullback_zone_ok(trend, rsi_val, pb_low, pb_high)

    score = 0
    score += 2 if strength >= 75 else 1 if strength >= 65 else 0
    score += 1 if trend == "UP" else 0
    score += 1 if rot_c > 0 else 0
    score += 1 if has_trigger else 0
    score += 1 if pb_ok else 0

    if score >= 5:
        return "A"
    if score >= 3:
        return "B"
    return "C"

def atrp_bucket(atrp: Optional[float]) -> str:
    if atrp is None:
        return "unknown"
    if atrp < 1.2:
        return "small"
    if atrp < 2.5:
        return "normal"
    return "aggressive"

def targets_from_range(d: pd.DataFrame, entry: Optional[float], direction: str) -> Tuple[Optional[float], Optional[float]]:
    # Simple swing targets:
    # T1 = 20-day extreme, T2 = 63-day extreme
    if d is None or d.empty or len(d) < 70:
        return None, None

    hi20 = float(d["High"].rolling(20).max().iloc[-1])
    lo20 = float(d["Low"].rolling(20).min().iloc[-1])
    hi63 = float(d["High"].rolling(63).max().iloc[-1])
    lo63 = float(d["Low"].rolling(63).min().iloc[-1])

    if direction == "LONG":
        return hi20, hi63
    else:
        return lo20, lo63

def trade_plan_notes(
    ticker: str,
    trend: str,
    rsi_val: float,
    strength: int,
    rotation: float,
    trigger_status: str,
    tf: Optional[str],
    entry: Optional[float],
    stop: Optional[float],
    d_df: pd.DataFrame,
    pb_low: float,
    pb_high: float,
) -> Dict[str, str]:
    has_trigger = (entry is not None and stop is not None) and ("READY" in trigger_status)

    grade = pick_grade(strength, trend, rotation, has_trigger, rsi_val, pb_low, pb_high)

    # Play decision
    if trend != "UP":
        play = "AVOID (trend not UP) or treat as short-only if you add short logic later."
    else:
        if has_trigger:
            play = "BREAKOUT PLAY: place a stop order at entry + use stop as defined."
        else:
            pb_ok = pullback_zone_ok(trend, rsi_val, pb_low, pb_high)
            play = "WAIT: no inside bar trigger yet." if not pb_ok else "PULLBACK PLAY: strong name in pullback zone—wait for inside bar."

    # Risk unit
    risk_unit = None
    if entry is not None and stop is not None:
        risk_unit = abs(entry - stop)

    # ATR% sizing hint
    atr = atr14(d_df)
    atrp = None
    if math.isfinite(atr) and atr > 0:
        close = float(d_df["Close"].iloc[-1])
        if close > 0:
            atrp = (atr / close) * 100.0

    size_hint = atrp_bucket(atrp)
    if size_hint == "small":
        sizing_note = "ATR% is small → position can be a bit larger, but expect slower movement."
    elif size_hint == "normal":
        sizing_note = "ATR% is normal → standard sizing."
    elif size_hint == "aggressive":
        sizing_note = "ATR% is high → size down, wider swings."
    else:
        sizing_note = "ATR% unavailable."

    # Targets / invalidation
    t1, t2 = targets_from_range(d_df, entry, "LONG")
    invalidation = "Invalidation = break below STOP (or close below if you trade close-based)."
    improve = []
    if trend != "UP":
        improve.append("Improve: reclaim and hold above your trend EMA + RS turns positive.")
    if clamp_rs(rotation, -ROT_CAP, ROT_CAP) <= 0:
        improve.append("Improve: rotation flips positive (short-term RS > long-term RS).")
    if not pullback_zone_ok(trend, rsi_val, pb_low, pb_high):
        improve.append(f"Improve: RSI pulls into {pb_low}-{pb_high} zone without breaking trend.")
    if not has_trigger:
        improve.append("Improve: print a Daily or Weekly Inside Bar for clean entry/stop.")

    rr_hint = ""
    if risk_unit is not None and t2 is not None and entry is not None:
        reward = max(0.0, t2 - entry)
        rr = reward / risk_unit if risk_unit > 0 else None
        if rr is not None and math.isfinite(rr):
            rr_hint = f"Approx RR to T2: ~{rr:.2f} (rough, uses 63d high as T2)."

    return {
        "Grade": grade,
        "Play": play,
        "RiskUnit": f"{risk_unit:.2f}" if risk_unit is not None else "n/a",
        "Targets": f"T1: {t1:.2f} | T2: {t2:.2f}" if (t1 is not None and t2 is not None) else "Targets: n/a",
        "Invalidation": invalidation,
        "Improve": " | ".join(improve) if improve else "Improve: keep it simple—wait for trigger.",
        "Sizing": sizing_note,
        "RRHint": rr_hint or "RR: n/a",
    }

# =========================
# EXPLAINABLE ANALYZER
# =========================
def analyze_ticker(
    ticker: str,
    spy_close: pd.Series,
    rs_short: int,
    rs_long: int,
    ema_trend_len: int,
    rsi_len: int,
) -> Optional[Dict]:
    d = get_hist(ticker)
    if d.empty:
        return None

    close = d["Close"].dropna()
    if close.empty or len(close) < max(rs_long, 80) + 10:
        return None

    tr = trend_label(close, ema_trend_len)
    rsi_v = float(rsi_wilder(close, rsi_len).iloc[-1])

    rs_s = float(rs_vs_spy(close, spy_close, rs_short).iloc[-1])
    rs_l = float(rs_vs_spy(close, spy_close, rs_long).iloc[-1])

    rs_s_c = clamp_rs(rs_s, -RS_CAP, RS_CAP)
    rs_l_c = clamp_rs(rs_l, -RS_CAP, RS_CAP)

    rot = rs_s - rs_l
    rot_c = clamp_rs(rot, -ROT_CAP, ROT_CAP)

    strength = strength_meter(rs_s_c, rot_c, tr)
    meter = strength_label(strength)

    d_tf, w_tf, m_tf = tf_frames(d)
    flags = compute_flags(d_tf, w_tf, m_tf)

    tf, entry, stop = best_trigger("LONG", d_tf, w_tf)
    trigger_status = "READY" if (flags["W_Inside"] or flags["D_Inside"]) else "WAIT (No Inside Bar)"
    entry_r = None if entry is None else round(float(entry), 2)
    stop_r  = None if stop  is None else round(float(stop), 2)

    explain = [
        f"Trend = {tr} (price vs {ema_trend_len} EMA)",
        f"RSI({rsi_len}) = {rsi_v:.1f}",
        f"RS vs SPY short ({rs_short}) = {rs_s*100:.1f}% (capped to {rs_s_c*100:.1f}%)",
        f"RS vs SPY long ({rs_long}) = {rs_l*100:.1f}% (capped to {rs_l_c*100:.1f}%)",
        f"Rotation = (RS short - RS long) = {rot*100:.1f}% (capped to {rot_c*100:.1f}%)",
        f"Strength Score = {strength}/100 ({meter})",
    ]

    strat_note = []
    if flags["M_Bull"]: strat_note.append("Monthly: Bull")
    if flags["W_Bull"]: strat_note.append("Weekly: Bull")
    if flags["D_Bull"]: strat_note.append("Daily: Bull")
    if flags["W_Inside"]: strat_note.append("Weekly: Inside Bar")
    if flags["D_Inside"]: strat_note.append("Daily: Inside Bar")
    if flags["W_212Up"]: strat_note.append("Weekly: 2-1-2 Up")
    if flags["D_212Up"]: strat_note.append("Daily: 2-1-2 Up")
    if not strat_note:
        strat_note = ["No STRAT alignment flags currently"]

    return {
        "Ticker": ticker.upper(),
        "Trend": tr,
        "RSI": rsi_v,
        "RS_short": rs_s,
        "RS_long": rs_l,
        "Rotation": rot,
        "Strength": strength,
        "Meter": meter,
        "TriggerStatus": trigger_status,
        "TF": tf,
        "Entry": entry_r,
        "Stop": stop_r,
        "Flags": flags,
        "Explain": explain,
        "STRAT_Notes": strat_note,
        "DailyDF": d,   # keep for trade plan / targets / ATR%
    }

def writeup_block(info: Dict, pb_low: float, pb_high: float) -> None:
    t = info["Ticker"]
    st.markdown(f"#### {t} — {info['Meter']} ({info['Strength']}/100)")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.write(f"**Trend:** {info['Trend']}")
    with c2: st.write(f"**RSI:** {info['RSI']:.1f}")
    with c3: st.write(f"**RS short:** {info['RS_short']*100:.1f}%")
    with c4: st.write(f"**Rotation:** {info['Rotation']*100:.1f}%")

    pb_ok = pullback_zone_ok(info["Trend"], info["RSI"], pb_low, pb_high)
    st.write(f"**Pullback Zone ({pb_low}-{pb_high}) OK?** {'✅ YES' if pb_ok else '❌ NO'}")

    st.write(
        f"**Trigger:** {info['TriggerStatus']}"
        + (f" | TF: **{info['TF']}** | Entry: **{info['Entry']}** | Stop: **{info['Stop']}**" if info["Entry"] else "")
    )

    plan = trade_plan_notes(
        ticker=t,
        trend=info["Trend"],
        rsi_val=float(info["RSI"]),
        strength=int(info["Strength"]),
        rotation=float(info["Rotation"]),
        trigger_status=info["TriggerStatus"],
        tf=info["TF"],
        entry=info["Entry"],
        stop=info["Stop"],
        d_df=info["DailyDF"],
        pb_low=pb_low,
        pb_high=pb_high,
    )

    st.markdown("### 🧠 Trade Plan Notes")
    p1, p2, p3 = st.columns([1, 2, 2])
    with p1:
        st.metric("Grade", plan["Grade"])
    with p2:
        st.write(f"**Play:** {plan['Play']}")
    with p3:
        st.write(f"**Risk Unit:** {plan['RiskUnit']} | **{plan['RRHint']}**")

    st.write(f"**Targets:** {plan['Targets']}")
    st.write(f"**Invalidation:** {plan['Invalidation']}")
    st.write(f"**Sizing Hint:** {plan['Sizing']}")
    st.write(f"**What makes it better:** {plan['Improve']}")

    with st.expander("Why this scores the way it does"):
        for line in info["Explain"]:
            st.write(f"- {line}")
        st.write("**STRAT context:**")
        for s in info["STRAT_Notes"]:
            st.write(f"- {s}")

# =========================
# PAGES
# =========================
def show_user_guide():
    st.title("📘 STRAT Regime Scanner — User Guide (V1.3.0)")
    st.markdown("""
## What this app does
- **Scanner**: STRAT regime + inside bar triggers + ranked trade ideas
- **Market Dashboard**: overall sentiment + sector rotation + strength leaders
- **Today Watchlist**: auto-builds a list of names based on rotation/strength + filters
- **Ticker Analyzer**: type any ticker and see exactly why it is scoring the way it is
- **Trade Plan Notes**: A/B/C grade + targets + invalidation + what makes it better

---

## How to use (2–5 min)
1) Market Dashboard → **Top Groups IN**
2) Today Watchlist → **pick 3–10 tickers**
3) Open write-ups → take only **A/B** ideas
4) Enter only on **Inside Bar trigger** (or wait)
""")

def show_market_dashboard():
    st.title("📊 Market Dashboard (Sentiment • Rotation • Leaders • Watchlist) — V1.3.0")
    st.caption("Strength meter is capped for stability. Write-ups include Trade Plan Notes.")

    with st.expander("Dashboard Settings", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.2])
        with c1:
            rs_short = st.selectbox("RS Lookback (short)", [21, 30, 42], index=0)
        with c2:
            rs_long = st.selectbox("RS Lookback (long)", [63, 90, 126], index=0)
        with c3:
            ema_trend_len = st.selectbox("Trend EMA", [50, 100, 200], index=0)
        with c4:
            rsi_len = st.selectbox("RSI Length", [7, 14, 21], index=1)
        with c5:
            if st.button("Refresh data"):
                st.cache_data.clear()
                st.rerun()

    with st.expander("Today Watchlist Settings", expanded=True):
        w1, w2, w3, w4, w5 = st.columns([1, 1, 1, 1, 1.2])
        with w1:
            top_sectors_in = st.slider("Top Sectors IN", 1, 6, 3)
        with w2:
            leaders_per_sector = st.slider("Leaders per sector", 3, 10, 5)
        with w3:
            pb_low = st.slider("RSI Pullback Low (UP trend)", 25, 60, 40)
        with w4:
            pb_high = st.slider("RSI Pullback High (UP trend)", 35, 75, 55)
        with w5:
            strict_pullback = st.checkbox("Strict pullback filter (only show RSI-in-zone)", value=False)

    st.subheader("Overall Market Sentiment")
    market_syms = list(MARKET_ETFS.values()) + ["^VIX"]
    mcols = st.columns(len(market_syms))

    for i, sym in enumerate(market_syms):
        d = get_hist(sym)
        if d.empty:
            with mcols[i]:
                st.metric(sym, "n/a", "n/a")
            continue

        close = d["Close"].dropna()
        if close.empty or len(close) < 10:
            with mcols[i]:
                st.metric(sym, "n/a", "n/a")
            continue

        tr = trend_label(close, int(ema_trend_len))
        r = float(rsi_wilder(close, int(rsi_len)).iloc[-1])
        ret = float(total_return(close, int(rs_short)).iloc[-1]) if len(close) > rs_short else np.nan

        with mcols[i]:
            st.metric(sym, f"{close.iloc[-1]:.2f}", f"{(ret*100):.1f}%" if np.isfinite(ret) else "n/a")
            st.write(f"Trend: **{tr}**")
            st.write(f"RSI: **{r:.1f}**")

    spy_df = get_hist("SPY")
    if spy_df.empty:
        st.warning("SPY data unavailable; cannot compute RS vs SPY.")
        return

    spy = spy_df["Close"].dropna()
    if len(spy) < (rs_long + 10):
        st.warning("Not enough SPY history for these lookbacks.")
        return

    st.subheader("Sector / Metals Rotation + Strength (Relative Strength vs SPY)")
    sector_rows = []
    for name, etf in SECTOR_ETFS.items():
        d = get_hist(etf)
        if d.empty:
            continue
        close = d["Close"].dropna()
        if len(close) < (rs_long + 10):
            continue

        rs_s = float(rs_vs_spy(close, spy, int(rs_short)).iloc[-1])
        rs_l = float(rs_vs_spy(close, spy, int(rs_long)).iloc[-1])

        rs_s_c = clamp_rs(rs_s, -RS_CAP, RS_CAP)
        rs_l_c = clamp_rs(rs_l, -RS_CAP, RS_CAP)

        rot = rs_s - rs_l
        rot_c = clamp_rs(rot, -ROT_CAP, ROT_CAP)

        tr = trend_label(close, int(ema_trend_len))
        r = float(rsi_wilder(close, int(rsi_len)).iloc[-1])

        score = strength_meter(rs_s_c, rot_c, tr)

        sector_rows.append({
            "Group": name,
            "ETF": etf,
            "Strength": score,
            "Meter": strength_label(score),
            f"RS vs SPY ({rs_short})": rs_s,
            f"RS vs SPY ({rs_long})": rs_l,
            "Rotation (RS short - RS long)": rot,
            "Trend": tr,
            "RSI": r
        })

    sectors = pd.DataFrame(sector_rows)
    if sectors.empty:
        st.warning("Sector data unavailable right now (yfinance returned empty). Try Refresh.")
        return

    sectors = sectors.sort_values(["Strength", "Rotation (RS short - RS long)"], ascending=[False, False])

    styled = (
        sectors
        .style
        .format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            f"RS vs SPY ({rs_long})": "{:.2%}",
            "Rotation (RS short - RS long)": "{:.2%}",
            "RSI": "{:.1f}"
        })
        .applymap(meter_style, subset=["Meter"])
        .applymap(strength_style, subset=["Strength"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

    st.subheader("✅ Today Watchlist (Auto-built from Rotation IN + Leaders)")

    top_groups = sectors.head(int(top_sectors_in))[["Group","ETF","Strength","Meter"]].to_dict("records")
    st.write("**Top Groups IN:** " + ", ".join([f"{g['Group']}({g['ETF']}) {g['Meter']} {g['Strength']}" for g in top_groups]))

    watchlist: List[Dict] = []
    for g in top_groups:
        group_name = g["Group"]
        names = SECTOR_TICKERS.get(group_name, [])
        if not names:
            continue

        infos = []
        for sym in names[:min(30, len(names))]:
            info = analyze_ticker(sym, spy, int(rs_short), int(rs_long), int(ema_trend_len), int(rsi_len))
            if info is not None:
                info["Group"] = group_name
                infos.append(info)

        if not infos:
            continue

        infos = sorted(infos, key=lambda x: (x["Strength"], x["Rotation"], x["RS_short"]), reverse=True)

        if strict_pullback:
            infos = [x for x in infos if pullback_zone_ok(x["Trend"], x["RSI"], pb_low, pb_high)]

        pick = infos[:int(leaders_per_sector)]
        watchlist.extend(pick)

    if not watchlist:
        st.warning("Watchlist is empty under current settings. Loosen pullback filter or increase scan sizes.")
        return

    wdf = pd.DataFrame([{
        "Group": x["Group"],
        "Ticker": x["Ticker"],
        "Strength": x["Strength"],
        "Meter": x["Meter"],
        "Trend": x["Trend"],
        "RSI": x["RSI"],
        f"RS vs SPY ({rs_short})": x["RS_short"],
        "Rotation": x["Rotation"],
        "Trigger": x["TriggerStatus"],
        "TF": x["TF"],
        "Entry": x["Entry"],
        "Stop": x["Stop"],
    } for x in watchlist]).sort_values(["Strength","Rotation"], ascending=[False, False])

    wstyled = (
        wdf.style
        .format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            "Rotation": "{:.2%}",
            "RSI": "{:.1f}"
        })
        .applymap(meter_style, subset=["Meter"])
        .applymap(strength_style, subset=["Strength"])
    )
    st.dataframe(wstyled, use_container_width=True, hide_index=True, height=420)

    st.write("### 📌 Watchlist Write-ups (click to expand)")
    for rec in wdf.head(20).to_dict("records"):
        full = analyze_ticker(rec["Ticker"], spy, int(rs_short), int(rs_long), int(ema_trend_len), int(rsi_len))
        if full is None:
            continue
        full["Group"] = rec["Group"]
        with st.expander(f"{full['Group']} — {full['Ticker']} | {full['Meter']} {full['Strength']}/100 | {full['TriggerStatus']}"):
            writeup_block(full, pb_low, pb_high)

    st.subheader("🔎 Quick Ticker Search (Why is this a candidate?)")
    q = st.text_input("Type a ticker:", value="AAPL")
    if q:
        info = analyze_ticker(q.strip().upper(), spy, int(rs_short), int(rs_long), int(ema_trend_len), int(rsi_len))
        if info is None:
            st.warning("No data returned (bad ticker or yfinance empty). Try another symbol.")
        else:
            writeup_block(info, pb_low, pb_high)

def show_ticker_analyzer():
    st.title("🔎 Ticker Analyzer — Explain the Score + STRAT Context + Trade Plan Notes (V1.3.0)")
    st.caption("Type any ticker and get a swing-trader style gameplan automatically.")

    with st.expander("Analyzer Settings", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.2])
        with c1:
            ticker = st.text_input("Ticker", value="AAPL")
        with c2:
            rs_short = st.selectbox("RS Lookback (short)", [21, 30, 42], index=0, key="ta_rs_s")
        with c3:
            rs_long = st.selectbox("RS Lookback (long)", [63, 90, 126], index=0, key="ta_rs_l")
        with c4:
            ema_trend_len = st.selectbox("Trend EMA", [50, 100, 200], index=0, key="ta_ema")
        with c5:
            rsi_len = st.selectbox("RSI Length", [7, 14, 21], index=1, key="ta_rsi")

    pb1, pb2 = st.columns(2)
    with pb1:
        pb_low = st.slider("RSI Pullback Low (UP trend)", 25, 60, 40, key="ta_pb_low")
    with pb2:
        pb_high = st.slider("RSI Pullback High (UP trend)", 35, 75, 55, key="ta_pb_high")

    spy_df = get_hist("SPY")
    if spy_df.empty:
        st.warning("SPY data unavailable; cannot compute RS vs SPY.")
        return

    spy = spy_df["Close"].dropna()
    if len(spy) < (rs_long + 10):
        st.warning("Not enough SPY history for these lookbacks.")
        return

    if ticker:
        info = analyze_ticker(ticker.strip().upper(), spy, int(rs_short), int(rs_long), int(ema_trend_len), int(rsi_len))
        if info is None:
            st.warning("No data returned (bad ticker or yfinance empty). Try another symbol.")
        else:
            writeup_block(info, pb_low, pb_high)

# =========================
# SCANNER (UNCHANGED CORE)
# =========================
def magnitude_metrics(
    bias: str,
    d: pd.DataFrame,
    entry: Optional[float],
    stop: Optional[float]
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if d is None or d.empty or len(d) < 80:
        return None, None, None, None

    close = float(d["Close"].iloc[-1])
    atr = atr14(d)
    if not math.isfinite(atr) or atr <= 0:
        return None, None, None, None

    atrp = (atr / close) * 100.0

    if entry is None or stop is None:
        return None, atrp, None, None

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

def show_scanner():
    st.title("STRAT Regime Scanner (Auto LONG/SHORT + Magnitude) — V1.3.0")
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

        colR1, _ = st.columns([1, 3])
        with colR1:
            if st.button("Refresh data"):
                st.cache_data.clear()
                st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

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

    st.subheader("Sectors + Metals — ranked after bias is known")
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

    st.subheader("Drill into a group (ranks candidates in bias direction + magnitude)")

    sector_choice = st.selectbox("Choose a sector/metals group:", options=list(SECTOR_TICKERS.keys()), index=0)
    tickers = SECTOR_TICKERS.get(sector_choice, [])
    st.write(f"Selected: **{sector_choice}** ({SECTOR_ETFS.get(sector_choice,'')}) — tickers in list: **{len(tickers)}**")

    scan_n = st.slider("How many tickers to scan", min_value=1, max_value=len(tickers), value=min(15, len(tickers)))
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

        trigger_status = "READY" if (flags["W_Inside"] or flags["D_Inside"]) else "WAIT (No Inside Bar)"

        cand = {
            "Ticker": t,
            "TriggerStatus": trigger_status,
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
            "Ticker","TriggerStatus","TotalScore","SetupScore","MagScore","TF","Entry","Stop","Room","RR","ATR%",
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
                "Ticker","TriggerStatus","SetupScore","MagScore","TotalScore","TF","Entry","Stop","Room","RR","ATR%",
                "W_Inside","D_Inside","W_212Up","D_212Up","W_212Dn","D_212Dn",
                "M_Bull","W_Bull","D_Bull","M_Bear","W_Bear","D_Bear"
            ]],
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Quick Market Read")

    if bias in ("LONG", "SHORT"):
        rotation_in = [f"{r['Sector']}({r['ETF']})" for _, r in sectors_df.head(3).iterrows()]
        rotation_out = [f"{r['Sector']}({r['ETF']})" for _, r in sectors_df.tail(3).iterrows()]
    else:
        rotation_in, rotation_out = [], []

    if bias == "MIXED" or strength < 50:
        plan = "Plan: Defensive. Trade smaller, or wait for A+ triggers."
        badge = "🟠"
    elif bias == "LONG":
        plan = "Plan: LONG only. Focus strong groups with triggers."
        badge = "🟢"
    else:
        plan = "Plan: SHORT only. Focus weak groups with triggers."
        badge = "🔴"

    st.write(
        f"Bias: **{bias}** {badge} | "
        f"Strength: **{strength}/100** | "
        f"Bull–Bear diff: **{bull_bear_diff}**"
    )

    if rotation_in:
        st.write("### Rotation IN")
        st.write(", ".join(rotation_in))

    if rotation_out:
        st.write("### Rotation OUT")
        st.write(", ".join(rotation_out))

    st.success(plan)
    st.caption(
        "Trigger logic: LONG = break of Inside Bar high / stop below low. "
        "SHORT = break of Inside Bar low / stop above high. Weekly triggers preferred."
    )

# =========================
# SIDEBAR NAV
# =========================
st.sidebar.title("Navigation")
show_market_dash = st.sidebar.toggle("Enable Market Dashboard", value=True)

pages = ["Scanner", "📘 User Guide", "🔎 Ticker Analyzer"]
if show_market_dash:
    pages.insert(1, "📊 Market Dashboard")

page = st.sidebar.radio("Go to", pages)
st.sidebar.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# =========================
# ROUTING
# =========================
if page == "📘 User Guide":
    show_user_guide()
elif page == "📊 Market Dashboard":
    show_market_dashboard()
elif page == "🔎 Ticker Analyzer":
    show_ticker_analyzer()
else:
    show_scanner()
