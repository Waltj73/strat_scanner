import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# =========================
# SECTOR ROTATION CONFIG
# =========================
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

BENCHMARK = "SPY"

# Breadth proxy baskets (editable)
SECTOR_BREADTH_BASKETS = {
    "Technology": ["AAPL","MSFT","NVDA","AVGO","CRM","ADBE","AMD","ORCL"],
    "Financials": ["JPM","BAC","WFC","GS","MS","C","BLK","SCHW"],
    "Health Care": ["UNH","JNJ","LLY","PFE","MRK","ABBV","TMO","ABT"],
    "Consumer Discretionary": ["AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","BKNG"],
    "Consumer Staples": ["PG","KO","PEP","WMT","COST","PM","MDLZ","CL"],
    "Energy": ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO"],
    "Industrials": ["CAT","GE","HON","UNP","DE","BA","LMT","ETN"],
    "Materials": ["LIN","APD","ECL","SHW","FCX","NEM","DOW","DD"],
    "Utilities": ["NEE","DUK","SO","AEP","EXC","SRE","XEL","ED"],
    "Real Estate": ["PLD","AMT","EQIX","O","PSA","CCI","WELL","SPG"],
    "Communication Services": ["GOOGL","META","NFLX","DIS","TMUS","VZ","T","CHTR"],
}

# =========================
# DATA
# =========================
@st.cache_data(ttl=60 * 30)
def fetch_close_volume(tickers, period="9mo", interval="1d"):
    """
    Returns DataFrame with MultiIndex columns:
      level0: Close/Volume
      level1: ticker
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(dict.fromkeys([t.upper().strip() for t in tickers if t]))

    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="column",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    # yfinance can return different shapes depending on number of tickers
    if isinstance(data.columns, pd.MultiIndex):
        # expected
        return data

    # Single ticker fallback (flat columns)
    if "Close" in data.columns and "Volume" in data.columns and len(tickers) == 1:
        t = tickers[0]
        out = data[["Close", "Volume"]].copy()
        out.columns = pd.MultiIndex.from_product([out.columns, [t]])
        return out

    # Best effort fallback
    raise ValueError("Unexpected yfinance format. Try fewer tickers or check data source.")

# =========================
# INDICATORS (close/volume only)
# =========================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1/length, adjust=False).mean() / down.ewm(alpha=1/length, adjust=False).mean()
    return 100 - (100 / (1 + rs))

def atr_pct(close: pd.Series, length: int = 14) -> pd.Series:
    # proxy ATR using absolute returns
    tr = close.pct_change().abs()
    return tr.rolling(length).mean()

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def score_0_20(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 10.0
    t = (x - lo) / (hi - lo)
    return 20.0 * clamp01(t)

# =========================
# ROTATION SCORE (0-100)
# =========================
def compute_rotation_score(sym_close: pd.Series, sym_vol: pd.Series,
                           bench_close: pd.Series,
                           lookback_rs=20, lookback_trend=5) -> dict:
    df = pd.DataFrame({"sym": sym_close, "bench": bench_close, "vol": sym_vol}).dropna()
    if len(df) < (lookback_rs + lookback_trend + 20):
        return {"total": np.nan}

    sym = df["sym"]
    bench = df["bench"]
    vol = df["vol"]

    # 1) Relative strength trend (vs SPY)
    sym_ret = sym.pct_change(lookback_rs) * 100
    bench_ret = bench.pct_change(lookback_rs) * 100
    rs = sym_ret - bench_ret
    rs_trend_raw = rs.iloc[-1] - rs.shift(lookback_trend).iloc[-1]
    rs_score = score_0_20(rs_trend_raw, lo=-2.0, hi=2.0)

    # 2) Momentum improvement (RSI trend + “early rotation zone” bonus)
    r = rsi(sym, 14)
    r_now = float(r.iloc[-1])
    r_trend = float(r.iloc[-1] - r.shift(lookback_trend).iloc[-1])
    mom_score = score_0_20(r_trend, lo=-6.0, hi=6.0)
    if 45 <= r_now <= 65:
        mom_score = min(20.0, mom_score + 6.0)

    # 3) Pullback strength (last 5d vs SPY)
    sym_5 = float(sym.pct_change(5).iloc[-1] * 100)
    bench_5 = float(bench.pct_change(5).iloc[-1] * 100)
    pull_raw = sym_5 - bench_5
    pull_score = score_0_20(pull_raw, lo=-2.0, hi=2.0)

    # 4) Volume accumulation (up-day vol / down-day vol)
    rets = sym.pct_change()
    recent = pd.DataFrame({"ret": rets, "vol": vol}).dropna().tail(lookback_rs)
    up = recent.loc[recent["ret"] > 0, "vol"].mean()
    dn = recent.loc[recent["ret"] < 0, "vol"].mean()
    if np.isnan(up) or np.isnan(dn) or dn == 0:
        vol_ratio = 1.0
    else:
        vol_ratio = float(up / dn)
    vol_score = score_0_20(vol_ratio, lo=0.8, hi=1.3)

    # 5) Breadth is added outside for sectors; neutral here
    breadth_score = 10.0

    total = float(rs_score + mom_score + pull_score + vol_score + breadth_score)
    return {
        "total": total,
        "rs
