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
        "rs_score": rs_score,
        "mom_score": mom_score,
        "pull_score": pull_score,
        "vol_score": vol_score,
        "breadth_score": breadth_score,
        "rs_trend_raw": float(rs_trend_raw),
        "rsi": r_now,
        "vol_ratio": float(vol_ratio),
    }

def sector_breadth_score(basket_closes: dict) -> float:
    """
    Breadth = % of basket above 50EMA AND 21EMA rising over last week
    Returns 0..20
    """
    if not basket_closes:
        return 10.0

    hits = 0
    total = 0
    for _, c in basket_closes.items():
        c = c.dropna()
        if len(c) < 80:
            continue
        e50 = ema(c, 50)
        e21 = ema(c, 21)
        ok = (c.iloc[-1] > e50.iloc[-1]) and (e21.iloc[-1] > e21.iloc[-6])
        total += 1
        hits += int(ok)

    if total == 0:
        return 10.0

    pct = hits / total
    return float(20.0 * pct)

def build_sector_rotation_table(period="9mo"):
    sectors = list(SECTOR_ETFS.keys())
    etfs = [SECTOR_ETFS[s] for s in sectors]
    symbols = [BENCHMARK] + etfs

    data = fetch_close_volume(symbols, period=period, interval="1d")
    closes = data["Close"]
    vols = data["Volume"]

    bench_close = closes[BENCHMARK].dropna()

    rows = []
    for sector, etf in SECTOR_ETFS.items():
        if etf not in closes.columns:
            continue

        sym_close = closes[etf].dropna()
        sym_vol = vols[etf].dropna()

        base = compute_rotation_score(sym_close, sym_vol, bench_close)

        basket = SECTOR_BREADTH_BASKETS.get(sector, [])
        basket_closes = {}
        if basket:
            bdata = fetch_close_volume(basket, period=period, interval="1d")
            bclose = bdata["Close"]
            for t in basket:
                if t in bclose.columns:
                    basket_closes[t] = bclose[t]

        breadth = sector_breadth_score(basket_closes)

        # Replace neutral breadth with real breadth
        total = (base["total"] - base["breadth_score"]) + breadth

        # Acceleration = score today - score 5 trading days ago
        accel = np.nan
        if len(sym_close) > 60 and len(bench_close) > 60:
            base_ago = compute_rotation_score(sym_close.iloc[:-5], sym_vol.iloc[:-5], bench_close.iloc[:-5])
            total_ago = (base_ago["total"] - base_ago["breadth_score"]) + breadth
            accel = total - total_ago

        rows.append({
            "Sector": sector,
            "ETF": etf,
            "Rotation Score": round(total, 1),
            "Accel (1w)": round(float(accel), 1) if pd.notna(accel) else np.nan,
            "Breadth (0-20)": round(breadth, 1),
        })

    df = pd.DataFrame(rows).sort_values(["Rotation Score", "Accel (1w)"], ascending=False).reset_index(drop=True)
    return df

# =========================
# WATCHLIST + EARLY BREAKOUTS
# =========================
def compute_early_breakout_score(close: pd.Series, vol: pd.Series, bench_close: pd.Series) -> dict:
    df = pd.DataFrame({"c": close, "v": vol, "b": bench_close}).dropna()
    if len(df) < 90:
        return {"breakout_score": np.nan}

    c = df["c"]; v = df["v"]; b = df["b"]

    # RS improvement (0-25)
    rs20 = (c.pct_change(20) - b.pct_change(20)) * 100
    rs_tr = float(rs20.iloc[-1] - rs20.iloc[-6])
    s_rs = score_0_20(rs_tr, lo=-2.0, hi=2.0) * 1.25

    # EMA alignment (0-20)
    e21 = ema(c, 21)
    ema_ok = (c.iloc[-1] > e21.iloc[-1]) and (e21.iloc[-1] > e21.iloc[-6])
    s_ema = 20.0 if ema_ok else 0.0

    # Compression (0-20): ATR% down vs 10 days ago
    a = atr_pct(c, 14)
    comp = float(a.iloc[-11] - a.iloc[-1])
    s_comp = score_0_20(comp, lo=-0.002, hi=0.002)

    # Up-volume dominance (0-20)
    rets = c.pct_change()
    recent = pd.DataFrame({"ret": rets, "vol": v}).dropna().tail(20)
    up = recent.loc[recent["ret"] > 0, "vol"].mean()
    dn = recent.loc[recent["ret"] < 0, "vol"].mean()
    vr = 1.0 if (np.isnan(up) or np.isnan(dn) or dn == 0) else float(up / dn)
    s_vol = score_0_20(vr, lo=0.85, hi=1.35)

    # Near resistance (0-15): within 1.5% of 20d high but not above
    hi20 = float(c.rolling(20).max().iloc[-1])
    dist = float((hi20 - c.iloc[-1]) / hi20)
    near = (0 <= dist <= 0.015)
    s_res = 15.0 if near else 0.0

    total = float(s_rs + s_ema + s_comp + s_vol + s_res)
    return {
        "breakout_score": round(total, 1),
        "rs_trend": round(rs_tr, 2),
        "vol_ratio": round(vr, 2),
        "compression": round(comp, 4),
        "near_20d_high": bool(near),
    }

def build_rotation_watchlist(sector_threshold=70, top_sectors=6, period="9mo"):
    sector_df = build_sector_rotation_table(period=period)
    leaders = sector_df[sector_df["Rotation Score"] >= sector_threshold].head(top_sectors)
    sectors = leaders["Sector"].tolist()

    tickers = []
    for s in sectors:
        tickers += SECTOR_BREADTH_BASKETS.get(s, [])
    tickers = sorted(list(set(tickers)))

    return leaders, tickers
