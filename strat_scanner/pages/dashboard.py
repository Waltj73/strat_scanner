from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from strat_scanner.data import get_hist
from strat_scanner.indicators import (
    rsi_wilder, rs_vs_spy, trend_label,
    strength_meter, strength_label, pullback_zone_ok,
    RS_CAP, ROT_CAP, clamp_float
)

MARKET_ETFS = {"SPY":"SPY","QQQ":"QQQ","IWM":"IWM","DIA":"DIA","^VIX":"^VIX"}

SECTOR_ETFS = {
    "Energy":"XLE","Comm Services":"XLC","Staples":"XLP","Materials":"XLB","Industrials":"XLI",
    "Real Estate":"XLRE","Discretionary":"XLY","Utilities":"XLU","Financials":"XLF","Technology":"XLK","Health Care":"XLV",
    "Metals - Gold":"GLD","Metals - Silver":"SLV","Metals - Copper":"CPER","Metals - Platinum":"PPLT","Metals - Palladium":"PALL",
}

SECTOR_TICKERS: Dict[str, List[str]] = {
    "Energy": ["XOM","CVX","COP","EOG","SLB","HAL","PSX","MPC","VLO","OXY"],
    "Comm Services": ["GOOGL","META","NFLX","TMUS","VZ","DIS","CMCSA","TTWO","SPOT","ROKU"],
    "Staples": ["PG","KO","PEP","WMT","COST","MDLZ","CL","KMB","GIS","HSY"],
    "Materials": ["LIN","SHW","NUE","FCX","NEM","VMC","ALB","MOS","DD","APD"],
    "Industrials": ["CAT","DE","HON","GE","LMT","RTX","BA","UNP","UPS","ETN"],
    "Real Estate": ["PLD","AMT","EQIX","PSA","O","WELL","SPG","CCI","VICI","AVB"],
    "Discretionary": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","CMG"],
    "Utilities": ["NEE","DUK","SO","AEP","EXC","XEL","SRE","ED","PEG","AWK"],
    "Financials": ["BRK-B","JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP"],
    "Technology": ["AAPL","MSFT","NVDA","AVGO","CRM","ORCL","ADBE","AMD","QCOM","TXN"],
    "Health Care": ["UNH","JNJ","LLY","PFE","MRK","ABBV","TMO","ABT","DHR","ISRG"],
    "Metals - Gold": ["GLD"],
    "Metals - Silver": ["SLV"],
    "Metals - Copper": ["CPER"],
    "Metals - Platinum": ["PPLT"],
    "Metals - Palladium": ["PALL"],
}

def meter_style(val: str) -> str:
    if val == "STRONG":
        return "background-color:#114b2b;color:white;font-weight:700;"
    if val == "NEUTRAL":
        return "background-color:#5a4b11;color:white;font-weight:700;"
    return "background-color:#5a1111;color:white;font-weight:700;"

def strength_style(v):
    try:
        x = float(v)
    except Exception:
        return ""
    x = max(0.0, min(100.0, x))
    # simple red→green
    g = int((x / 100.0) * 120)
    r = 120 - g
    return f"background-color: rgb({r},{g},30); color: white; font-weight: 700;"

def analyze_one(ticker: str, spy_close: pd.Series, rs_short: int, rs_long: int, ema_len: int, rsi_len: int):
    d = get_hist(ticker)
    if d.empty:
        return None
    close = d["Close"].dropna()
    if len(close) < (rs_long + 20):
        return None

    tr = trend_label(close, ema_len)
    rsi_v = float(rsi_wilder(close, rsi_len).iloc[-1])

    rs_s = float(rs_vs_spy(close, spy_close, rs_short).iloc[-1])
    rs_l = float(rs_vs_spy(close, spy_close, rs_long).iloc[-1])
    rot = rs_s - rs_l

    # cap so scores don’t saturate
    rs_s_c = clamp_float(rs_s, -RS_CAP, RS_CAP)
    rot_c = clamp_float(rot, -ROT_CAP, ROT_CAP)

    strength = strength_meter(rs_s_c, rot_c, tr)
    meter = strength_label(strength)

    return {
        "Ticker": ticker,
        "Trend": tr,
        "RSI": rsi_v,
        "RS_short": rs_s,
        "RS_long": rs_l,
        "Rotation": rot,
        "Strength": strength,
        "Meter": meter
    }

def show_dashboard():
    st.title("📊 Market Dashboard (Rotation • Strength • Leaders • Watchlist)")

    with st.expander("Dashboard Settings", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1.2])
        with c1:
            rs_short = st.selectbox("RS short", [21, 30, 42], index=0)
        with c2:
            rs_long = st.selectbox("RS long", [63, 90, 126], index=0)
        with c3:
            ema_len = st.selectbox("Trend EMA", [50, 100, 200], index=0)
        with c4:
            rsi_len = st.selectbox("RSI len", [7, 14, 21], index=1)
        with c5:
            if st.button("Refresh data"):
                st.cache_data.clear()
                st.rerun()

    with st.expander("Watchlist Settings", expanded=True):
        w1, w2, w3, w4, w5 = st.columns([1,1,1,1,1.2])
        with w1:
            top_sectors_in = st.slider("Top Sectors IN", 1, 8, 3)
        with w2:
            leaders_per_sector = st.slider("Leaders per sector", 3, 10, 5)
        with w3:
            pb_low = st.slider("Pullback RSI Low", 25, 60, 40)
        with w4:
            pb_high = st.slider("Pullback RSI High", 35, 75, 55)
        with w5:
            strict_pb = st.checkbox("Strict pullback only", value=False)

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # sentiment bar
    st.subheader("Overall Market Sentiment")
    cols = st.columns(len(MARKET_ETFS))
    for i, sym in enumerate(MARKET_ETFS.values()):
        d = get_hist(sym)
        if d.empty:
            cols[i].metric(sym, "n/a", "n/a")
            continue
        close = d["Close"].dropna()
        if len(close) < 30:
            cols[i].metric(sym, "n/a", "n/a")
            continue
        tr = trend_label(close, ema_len)
        r = float(rsi_wilder(close, rsi_len).iloc[-1])
        ret = float((close.iloc[-1] / close.iloc[-rs_short] - 1)) if len(close) > rs_short else np.nan
        cols[i].metric(sym, f"{close.iloc[-1]:.2f}", f"{ret*100:.1f}%" if np.isfinite(ret) else "n/a")
        cols[i].write(f"Trend: **{tr}**")
        cols[i].write(f"RSI: **{r:.1f}**")

    spy_df = get_hist("SPY")
    if spy_df.empty:
        st.error("SPY data unavailable.")
        return
    spy_close = spy_df["Close"].dropna()
    if len(spy_close) < (rs_long + 30):
        st.error("Not enough SPY history for selected lookbacks.")
        return

    # sector rotation table
    st.subheader("Sector / Metals Rotation + Strength (RS vs SPY)")
    rows = []
    for grp, etf in SECTOR_ETFS.items():
        info = analyze_one(etf, spy_close, rs_short, rs_long, ema_len, rsi_len)
        if not info:
            continue
        rows.append({
            "Group": grp,
            "ETF": etf,
            "Strength": info["Strength"],
            "Meter": info["Meter"],
            f"RS vs SPY ({rs_short})": info["RS_short"],
            f"RS vs SPY ({rs_long})": info["RS_long"],
            "Rotation (short-long)": info["Rotation"],
            "Trend": info["Trend"],
            "RSI": info["RSI"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        st.error("No sector ETF rows built (yfinance returned empty).")
        return

    df = df.sort_values(["Strength","Rotation (short-long)"], ascending=[False, False])

    styled = (
        df.style
        .format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            f"RS vs SPY ({rs_long})": "{:.2%}",
            "Rotation (short-long)": "{:.2%}",
            "RSI": "{:.1f}",
        })
        .applymap(meter_style, subset=["Meter"])
        .applymap(strength_style, subset=["Strength"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

    # watchlist
    st.subheader("✅ Today Watchlist (Auto-built from Rotation IN + Leaders)")
    top_groups = df.head(int(top_sectors_in))[["Group","ETF","Strength","Meter"]].to_dict("records")
    st.write("Top Groups IN: " + ", ".join([f"{g['Group']}({g['ETF']}) {g['Meter']} {g['Strength']}" for g in top_groups]))

    picks = []
    for g in top_groups:
        group = g["Group"]
        names = SECTOR_TICKERS.get(group, [])
        if not names:
            continue

        infos = []
        for t in names[:25]:
            r = analyze_one(t, spy_close, rs_short, rs_long, ema_len, rsi_len)
            if r:
                r["Group"] = group
                infos.append(r)

        infos = sorted(infos, key=lambda x: (x["Strength"], x["Rotation"], x["RS_short"]), reverse=True)

        if strict_pb:
            infos = [x for x in infos if pullback_zone_ok(x["Trend"], x["RSI"], pb_low, pb_high)]

        picks.extend(infos[:leaders_per_sector])

    if not picks:
        st.warning("Watchlist empty with current settings. Loosen strict pullback.")
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
    } for x in picks]).sort_values(["Strength","Rotation"], ascending=[False, False])

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


