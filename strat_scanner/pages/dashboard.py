from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from strat_scanner.data import get_hist
from strat_scanner.indicators import rsi_wilder, total_return, rs_vs_spy, trend_label, strength_meter, strength_label
from strat_scanner.engine import analyze_ticker, writeup_block

MARKET_ETFS = {"SPY":"SPY","QQQ":"QQQ","IWM":"IWM","DIA":"DIA"}

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
    "Metals - Gold": "GLD",
    "Metals - Silver": "SLV",
    "Metals - Copper": "CPER",
    "Metals - Platinum": "PPLT",
    "Metals - Palladium": "PALL",
}

SECTOR_TICKERS: Dict[str, List[str]] = {
    "Energy": ["XOM","CVX","COP","EOG","SLB"],
    "Comm Services": ["GOOGL","META","NFLX","TMUS","DIS"],
    "Staples": ["PG","KO","PEP","WMT","COST"],
    "Materials": ["LIN","SHW","NUE","FCX","NEM"],
    "Industrials": ["CAT","HON","GE","LMT","RTX"],
    "Real Estate": ["PLD","AMT","EQIX","O","SPG"],
    "Discretionary": ["AMZN","TSLA","HD","MCD","NKE"],
    "Utilities": ["NEE","DUK","SO","AEP","EXC"],
    "Financials": ["BRK-B","JPM","BAC","GS","SCHW"],
    "Technology": ["AAPL","MSFT","NVDA","AVGO","AMD"],
    "Health Care": ["UNH","JNJ","LLY","PFE","MRK"],
}

def show_dashboard():
    st.title("📊 Market Dashboard (Sentiment • Rotation • Leaders • Watchlist)")

    with st.expander("Dashboard Settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        rs_short = c1.selectbox("RS Lookback (short)", [21, 30, 42], index=0)
        rs_long = c2.selectbox("RS Lookback (long)", [63, 90, 126], index=0)
        ema_trend_len = c3.selectbox("Trend EMA", [50, 100, 200], index=0)
        rsi_len = c4.selectbox("RSI Length", [7, 14, 21], index=1)

        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    with st.expander("Today Watchlist Settings", expanded=True):
        w1, w2, w3, w4, w5 = st.columns([1, 1, 1, 1, 1.2])
        top_sectors_in = w1.slider("Top Sectors IN", 1, 6, 3)
        leaders_per_sector = w2.slider("Leaders per sector", 3, 10, 5)
        pb_low = w3.slider("RSI Pullback Low (UP trend)", 25, 60, 40)
        pb_high = w4.slider("RSI Pullback High (UP trend)", 35, 75, 55)
        strict_pullback = w5.checkbox("Strict pullback filter", value=False)

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    st.subheader("Overall Market Sentiment")
    mcols = st.columns(len(MARKET_ETFS))
    for i, sym in enumerate(MARKET_ETFS.values()):
        d = get_hist(sym)
        if d is None or d.empty:
            mcols[i].metric(sym, "n/a", "n/a")
            continue
        close = d["Close"].dropna()
        tr = trend_label(close, int(ema_trend_len))
        r = float(rsi_wilder(close, int(rsi_len)).iloc[-1])
        ret = float(total_return(close, int(rs_short)).iloc[-1]) if len(close) > rs_short else np.nan
        mcols[i].metric(sym, f"{close.iloc[-1]:.2f}", f"{(ret*100):.1f}%" if np.isfinite(ret) else "n/a")
        mcols[i].write(f"Trend: **{tr}**")
        mcols[i].write(f"RSI: **{r:.1f}**")

    spy_df = get_hist("SPY")
    if spy_df is None or spy_df.empty:
        st.error("SPY data unavailable.")
        return
    spy = spy_df["Close"].dropna()
    if len(spy) < (rs_long + 10):
        st.error("Not enough SPY history.")
        return

    st.subheader("Sector / Metals Rotation + Strength (Relative Strength vs SPY)")
    sector_rows = []
    for name, etf in SECTOR_ETFS.items():
        d = get_hist(etf)
        if d is None or d.empty:
            continue
        close = d["Close"].dropna()
        if len(close) < (rs_long + 10):
            continue
        rs_s = float(rs_vs_spy(close, spy, int(rs_short)).iloc[-1])
        rs_l = float(rs_vs_spy(close, spy, int(rs_long)).iloc[-1])
        rot = rs_s - rs_l
        tr = trend_label(close, int(ema_trend_len))
        r = float(rsi_wilder(close, int(rsi_len)).iloc[-1])
        strength = int(strength_meter(rs_s, rot, tr))
        sector_rows.append({"Group": name, "ETF": etf, "Strength": strength, "Meter": strength_label(strength),
                            "Rotation": rot, "Trend": tr, "RSI": r})
    sectors = pd.DataFrame(sector_rows).sort_values(["Strength","Rotation"], ascending=[False, False])
    st.dataframe(sectors, use_container_width=True, hide_index=True, height=420)

    st.subheader("✅ Today Watchlist (Auto-built from Rotation IN + Leaders)")
    top_groups = sectors.head(int(top_sectors_in))["Group"].tolist()

    watchlist = []
    for g in top_groups:
        for sym in SECTOR_TICKERS.get(g, [])[:30]:
            info = analyze_ticker(sym, spy, int(rs_short), int(rs_long), int(ema_trend_len), int(rsi_len))
            if info:
                info["Group"] = g
                watchlist.append(info)

    if strict_pullback:
        watchlist = [x for x in watchlist if x["Trend"] == "UP" and pb_low <= x["RSI"] <= pb_high]

    watchlist = sorted(watchlist, key=lambda x: (x["Strength"], x["Rotation"], x["RS_short"]), reverse=True)
    watchlist = watchlist[: int(top_sectors_in) * int(leaders_per_sector)]

    if not watchlist:
        st.warning("Watchlist empty. Loosen filters.")
        return

    watch_df = pd.DataFrame(watchlist)
    st.dataframe(
        watch_df[["Group","Ticker","Strength","Meter","Trend","RSI","RS_short","Rotation","TriggerStatus","Entry","Stop"]],
        use_container_width=True,
        hide_index=True,
        height=420
    )

    st.write("### 📌 Watchlist Write-ups (click to expand)")
    for rec in watch_df.head(20).to_dict("records"):
        with st.expander(f"{rec['Group']} — {rec['Ticker']} | {rec['Meter']} {rec['Strength']}/100 | {rec['TriggerStatus']}"):
            writeup_block(rec, pb_low, pb_high)
