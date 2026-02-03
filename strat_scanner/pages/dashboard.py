# strat_scanner/pages/dashboard.py
# Market Dashboard page (rotation + strength + watchlist + writeups + quick analyzer)

from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from strat_scanner.data import get_hist
from strat_scanner.indicators import (
    rsi_wilder,
    total_return,
    rs_vs_spy,
    trend_label,
    clamp_rs,
    strength_meter,
    strength_label,
)

from strat_scanner.strat import best_trigger
from strat_scanner.engine import analyze_ticker, writeup_block

# -------------------------
# Universe (ETFs + tickers)
# -------------------------
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


def show_dashboard():
    st.title("📊 Market Dashboard (Sentiment • Rotation • Leaders • Watchlist)")

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

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # -------------------------
    # Market sentiment panel
    # -------------------------
    st.subheader("Overall Market Sentiment")
    market_syms = list(MARKET_ETFS.values()) + ["^VIX"]
    mcols = st.columns(len(market_syms))

    for i, sym in enumerate(market_syms):
        d = get_hist(sym)
        if d is None or d.empty:
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

    # -------------------------
    # SPY anchor
    # -------------------------
    spy_df = get_hist("SPY")
    if spy_df is None or spy_df.empty:
        st.warning("SPY data unavailable; cannot compute RS vs SPY.")
        return

    spy = spy_df["Close"].dropna()
    if len(spy) < (rs_long + 10):
        st.warning("Not enough SPY history for these lookbacks.")
        return

    # -------------------------
    # Sector / metals rotation table
    # -------------------------
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

        score = strength_meter(rs_s, rot, tr)
        sector_rows.append({
            "Group": name,
            "ETF": etf,
            "Strength": int(score),
            "Meter": strength_label(int(score)),
            f"RS vs SPY ({rs_short})": rs_s,
            f"RS vs SPY ({rs_long})": rs_l,
            "Rotation (RS short - RS long)": rot,
            "Trend": tr,
            "RSI": r
        })

    sectors = pd.DataFrame(sector_rows)
    if sectors.empty:
        st.warning("Sector data unavailable right now. Try Refresh.")
        return

    sectors = sectors.sort_values(["Strength", "Rotation (RS short - RS long)"], ascending=[False, False])

    st.dataframe(
        sectors.style.format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            f"RS vs SPY ({rs_long})": "{:.2%}",
            "Rotation (RS short - RS long)": "{:.2%}",
            "RSI": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=420
    )

    # -------------------------
    # Auto Watchlist
    # -------------------------
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
            infos = [x for x in infos if (x["Trend"] == "UP" and pb_low <= x["RSI"] <= pb_high)]

        watchlist.extend(infos[:int(leaders_per_sector)])

    if not watchlist:
        st.warning("Watchlist is empty under current settings. Loosen pullback filter or increase scan sizes.")
        return

    watch_df = pd.DataFrame([{
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

    st.dataframe(
        watch_df.style.format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            "Rotation": "{:.2%}",
            "RSI": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=420
    )

    # -------------------------
    # Watchlist Write-ups
    # -------------------------
    st.write("### 📌 Watchlist Write-ups (click to expand)")
    for rec in watch_df.head(20).to_dict("records"):
        full = analyze_ticker(rec["Ticker"], spy, int(rs_short), int(rs_long), int(ema_trend_len), int(rsi_len))
        if full is None:
            continue
        full["Group"] = rec["Group"]
        with st.expander(f"{full['Group']} — {full['Ticker']} | {full['Meter']} {full['Strength']}/100 | {full['TriggerStatus']}"):
            writeup_block(full, pb_low, pb_high)

    # -------------------------
    # Quick Search
    # -------------------------
    st.subheader("🔎 Quick Ticker Search (Why is this a candidate?)")
    q = st.text_input("Type a ticker:", value="AAPL")
    if q:
        info = analyze_ticker(q.strip().upper(), spy, int(rs_short), int(rs_long), int(ema_trend_len), int(rsi_len))
        if info is None:
            st.warning("No data returned (bad ticker or yfinance empty). Try another symbol.")
        else:
            writeup_block(info, pb_low, pb_high)
