from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from strat_scanner.data import get_hist
from strat_scanner.engine import analyze_ticker, writeup_block
from strat_scanner.indicators import rsi_wilder, total_return, rs_vs_spy, trend_label, strength_meter, strength_label

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
    "Energy": ["XOM","CVX","COP","EOG","SLB","HAL","MPC","VLO","OXY","KMI","WMB"],
    "Comm Services": ["GOOGL","META","NFLX","TMUS","VZ","T","DIS","CMCSA","SPOT","ROKU"],
    "Staples": ["PG","KO","PEP","WMT","COST","PM","MO","MDLZ","CL","KMB","GIS","KHC"],
    "Materials": ["LIN","APD","SHW","NUE","DOW","PPG","ECL","FCX","NEM","MLM","VMC"],
    "Industrials": ["CAT","DE","HON","GE","LMT","RTX","BA","UNP","UPS","FDX","ETN"],
    "Real Estate": ["PLD","AMT","EQIX","PSA","O","WELL","DLR","SPG","CCI","VICI"],
    "Discretionary": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","MAR","CMG"],
    "Utilities": ["NEE","DUK","SO","D","AEP","EXC","XEL","SRE","ED","PEG"],
    "Financials": ["BRK-B","JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","SPGI"],
    "Technology": ["AAPL","MSFT","NVDA","AVGO","CRM","ORCL","ADBE","AMD","CSCO","QCOM"],
    "Health Care": ["UNH","JNJ","LLY","PFE","MRK","ABBV","TMO","ABT","DHR","AMGN"],
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
        rs_short = c1.selectbox("RS Lookback (short)", [21, 30, 42], index=0)
        rs_long  = c2.selectbox("RS Lookback (long)", [63, 90, 126], index=0)
        ema_len  = c3.selectbox("Trend EMA", [50, 100, 200], index=0)
        rsi_len  = c4.selectbox("RSI Length", [7, 14, 21], index=1)
        if c5.button("Refresh data"):
            get_hist.cache_clear()  # type: ignore
            st.rerun()

    with st.expander("Watchlist Settings", expanded=True):
        w1, w2, w3, w4 = st.columns([1, 1, 1, 1.3])
        top_groups_in = w1.slider("Top groups IN", 1, 8, 3)
        leaders_per   = w2.slider("Leaders per group", 3, 10, 5)
        pb_low        = w3.slider("Pullback RSI Low", 25, 60, 40)
        pb_high       = w4.slider("Pullback RSI High", 35, 75, 55)

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # Market sentiment
    st.subheader("Overall Market Sentiment")
    syms = list(MARKET_ETFS.values()) + ["^VIX"]
    cols = st.columns(len(syms))

    for i, sym in enumerate(syms):
        d = get_hist(sym)
        if d.empty:
            cols[i].metric(sym, "n/a", "n/a")
            continue
        close = d["Close"].dropna()
        if len(close) < 10:
            cols[i].metric(sym, "n/a", "n/a")
            continue

        tr = trend_label(close, int(ema_len))
        r  = float(rsi_wilder(close, int(rsi_len)).iloc[-1])
        ret = float(total_return(close, int(rs_short)).iloc[-1]) if len(close) > rs_short else np.nan

        cols[i].metric(sym, f"{close.iloc[-1]:.2f}", f"{(ret*100):.1f}%" if np.isfinite(ret) else "n/a")
        cols[i].write(f"Trend: **{tr}**")
        cols[i].write(f"RSI: **{r:.1f}**")

    # SPY anchor
    spy_df = get_hist("SPY")
    if spy_df.empty:
        st.warning("SPY data unavailable; cannot compute RS vs SPY.")
        return
    spy = spy_df["Close"].dropna()
    if len(spy) < (rs_long + 10):
        st.warning("Not enough SPY history for these lookbacks.")
        return

    # Sector rotation
    st.subheader("Sector / Metals Rotation + Strength (vs SPY)")
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
        rot  = rs_s - rs_l
        tr   = trend_label(close, int(ema_len))
        r    = float(rsi_wilder(close, int(rsi_len)).iloc[-1])

        score = int(strength_meter(rs_s, rot, tr))
        sector_rows.append({
            "Group": name,
            "ETF": etf,
            "Strength": score,
            "Meter": strength_label(score),
            f"RS vs SPY ({rs_short})": rs_s,
            f"RS vs SPY ({rs_long})": rs_l,
            "Rotation": rot,
            "Trend": tr,
            "RSI": r,
        })

    sectors = pd.DataFrame(sector_rows)
    if sectors.empty:
        st.warning("Sector data unavailable right now. Try Refresh.")
        return

    sectors = sectors.sort_values(["Strength", "Rotation"], ascending=[False, False])
    st.dataframe(
        sectors.style.format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            f"RS vs SPY ({rs_long})": "{:.2%}",
            "Rotation": "{:.2%}",
            "RSI": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=420
    )

    # Watchlist auto-build
    st.subheader("✅ Today Watchlist (Auto-built from Rotation IN + Leaders)")
    top_groups = sectors.head(int(top_groups_in))["Group"].tolist()

    watch = []
    for g in top_groups:
        for sym in SECTOR_TICKERS.get(g, [])[:25]:
            info = analyze_ticker(sym, spy, int(rs_short), int(rs_long), int(ema_len), int(rsi_len))
            if not info:
                continue
            info["Group"] = g
            watch.append(info)

    if not watch:
        st.warning("No watchlist candidates found.")
        return

    # sort & take top leaders per group
    watch_sorted = sorted(watch, key=lambda x: (x["Strength"], x["Rotation"], x["RS_short"]), reverse=True)

    final = []
    for g in top_groups:
        items = [x for x in watch_sorted if x["Group"] == g]
        final.extend(items[:int(leaders_per)])

    wdf = pd.DataFrame([{
        "Group": x["Group"],
        "Ticker": x["Ticker"],
        "Strength": x["Strength"],
        "Meter": x["Meter"],
        "Trend": x["Trend"],
        "RSI": x["RSI"],
        f"RS vs SPY ({rs_short})": x["RS_short"],
        "Rotation": x["Rotation"],
        "STRAT Prev": x.get("Strat_Prev", "n/a"),
        "STRAT Last": x.get("Strat_Last", "n/a"),
        "Trigger": x["TriggerStatus"],
    } for x in final]).sort_values(["Strength","Rotation"], ascending=[False, False])

    st.dataframe(
        wdf.style.format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            "Rotation": "{:.2%}",
            "RSI": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=420
    )

    st.write("### 📌 Watchlist Write-ups (click to expand)")
    for rec in wdf.head(20).to_dict("records"):
        full = analyze_ticker(rec["Ticker"], spy, int(rs_short), int(rs_long), int(ema_len), int(rsi_len))
        if not full:
            continue
        with st.expander(f"{rec['Group']} — {full['Ticker']} | {full['Meter']} {full['Strength']}/100 | {full['TriggerStatus']}"):
            writeup_block(full, pb_low, pb_high)
