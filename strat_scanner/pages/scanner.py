from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import streamlit as st

from strat_scanner.data import get_hist
from strat_scanner.engine import analyze_ticker, writeup_block
from strat_scanner.indicators import rs_vs_spy, rsi_wilder, trend_label, strength_meter, strength_label

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
    "Energy": ["XOM","CVX","COP","EOG","SLB","HAL","PSX","MPC","VLO","OXY","KMI","WMB","BKR","DVN"],
    "Comm Services": ["GOOGL","GOOG","META","NFLX","TMUS","VZ","T","DIS","CMCSA","CHTR","SPOT","ROKU"],
    "Staples": ["PG","KO","PEP","WMT","COST","PM","MO","MDLZ","CL","KMB","GIS","KHC","HSY"],
    "Materials": ["LIN","APD","SHW","NUE","DOW","PPG","ECL","FCX","NEM","MLM","VMC","ALB","MOS"],
    "Industrials": ["CAT","DE","HON","GE","LMT","RTX","BA","UNP","UPS","FDX","ETN","EMR","CSX","NSC"],
    "Real Estate": ["PLD","AMT","EQIX","PSA","O","WELL","DLR","SPG","CCI","VICI","AVB","EQR","IRM"],
    "Discretionary": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","MAR","ROST","ORLY","CMG"],
    "Utilities": ["NEE","DUK","SO","D","AEP","EXC","XEL","SRE","ED","PEG","EIX","PCG","WEC"],
    "Financials": ["BRK-B","JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","SPGI","ICE","CME"],
    "Technology": ["AAPL","MSFT","NVDA","AVGO","CRM","ORCL","ADBE","AMD","CSCO","INTC","QCOM","TXN","NOW"],
    "Health Care": ["UNH","JNJ","LLY","PFE","MRK","ABBV","TMO","ABT","DHR","BMY","AMGN","GILD","ISRG"],
    "Metals - Gold": ["GLD"],
    "Metals - Silver": ["SLV"],
    "Metals - Copper": ["CPER"],
    "Metals - Platinum": ["PPLT"],
    "Metals - Palladium": ["PALL"],
}


def show_scanner():
    st.title("📌 STRAT Scanner (Rotation • Leaders • Drilldown)")

    with st.expander("Scanner Settings", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.2])
        rs_short = c1.selectbox("RS Lookback (short)", [21, 30, 42], index=0)
        rs_long  = c2.selectbox("RS Lookback (long)", [63, 90, 126], index=0)
        ema_len  = c3.selectbox("Trend EMA", [50, 100, 200], index=0)
        rsi_len  = c4.selectbox("RSI Length", [7, 14, 21], index=1)
        if c5.button("Refresh data"):
            # bust yfinance cache
            get_hist.cache_clear()  # type: ignore
            st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    spy_df = get_hist("SPY")
    if spy_df.empty:
        st.error("SPY data unavailable.")
        return
    spy = spy_df["Close"].dropna()
    if len(spy) < (rs_long + 10):
        st.error("Not enough SPY history for these lookbacks.")
        return

    # --- rotation table
    st.subheader("Sector / Metals Rotation + Strength (vs SPY)")
    rows = []
    for name, etf in SECTOR_ETFS.items():
        df = get_hist(etf)
        if df.empty:
            continue
        close = df["Close"].dropna()
        if len(close) < (rs_long + 10):
            continue

        rs_s = float(rs_vs_spy(close, spy, int(rs_short)).iloc[-1])
        rs_l = float(rs_vs_spy(close, spy, int(rs_long)).iloc[-1])
        rot  = rs_s - rs_l

        tr = trend_label(close, int(ema_len))
        r  = float(rsi_wilder(close, int(rsi_len)).iloc[-1])

        score = int(strength_meter(rs_s, rot, tr))
        rows.append({
            "Group": name,
            "ETF": etf,
            "Strength": score,
            "Meter": strength_label(score),
            f"RS vs SPY ({rs_short})": rs_s,
            f"RS vs SPY ({rs_long})": rs_l,
            "Rotation (short-long)": rot,
            "Trend": tr,
            "RSI": r,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        st.warning("No sector rows returned. Try Refresh.")
        return

    out = out.sort_values(["Strength", "Rotation (short-long)"], ascending=[False, False])
    st.dataframe(
        out.style.format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            f"RS vs SPY ({rs_long})": "{:.2%}",
            "Rotation (short-long)": "{:.2%}",
            "RSI": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    # --- drilldown leaders
    st.write("### 🔎 Drilldown: Leaders inside a selected group")
    left, right = st.columns([2.2, 1.2])
    group = left.selectbox("Pick a group to drill into:", list(SECTOR_TICKERS.keys()), index=2)
    n_show = right.slider("How many tickers to show", 5, 20, 12)

    tickers = SECTOR_TICKERS.get(group, [])
    if not tickers:
        st.info("No tickers configured for this group.")
        return

    infos = []
    for sym in tickers:
        info = analyze_ticker(sym, spy, int(rs_short), int(rs_long), int(ema_len), int(rsi_len))
        if info:
            info["Group"] = group
            infos.append(info)

    if not infos:
        st.warning("No ticker data returned. Try Refresh or change lookbacks.")
        return

    leaders = pd.DataFrame([{
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
    } for x in infos]).sort_values(["Strength", "Rotation"], ascending=[False, False]).head(int(n_show))

    st.dataframe(
        leaders.style.format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            "Rotation": "{:.2%}",
            "RSI": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=420
    )

    st.write("### 📌 Leader Write-ups (click to expand)")
    for rec in leaders.to_dict("records"):
        full = analyze_ticker(rec["Ticker"], spy, int(rs_short), int(rs_long), int(ema_len), int(rsi_len))
        if not full:
            continue
        with st.expander(f"{group} — {full['Ticker']} | {full['Meter']} {full['Strength']}/100 | {full['TriggerStatus']}"):
            writeup_block(full, pb_low=40, pb_high=55)
