# strat_scanner/pages/scanner.py
# STRAT Scanner page (A-mode full table + drilldown)

from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import streamlit as st

from strat_scanner.data import get_hist
from strat_scanner.engine import analyze_ticker, writeup_block


# -------------------------
# Groups / ETFs / Universe
# -------------------------
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

# Keep this aligned with your dashboard universe (edit as desired)
SECTOR_TICKERS: Dict[str, List[str]] = {
    "Energy": ["XOM","CVX","COP","EOG","SLB","HAL","PSX","MPC","VLO","OXY","KMI","WMB"],
    "Comm Services": ["GOOGL","META","NFLX","TMUS","VZ","T","DIS","CMCSA","CHTR","SPOT"],
    "Staples": ["PG","KO","PEP","WMT","COST","PM","MO","MDLZ","CL","KMB"],
    "Materials": ["LIN","APD","SHW","NUE","DOW","ECL","FCX","NEM","VMC","MOS"],
    "Industrials": ["CAT","DE","HON","GE","LMT","RTX","BA","UNP","UPS","FDX"],
    "Real Estate": ["PLD","AMT","EQIX","PSA","O","WELL","DLR","SPG","CCI","VICI"],
    "Discretionary": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","ORLY"],
    "Utilities": ["NEE","DUK","SO","AEP","EXC","XEL","SRE","ED","PEG","PCG"],
    "Financials": ["BRK-B","JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP"],
    "Technology": ["AAPL","MSFT","NVDA","AVGO","CRM","ORCL","ADBE","AMD","CSCO","QCOM"],
    "Health Care": ["UNH","JNJ","LLY","PFE","MRK","ABBV","TMO","ABT","DHR","BMY"],
    "Metals - Gold": ["GLD"],
    "Metals - Silver": ["SLV"],
    "Metals - Copper": ["CPER"],
    "Metals - Platinum": ["PPLT"],
    "Metals - Palladium": ["PALL"],
}


def show_scanner():
    st.title("🧭 STRAT Scanner (Regime • Rotation • Leaders • STRAT Triggers)")

    # -------------------------
    # Settings
    # -------------------------
    with st.expander("Scanner Settings", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.2])
        with c1:
            rs_short = st.selectbox("RS Lookback (short)", [21, 30, 42], index=0)
        with c2:
            rs_long = st.selectbox("RS Lookback (long)", [63, 90, 126], index=0)
        with c3:
            ema_len = st.selectbox("Trend EMA", [50, 100, 200], index=0)
        with c4:
            rsi_len = st.selectbox("RSI Length", [7, 14, 21], index=1)
        with c5:
            if st.button("Refresh data"):
                st.cache_data.clear()
                st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # -------------------------
    # SPY anchor
    # -------------------------
    spy_df = get_hist("SPY")
    if spy_df is None or spy_df.empty:
        st.error("SPY data unavailable.")
        return

    if "Close" not in spy_df.columns:
        st.error("SPY returned without Close column (yfinance issue). Refresh.")
        return

    spy = spy_df["Close"].dropna()
    if len(spy) < (int(rs_long) + 10):
        st.error("Not enough SPY history for these lookbacks.")
        return

    # -------------------------
    # Sector rotation table (ETF-level)
    # -------------------------
    st.subheader("Sector / Metals Rotation + Strength (Relative Strength vs SPY)")

    sector_rows = []
    for group, etf in SECTOR_ETFS.items():
        info = analyze_ticker(etf, spy, int(rs_short), int(rs_long), int(ema_len), int(rsi_len))
        if not info:
            continue
        info["Group"] = group
        sector_rows.append({
            "Group": group,
            "ETF": etf,
            "Strength": info["Strength"],
            "Meter": info["Meter"],
            "Trend": info["Trend"],
            "RSI": info["RSI"],
            f"RS vs SPY ({rs_short})": info["RS_short"],
            "Rotation": info["Rotation"],
            "Setup": info["Setup"],
            "Status": info["TriggerStatus"],
            "Direction": info["Direction"],
        })

    sectors = pd.DataFrame(sector_rows)
    if sectors.empty:
        st.warning("No sector rows returned. Refresh or check data.")
        return

    sectors = sectors.sort_values(["Strength", "Rotation"], ascending=[False, False])

    st.dataframe(
        sectors.style.format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            "Rotation": "{:.2%}",
            "RSI": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=460,
    )

    # -------------------------
    # Drilldown: pick a group
    # -------------------------
    st.subheader("🔎 Drilldown: Leaders inside a selected group")

    left, right = st.columns([2.2, 1.0])
    with left:
        pick_group = st.selectbox("Pick a group to drill into:", list(SECTOR_TICKERS.keys()), index=2)
    with right:
        top_n = st.slider("How many tickers to show", 5, 20, 12)

    tickers = SECTOR_TICKERS.get(pick_group, [])
    if not tickers:
        st.info("No tickers defined for this group.")
        return

    rows = []
    for sym in tickers:
        info = analyze_ticker(sym, spy, int(rs_short), int(rs_long), int(ema_len), int(rsi_len))
        if not info:
            continue
        info["Group"] = pick_group
        rows.append(info)

    if not rows:
        st.warning("No tickers returned for this group.")
        return

    df = pd.DataFrame([{
        "Group": x["Group"],
        "Ticker": x["Ticker"],
        "Price": x["Price"],
        "Strength": x["Strength"],
        "Meter": x["Meter"],
        "Trend": x["Trend"],
        "RSI": x["RSI"],
        f"RS vs SPY ({rs_short})": x["RS_short"],
        "Rotation": x["Rotation"],
        "Setup": x["Setup"],
        "Status": x["TriggerStatus"],
        "Direction": x["Direction"],
        "Entry": x["Entry"],
        "Stop": x["Stop"],
        "T1": x.get("T1"),
        "T2": x.get("T2"),
    } for x in rows]).sort_values(["Strength", "Rotation"], ascending=[False, False]).head(int(top_n))

    st.dataframe(
        df.style.format({
            "Price": "{:.2f}",
            f"RS vs SPY ({rs_short})": "{:.2%}",
            "Rotation": "{:.2%}",
            "RSI": "{:.1f}",
            "Entry": "{:.2f}",
            "Stop": "{:.2f}",
            "T1": "{:.2f}",
            "T2": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=520,
    )

    # -------------------------
    # Expanders: full writeups
    # -------------------------
    st.write("### 📌 Drilldown Write-ups (click to expand)")
    pb_low, pb_high = 40, 55  # default pullback zone used in writeup block
    for rec in df.to_dict("records"):
        full = analyze_ticker(rec["Ticker"], spy, int(rs_short), int(rs_long), int(ema_len), int(rsi_len))
        if not full:
            continue
        full["Group"] = rec["Group"]
        with st.expander(f"{full['Group']} — {full['Ticker']} | {full['Meter']} {full['Strength']}/100 | {full['Setup']} {full['TriggerStatus']}"):
            writeup_block(full, pb_low, pb_high)
