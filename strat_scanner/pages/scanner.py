# strat_scanner/pages/scanner.py
# STRAT Scanner page (UI only)

from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import streamlit as st

from strat_scanner.data import get_hist
from strat_scanner.indicators import rsi_wilder, rs_vs_spy, trend_label, strength_label, strength_meter
from strat_scanner.engine import analyze_ticker

# -------------------------
# Universe
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

# Example tickers per sector (you can customize later)
SECTOR_TICKERS: Dict[str, List[str]] = {
    "Energy": ["XOM","CVX","COP","EOG","SLB","HAL","PSX","MPC","VLO","OXY"],
    "Comm Services": ["GOOGL","META","NFLX","TMUS","VZ","T","DIS","CMCSA","CHTR","SPOT"],
    "Staples": ["PG","KO","PEP","WMT","COST","PM","MO","MDLZ","CL","KMB"],
    "Materials": ["LIN","APD","SHW","NUE","DOW","ECL","FCX","NEM","VMC","ALB"],
    "Industrials": ["CAT","DE","HON","GE","LMT","RTX","BA","UNP","UPS","ETN"],
    "Real Estate": ["PLD","AMT","EQIX","PSA","O","WELL","DLR","SPG","CCI","VICI"],
    "Discretionary": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","CMG"],
    "Utilities": ["NEE","DUK","SO","AEP","EXC","XEL","SRE","ED","PEG","AWK"],
    "Financials": ["BRK-B","JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP"],
    "Technology": ["AAPL","MSFT","NVDA","AVGO","CRM","ORCL","ADBE","AMD","CSCO","QCOM"],
    "Health Care": ["UNH","JNJ","LLY","PFE","MRK","ABBV","TMO","ABT","DHR","AMGN"],
    "Metals - Gold": ["GLD"],
    "Metals - Silver": ["SLV"],
    "Metals - Copper": ["CPER"],
    "Metals - Platinum": ["PPLT"],
    "Metals - Palladium": ["PALL"],
}


def show_scanner():
    st.title("🧭 STRAT Scanner")

    with st.expander("Scanner Settings", expanded=True):
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

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # ---- SPY anchor
    spy_df = get_hist("SPY")
    if spy_df is None or spy_df.empty:
        st.error("SPY data unavailable. Cannot compute RS vs SPY.")
        return

    spy = spy_df["Close"].dropna()
    if len(spy) < (rs_long + 10):
        st.error("Not enough SPY history for these lookbacks.")
        return

    # ---- Sector rotation
    st.subheader("Sector / Metals Rotation + Strength (Relative Strength vs SPY)")

    rows = []
    for group, etf in SECTOR_ETFS.items():
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
        rsi_val = float(rsi_wilder(close, int(rsi_len)).iloc[-1])

        score = int(strength_meter(rs_s, rot, tr))
        rows.append({
            "Group": group,
            "ETF": etf,
            "Strength": score,
            "Meter": strength_label(score),
            f"RS vs SPY ({rs_short})": rs_s,
            f"RS vs SPY ({rs_long})": rs_l,
            "Rotation (RS short - RS long)": rot,
            "Trend": tr,
            "RSI": rsi_val,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No sector rows returned. Try Refresh.")
        return

    df = df.sort_values(["Strength", "Rotation (RS short - RS long)"], ascending=[False, False])

    st.dataframe(
        df.style.format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            f"RS vs SPY ({rs_long})": "{:.2%}",
            "Rotation (RS short - RS long)": "{:.2%}",
            "RSI": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    # ---- Drilldown
    st.subheader("🔍 Drilldown: pick a group and see candidates")
    groups = list(df["Group"].unique())
    pick = st.selectbox("Select a group:", groups, index=0)

    tickers = SECTOR_TICKERS.get(pick, [])
    if not tickers:
        st.info("No tickers configured for this group.")
        return

    cand = []
    for sym in tickers:
        info = analyze_ticker(sym, spy, int(rs_short), int(rs_long), int(ema_trend_len), int(rsi_len))
        if info is None:
            continue
        info["Group"] = pick
        cand.append(info)

    if not cand:
        st.warning("No candidates returned for this group.")
        return

    out = pd.DataFrame([{
        "Ticker": x["Ticker"],
        "Strength": x["Strength"],
        "Meter": x["Meter"],
        "Trend": x["Trend"],
        "RSI": x["RSI"],
        f"RS vs SPY ({rs_short})": x["RS_short"],
        "Rotation": x["Rotation"],
        "Trigger": x.get("TriggerStatus", "n/a"),
        "TF": x.get("TF", "n/a"),
        "Entry": x.get("Entry", None),
        "Stop": x.get("Stop", None),
    } for x in cand]).sort_values(["Strength","Rotation"], ascending=[False, False])

    st.dataframe(
        out.style.format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            "Rotation": "{:.2%}",
            "RSI": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=420,
    )
