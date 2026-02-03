# strat_scanner/pages/scanner.py
# STRAT Scanner page (sector ranking + drilldown into leaders)

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import streamlit as st

from strat_scanner.data import get_hist
from strat_scanner.indicators import (
    rs_vs_spy,
    rsi_wilder,
    trend_label,
    strength_meter,
    strength_label,
)

from strat_scanner.engine import analyze_ticker

# writeup_block is optional (dashboard uses it). If your engine.py doesn't have it,
# this file will still run without breaking.
try:
    from strat_scanner.engine import writeup_block  # type: ignore
except Exception:
    writeup_block = None  # type: ignore


# -------------------------
# Sector ETFs (Rotation IN/OUT)
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

# -------------------------
# Drilldown universe per group
# (You can expand these later; keep it stable for now)
# -------------------------
SECTOR_TICKERS: Dict[str, List[str]] = {
    "Energy": ["XOM","CVX","COP","EOG","SLB","HAL","PSX","MPC","VLO","OXY","KMI","WMB"],
    "Comm Services": ["GOOGL","META","NFLX","TMUS","VZ","T","DIS","CMCSA"],
    "Staples": ["PG","KO","PEP","WMT","COST","PM","MDLZ","CL"],
    "Materials": ["LIN","APD","SHW","NUE","DOW","ECL","FCX","NEM"],
    "Industrials": ["CAT","DE","HON","GE","LMT","RTX","BA","UNP","UPS","FDX"],
    "Real Estate": ["PLD","AMT","EQIX","PSA","O","WELL","DLR","SPG"],
    "Discretionary": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG"],
    "Utilities": ["NEE","DUK","SO","AEP","EXC","XEL","SRE","ED"],
    "Financials": ["BRK-B","JPM","BAC","WFC","GS","MS","C","BLK","SCHW"],
    "Technology": ["AAPL","MSFT","NVDA","AVGO","CRM","ORCL","ADBE","AMD","CSCO","QCOM"],
    "Health Care": ["UNH","JNJ","LLY","PFE","MRK","ABBV","TMO","ABT","DHR"],
    "Metals - Gold": ["GLD"],
    "Metals - Silver": ["SLV"],
    "Metals - Copper": ["CPER"],
    "Metals - Platinum": ["PPLT"],
    "Metals - Palladium": ["PALL"],
}


def show_scanner():
    st.title("📌 STRAT Scanner (Rotation • Leaders • Drilldown)")

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

    spy = spy_df["Close"].dropna()
    if len(spy) < (int(rs_long) + 10):
        st.error("Not enough SPY history for these lookbacks.")
        return

    # -------------------------
    # Build sector rotation table
    # -------------------------
    rows = []
    for group, etf in SECTOR_ETFS.items():
        df = get_hist(etf)
        if df is None or df.empty:
            continue

        close = df["Close"].dropna()
        if len(close) < (int(rs_long) + 10):
            continue

        rs_s = float(rs_vs_spy(close, spy, int(rs_short)).iloc[-1])
        rs_l = float(rs_vs_spy(close, spy, int(rs_long)).iloc[-1])
        rot = rs_s - rs_l

        tr = trend_label(close, int(ema_len))
        rsi_val = float(rsi_wilder(close, int(rsi_len)).iloc[-1])

        score = int(strength_meter(rs_s, rot, tr))

        rows.append({
            "Group": group,
            "ETF": etf,
            "Strength": score,
            "Meter": strength_label(score),
            f"RS vs SPY ({rs_short})": rs_s,
            f"RS vs SPY ({rs_long})": rs_l,
            "Rotation (short-long)": rot,
            "Trend": tr,
            "RSI": rsi_val,
        })

    sector_df = pd.DataFrame(rows)
    if sector_df.empty:
        st.warning("No sector rows returned. Try Refresh.")
        return

    sector_df = sector_df.sort_values(["Strength", "Rotation (short-long)"], ascending=[False, False])

    st.subheader("Sector / Metals Rotation + Strength (vs SPY)")
    st.dataframe(
        sector_df.style.format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            f"RS vs SPY ({rs_long})": "{:.2%}",
            "Rotation (short-long)": "{:.2%}",
            "RSI": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=520,
    )

    # -------------------------
    # Drilldown controls
    # -------------------------
    st.subheader("🔍 Drilldown: Leaders inside a selected group")

    left, right = st.columns([1.2, 1])
    with left:
        default_group = sector_df.iloc[0]["Group"]
        group_pick = st.selectbox("Pick a group to drill into:", sector_df["Group"].tolist(), index=0)
    with right:
        leaders_n = st.slider("How many tickers to show", 5, 25, 12)

    tickers = SECTOR_TICKERS.get(group_pick, [])
    if not tickers:
        st.info("No tickers configured for this group yet.")
        return

    # -------------------------
    # Analyze tickers
    # -------------------------
    results = []
    for sym in tickers:
        info = analyze_ticker(sym, spy, int(rs_short), int(rs_long), int(ema_len), int(rsi_len))
        if info is None:
            continue
        info["Group"] = group_pick
        results.append(info)

    if not results:
        st.warning("No ticker results returned (data provider may be empty).")
        return

    # sort by strength, rotation, rs_short (descending)
    results = sorted(results, key=lambda x: (x.get("Strength", 0), x.get("Rotation", 0), x.get("RS_short", 0)), reverse=True)
    results = results[: int(leaders_n)]

    out = pd.DataFrame([{
        "Group": r.get("Group"),
        "Ticker": r.get("Ticker"),
        "Strength": r.get("Strength"),
        "Meter": r.get("Meter"),
        "Trend": r.get("Trend"),
        "RSI": r.get("RSI"),
        f"RS vs SPY ({rs_short})": r.get("RS_short"),
        f"RS vs SPY ({rs_long})": r.get("RS_long"),
        "Rotation": r.get("Rotation"),
        "Trigger": r.get("TriggerStatus"),
        "TF": r.get("TF"),
        "Entry": r.get("Entry"),
        "Stop": r.get("Stop"),
    } for r in results])

    st.write(f"**Group:** {group_pick}  •  **Top {len(out)} tickers**")
    st.dataframe(
        out.style.format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            f"RS vs SPY ({rs_long})": "{:.2%}",
            "Rotation": "{:.2%}",
            "RSI": "{:.1f}",
            "Entry": "{:.2f}",
            "Stop": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=460
    )

    # -------------------------
    # Optional Write-ups
    # -------------------------
    st.write("### 🧾 Ticker Write-ups (click to expand)")
    for r in results:
        label = f"{r.get('Group')} — {r.get('Ticker')} | {r.get('Meter')} {r.get('Strength')}/100 | {r.get('TriggerStatus')}"
        with st.expander(label):
            if writeup_block is None:
                # fallback if you don't have writeup_block in engine.py
                st.write(r)
            else:
                # simple defaults; dashboard has sliders for these
                writeup_block(r, pb_low=40, pb_high=55)
