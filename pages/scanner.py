from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import streamlit as st

from strat_scanner.data import get_hist
from strat_scanner.strat import (
    tf_frames, compute_flags, score_regime, market_bias_and_strength,
    alignment_ok, best_trigger
)
from strat_scanner.engine import magnitude_metrics, calc_scores

MARKET_ETFS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Russell 2000": "IWM",
    "Dow Jones": "DIA",
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
    "Metals - Gold": "GLD",
    "Metals - Silver": "SLV",
    "Metals - Copper": "CPER",
    "Metals - Platinum": "PPLT",
    "Metals - Palladium": "PALL",
}

SECTOR_TICKERS = {
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

def show_scanner():
    st.title("📡 STRAT Regime Scanner")
    st.caption("Market bias → sector ranking → drilldown candidates → magnitude (RR/ATR/compression).")

    with st.expander("Filters", expanded=True):
        a, b, c, d = st.columns([1.2, 1.2, 1.6, 1.0])
        with a:
            only_inside = st.checkbox("ONLY Inside Bars (D or W)", value=False)
        with b:
            only_212 = st.checkbox("ONLY 2-1-2 (bias direction)", value=False)
        with c:
            require_align = st.checkbox("Require Monthly or Weekly alignment", value=True)
        with d:
            top_k = st.slider("Top Picks", 3, 10, 5)

        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # market regime
    market_rows: List[Dict] = []
    for name, etf in MARKET_ETFS.items():
        ddf = get_hist(etf)
        if ddf.empty:
            continue
        d_tf, w_tf, m_tf = tf_frames(ddf)
        flags = compute_flags(d_tf, w_tf, m_tf)
        bull, bear = score_regime(flags)
        row = {"Market": name, "ETF": etf, "BullScore": bull, "BearScore": bear}
        row.update(flags)
        market_rows.append(row)

    if not market_rows:
        st.error("No market data returned. yfinance may be throttling. Hit Refresh or wait a minute.")
        return

    bias, strength, diff = market_bias_and_strength(market_rows)

    st.subheader("Market Regime")
    st.dataframe(pd.DataFrame(market_rows)[["Market","ETF","BullScore","BearScore","M_Bull","W_Bull","D_Bull","M_Bear","W_Bear","D_Bear"]], use_container_width=True, hide_index=True)

    st.info(f"Bias: **{bias}** | Strength: **{strength}/100** | Bull–Bear diff: **{diff}**")

    # sector regime scoring (not RS rotation here — that’s dashboard)
    sector_rows: List[Dict] = []
    for sector, etf in SECTOR_ETFS.items():
        ddf = get_hist(etf)
        if ddf.empty:
            continue
        d_tf, w_tf, m_tf = tf_frames(ddf)
        flags = compute_flags(d_tf, w_tf, m_tf)
        bull, bear = score_regime(flags)
        row = {"Sector": sector, "ETF": etf, "BullScore": bull, "BearScore": bear}
        row.update(flags)
        sector_rows.append(row)

    sectors_df = pd.DataFrame(sector_rows)
    if sectors_df.empty:
        st.error("No sector ETF data returned.")
        return

    if bias == "LONG":
        sectors_df = sectors_df.sort_values(["BullScore","BearScore"], ascending=[False, True])
    elif bias == "SHORT":
        sectors_df = sectors_df.sort_values(["BearScore","BullScore"], ascending=[False, True])
    else:
        sectors_df["Dominance"] = (sectors_df["BullScore"] - sectors_df["BearScore"]).abs()
        sectors_df = sectors_df.sort_values("Dominance", ascending=False)

    st.subheader("Sectors / Metals ranked after bias is known")
    st.dataframe(sectors_df[["Sector","ETF","BullScore","BearScore","M_Bull","W_Bull","D_Bull","M_Bear","W_Bear","D_Bear","W_Inside","D_Inside","W_212Up","D_212Up","W_212Dn","D_212Dn"]],
                 use_container_width=True, hide_index=True)

    st.subheader("Drilldown")
    sector_choice = st.selectbox("Choose group:", list(SECTOR_TICKERS.keys()), index=0)
    tickers = SECTOR_TICKERS.get(sector_choice, [])
    scan_n = st.slider("How many tickers to scan", 1, max(1, len(tickers)), min(15, len(tickers)))
    scan_list = tickers[:scan_n]

    cand_rows: List[Dict] = []
    for t in scan_list:
        ddf = get_hist(t)
        if ddf.empty:
            continue

        d_tf, w_tf, m_tf = tf_frames(ddf)
        flags = compute_flags(d_tf, w_tf, m_tf)

        eff_bias = bias if bias in ("LONG","SHORT") else "LONG"

        if require_align and eff_bias in ("LONG","SHORT") and not alignment_ok(eff_bias, flags):
            continue

        if only_inside and not (flags["D_Inside"] or flags["W_Inside"]):
            continue

        if only_212:
            if eff_bias == "LONG" and not (flags["D_212Up"] or flags["W_212Up"]):
                continue
            if eff_bias == "SHORT" and not (flags["D_212Dn"] or flags["W_212Dn"]):
                continue

        tf, entry, stop = best_trigger(eff_bias, d_tf, w_tf)
        rr, atrp, room, compression = magnitude_metrics(eff_bias, d_tf, entry, stop)
        setup_score, mag_score, total_score = calc_scores(eff_bias, flags, rr, atrp, compression, entry, stop)

        trigger_status = "READY" if (flags["W_Inside"] or flags["D_Inside"]) else "WAIT"

        cand_rows.append({
            "Ticker": t,
            "Trigger": trigger_status,
            "Total": total_score,
            "Setup": setup_score,
            "Mag": mag_score,
            "TF": tf,
            "Entry": None if entry is None else round(float(entry), 2),
            "Stop": None if stop is None else round(float(stop), 2),
            "RR": None if rr is None else round(float(rr), 2),
            "ATR%": None if atrp is None else round(float(atrp), 2),
        })

    if not cand_rows:
        st.warning("No matches under current filters. Loosen filters or try another group.")
        return

    df = pd.DataFrame(cand_rows).sort_values("Total", ascending=False)

    st.subheader(f"Top Ideas — Bias: {bias}")
    st.dataframe(df.head(top_k), use_container_width=True, hide_index=True)

    st.subheader("All Matches")
    st.dataframe(df, use_container_width=True, hide_index=True)
