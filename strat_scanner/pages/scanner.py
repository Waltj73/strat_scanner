from datetime import datetime, timezone
import pandas as pd
import streamlit as st

from strat_scanner.data import get_hist
from strat_scanner.engine import analyze_ticker
from strat_scanner.indicators import rs_vs_spy, rsi_wilder, trend_label, strength_meter, strength_label

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

def show_scanner():
    st.title("📌 STRAT Scanner")

    with st.expander("Scanner Settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        rs_short = c1.selectbox("RS Lookback (short)", [21, 30, 42], index=0)
        rs_long  = c2.selectbox("RS Lookback (long)", [63, 90, 126], index=0)
        ema_len  = c3.selectbox("Trend EMA", [50, 100, 200], index=0)
        rsi_len  = c4.selectbox("RSI Length", [7, 14, 21], index=1)

        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    spy_df = get_hist("SPY")
    if spy_df is None or spy_df.empty:
        st.error("SPY data unavailable.")
        return

    spy = spy_df["Close"].dropna()
    if len(spy) < (rs_long + 10):
        st.error("Not enough SPY history for these lookbacks.")
        return

    rows = []
    for name, etf in SECTOR_ETFS.items():
        df = get_hist(etf)
        if df is None or df.empty:
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
            "Rotation": rot,
            "Trend": tr,
            "RSI": r,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        st.warning("No rows returned. Try Refresh.")
        return

    out = out.sort_values(["Strength", "Rotation"], ascending=[False, False])
    st.dataframe(
        out.style.format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            f"RS vs SPY ({rs_long})": "{:.2%}",
            "Rotation": "{:.2%}",
            "RSI": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=520,
    )
