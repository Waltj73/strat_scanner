import streamlit as st
from strat_scanner.data import get_hist
from strat_scanner.engine import analyze_ticker
from strat_scanner.ui import writeup_block


def show_analyzer():
    st.title("🔎 Ticker Analyzer")

    spy_df = get_hist("SPY")
    if spy_df is None or spy_df.empty:
        st.error("SPY data unavailable.")
        return
    spy = spy_df["Close"].dropna()

    c1, c2, c3, c4 = st.columns(4)
    rs_short = c1.selectbox("RS short", [21, 30, 42], index=0)
    rs_long = c2.selectbox("RS long", [63, 90, 126], index=0)
    ema_len = c3.selectbox("Trend EMA", [50, 100, 200], index=0)
    rsi_len = c4.selectbox("RSI length", [7, 14, 21], index=1)

    pb_low, pb_high = st.slider("Pullback RSI zone (for UP trend)", 20, 80, (40, 55))

    t = st.text_input("Ticker", value="AAPL").strip().upper()
    if not t:
        return

    info = analyze_ticker(t, spy, int(rs_short), int(rs_long), int(ema_len), int(rsi_len))
    if not info:
        st.warning("No data returned (bad ticker or empty yfinance).")
        return

    writeup_block(info, float(pb_low), float(pb_high))
