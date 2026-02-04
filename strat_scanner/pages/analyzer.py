import streamlit as st
import plotly.graph_objects as go

from strat_scanner.data import get_hist
from strat_scanner.engine import analyze_ticker, writeup_block


def show_analyzer():
    st.title("🔎 Ticker Analyzer (Deep Dive)")

    c1, c2 = st.columns([1.2, 2.0])
    ticker = c1.text_input("Ticker", value="AAPL").strip().upper()

    rs_short = c2.selectbox("RS short lookback", [21, 30, 42], index=0)
    rs_long  = c2.selectbox("RS long lookback", [63, 90, 126], index=0)
    ema_len  = c2.selectbox("Trend EMA", [50, 100, 200], index=0)
    rsi_len  = c2.selectbox("RSI Length", [7, 14, 21], index=1)

    spy_df = get_hist("SPY")
    if spy_df.empty:
        st.error("SPY data unavailable.")
        return
    spy = spy_df["Close"].dropna()

    info = analyze_ticker(ticker, spy, int(rs_short), int(rs_long), int(ema_len), int(rsi_len))
    if not info:
        st.warning("No data returned. Bad ticker or empty yfinance.")
        return

    # chart
    df = get_hist(ticker)
    if not df.empty and {"Open","High","Low","Close"}.issubset(df.columns):
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name=ticker
        )])
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    writeup_block(info, pb_low=40, pb_high=55)
