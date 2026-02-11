from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go

from data.fetch import get_hist
from indicators.rs_rsi import rsi_wilder, rs_vs_spy, trend_label, strength_score, strength_trend

def analyzer_main():
    st.title("🔎 Ticker Analyzer (RSI/RS-only)")

    with st.expander("Settings", expanded=True):
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1:
            ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        with c2:
            rs_short = st.selectbox("RS short", [21,30,42], index=0)
        with c3:
            rs_long = st.selectbox("RS long", [63,90,126], index=0)
        with c4:
            ema_len = st.selectbox("Trend EMA", [50,100,200], index=0)
        with c5:
            rsi_len = st.selectbox("RSI length", [7,14,21], index=1)

        lookback_bars = st.selectbox("StrengthTrend lookback (bars)", [5,10], index=0)

    spy = get_hist("SPY")
    if spy.empty:
        st.error("SPY data unavailable.")
        return
    spy_close = spy["Close"].dropna()

    df = get_hist(ticker)
    if df.empty:
        st.error("No data for that ticker.")
        return

    close = df["Close"].dropna()
    rsi = float(rsi_wilder(close, int(rsi_len)).iloc[-1])
    tr  = trend_label(close, int(ema_len))
    rs_s = float(rs_vs_spy(close, spy_close, int(rs_short)))
    rs_l = float(rs_vs_spy(close, spy_close, int(rs_long)))
    rot  = rs_s - rs_l

    score = strength_score(rs_s, rs_l, rsi, tr)
    s_trend = strength_trend(close, spy_close, int(rs_short), int(rs_long), int(ema_len), int(rsi_len), int(lookback_bars))

    st.subheader(f"{ticker} — Strength {score}/100")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Trend", tr)
    c2.metric(f"RS({rs_short})", f"{rs_s*100:.1f}%")
    c3.metric("Rotation", f"{rot*100:.1f}%")
    c4.metric(f"RSI({rsi_len})", f"{rsi:.1f}")
    c5.metric("StrengthTrend", f"{s_trend:+.0f}")

    st.markdown("### Chart")
    bars = df.tail(220)
    fig = go.Figure(data=[go.Candlestick(
        x=bars.index, open=bars["Open"], high=bars["High"], low=bars["Low"], close=bars["Close"], name=ticker
    )])
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, title=f"{ticker} — Daily Candles")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### How to read this")
    st.write(
        "- **Trend**: price above EMA and EMA rising.\n"
        "- **RS(short/long)**: returns vs SPY over those windows.\n"
        "- **Rotation**: RS(short) − RS(long). Positive = improving leadership.\n"
        "- **StrengthTrend**: Strength(today) minus Strength(~1 week ago). Positive = strengthening."
    )
