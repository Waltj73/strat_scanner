import streamlit as st
from strat_scanner.data import get_hist
from strat_scanner.indicators import rsi_wilder, rs_vs_spy, trend_label, strength_meter, strength_label, RS_CAP, ROT_CAP, clamp_float

def show_analyzer():
    st.title("🔎 Ticker Analyzer (Explain the Score)")
    with st.expander("Settings", expanded=True):
        c1, c2, c3, c4 = st.columns([1.2,1,1,1])
        with c1:
            ticker = st.text_input("Ticker", "AAPL").strip().upper()
        with c2:
            rs_short = st.selectbox("RS short", [21,30,42], index=0)
        with c3:
            rs_long = st.selectbox("RS long", [63,90,126], index=0)
        with c4:
            ema_len = st.selectbox("Trend EMA", [50,100,200], index=0)

        rsi_len = st.selectbox("RSI length", [7,14,21], index=1)

        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    spy_df = get_hist("SPY")
    if spy_df.empty:
        st.error("SPY data unavailable.")
        return
    spy_close = spy_df["Close"].dropna()

    d = get_hist(ticker)
    if d.empty:
        st.error("No data returned. Try a different ticker.")
        return
    close = d["Close"].dropna()
    if len(close) < (rs_long + 30):
        st.error("Not enough history for selected lookbacks.")
        return

    tr = trend_label(close, ema_len)
    rsi_v = float(rsi_wilder(close, rsi_len).iloc[-1])

    rs_s = float(rs_vs_spy(close, spy_close, rs_short).iloc[-1])
    rs_l = float(rs_vs_spy(close, spy_close, rs_long).iloc[-1])
    rot = rs_s - rs_l

    rs_s_c = clamp_float(rs_s, -RS_CAP, RS_CAP)
    rot_c  = clamp_float(rot, -ROT_CAP, ROT_CAP)

    strength = strength_meter(rs_s_c, rot_c, tr)
    meter = strength_label(strength)

    st.subheader(f"{ticker} — {meter} ({strength}/100)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trend", tr)
    c2.metric("RSI", f"{rsi_v:.1f}")
    c3.metric(f"RS vs SPY ({rs_short})", f"{rs_s*100:.1f}%")
    c4.metric("Rotation", f"{rot*100:.1f}%")

    st.markdown("### Why it scored this way")
    st.write(f"- Trend is **{tr}**")
    st.write(f"- RSI({rsi_len}) = **{rsi_v:.1f}**")
    st.write(f"- RS short capped: **{rs_s_c*100:.1f}%** (cap ±{RS_CAP*100:.0f}%)")
    st.write(f"- Rotation capped: **{rot_c*100:.1f}%** (cap ±{ROT_CAP*100:.0f}%)")
    st.write(f"- Strength meter: **{strength}/100**")
