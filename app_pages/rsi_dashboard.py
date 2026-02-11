from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config.universe import MARKET_ETFS, SECTOR_ETFS, SECTOR_TICKERS
from data.fetch import get_hist
from indicators.rs_rsi import rsi_wilder, rs_vs_spy, trend_label, strength_score

def _meter(score: int) -> str:
    if score >= 70: return "STRONG"
    if score >= 45: return "NEUTRAL"
    return "WEAK"

def _rotation_in_out(df: pd.DataFrame, n: int, bias: str):
    # bias LONG: IN = highest rotation, OUT = lowest rotation
    # bias SHORT: IN = lowest rotation (most negative), OUT = highest
    if df.empty:
        return df, df
    if bias == "SHORT":
        rot_in = df.sort_values("Rotation", ascending=True).head(n)
        rot_out = df.sort_values("Rotation", ascending=False).head(n)
    else:
        rot_in = df.sort_values("Rotation", ascending=False).head(n)
        rot_out = df.sort_values("Rotation", ascending=True).head(n)
    return rot_in, rot_out

def rsi_dashboard_main():
    st.title("📊 RSI/RS Rotation Dashboard (RSI/RS-only)")

    top = st.columns([1,1,6])
    with top[0]:
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    with st.expander("Settings", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            rs_short = st.selectbox("RS short lookback", [21,30,42], index=0)
        with c2:
            rs_long = st.selectbox("RS long lookback", [63,90,126], index=0)
        with c3:
            ema_len = st.selectbox("Trend EMA", [50,100,200], index=0)
        with c4:
            rsi_len = st.selectbox("RSI length", [7,14,21], index=1)

        c5,c6,c7 = st.columns(3)
        with c5:
            top_sectors = st.slider("Top sectors to show", 3, 11, 6)
        with c6:
            leaders_per_sector = st.slider("Leaders per sector", 3, 10, 5)
        with c7:
            rotation_n = st.slider("Rotation IN/OUT count", 3, 8, 5)

    spy_df = get_hist("SPY")
    if spy_df.empty:
        st.error("SPY data unavailable.")
        return
    spy_close = spy_df["Close"].dropna()

    # ======================
    # Market Outlook
    # ======================
    st.subheader("Market Outlook (RSI + RS vs SPY + Strength)")

    market_rows = []
    for name, sym in MARKET_ETFS.items():
        d = get_hist(sym)
        if d.empty:
            continue
        close = d["Close"].dropna()
        if close.empty or len(close) < (rs_long + 10):
            continue

        rsi = float(rsi_wilder(close, int(rsi_len)).iloc[-1])
        tr  = trend_label(close, int(ema_len))

        rs_s = float(rs_vs_spy(close, spy_close, int(rs_short)))
        rs_l = float(rs_vs_spy(close, spy_close, int(rs_long)))
        rot  = rs_s - rs_l

        score = strength_score(rs_s, rs_l, rsi, tr)
        market_rows.append({
            "Name": name,
            "Ticker": sym,
            "Trend": tr,
            "RSI": rsi,
            f"RS({rs_short})": rs_s,
            f"RS({rs_long})": rs_l,
            "Rotation": rot,
            "Strength": score,
            "Meter": _meter(score),
        })

    mdf = pd.DataFrame(market_rows)
    if mdf.empty:
        st.warning("No market data returned.")
        return

    st.dataframe(
        mdf.style.format({
            "RSI": "{:.1f}",
            f"RS({rs_short})": "{:.2%}",
            f"RS({rs_long})": "{:.2%}",
            "Rotation": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True
    )

    # Simple overall bias: based on SPY+QQQ+IWM+DIA strength average
    core = mdf[mdf["Ticker"].isin(["SPY","QQQ","IWM","DIA"])].copy()
    core_strength = float(core["Strength"].mean()) if not core.empty else 50.0
    bias = "LONG" if core_strength >= 55 else "SHORT" if core_strength <= 45 else "MIXED"

    if bias == "LONG":
        st.success(f"Overall Bias: **LONG** 🟢 | Avg Strength: **{core_strength:.1f}/100**")
    elif bias == "SHORT":
        st.error(f"Overall Bias: **SHORT** 🔴 | Avg Strength: **{core_strength:.1f}/100**")
    else:
        st.warning(f"Overall Bias: **MIXED** 🟠 | Avg Strength: **{core_strength:.1f}/100**")

    # ======================
    # Sector Rotation
    # ======================
    st.subheader("Sector Rotation (RS vs SPY + Rotation + Strength)")

    sector_rows = []
    for sector, etf in SECTOR_ETFS.items():
        d = get_hist(etf)
        if d.empty:
            continue
        close = d["Close"].dropna()
        if close.empty or len(close) < (rs_long + 10):
            continue

        rsi = float(rsi_wilder(close, int(rsi_len)).iloc[-1])
        tr  = trend_label(close, int(ema_len))
        rs_s = float(rs_vs_spy(close, spy_close, int(rs_short)))
        rs_l = float(rs_vs_spy(close, spy_close, int(rs_long)))
        rot  = rs_s - rs_l
        score = strength_score(rs_s, rs_l, rsi, tr)

        sector_rows.append({
            "Sector": sector,
            "ETF": etf,
            "Trend": tr,
            "RSI": rsi,
            f"RS({rs_short})": rs_s,
            f"RS({rs_long})": rs_l,
            "Rotation": rot,
            "Strength": score,
            "Meter": _meter(score),
        })

    sdf = pd.DataFrame(sector_rows)
    if sdf.empty:
        st.warning("No sector data returned.")
        return

    sdf = sdf.sort_values(["Strength","Rotation"], ascending=[False, False])

    st.dataframe(
        sdf.style.format({
            "RSI": "{:.1f}",
            f"RS({rs_short})": "{:.2%}",
            f"RS({rs_long})": "{:.2%}",
            "Rotation": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
        height=420
    )

    rot_in, rot_out = _rotation_in_out(sdf, int(rotation_n), bias)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔁 Rotation IN")
        for _, r in rot_in.iterrows():
            st.write(f"✅ **{r['Sector']}** ({r['ETF']}) — Rot {r['Rotation']:.2%} | Strength {int(r['Strength'])}")
    with c2:
        st.markdown("### 🔁 Rotation OUT")
        for _, r in rot_out.iterrows():
            st.write(f"❌ **{r['Sector']}** ({r['ETF']}) — Rot {r['Rotation']:.2%} | Strength {int(r['Strength'])}")

    # ======================
    # Leaders
    # ======================
    st.subheader("✅ Leaders (top tickers from top sectors)")
    top_groups = sdf.head(int(top_sectors))["Sector"].tolist()

    leaders = []
    for sector in top_groups:
        names = SECTOR_TICKERS.get(sector, [])
        for t in names[:30]:
            d = get_hist(t)
            if d.empty:
                continue
            close = d["Close"].dropna()
            if close.empty or len(close) < (rs_long + 10):
                continue
            rsi = float(rsi_wilder(close, int(rsi_len)).iloc[-1])
            tr  = trend_label(close, int(ema_len))
            rs_s = float(rs_vs_spy(close, spy_close, int(rs_short)))
            rs_l = float(rs_vs_spy(close, spy_close, int(rs_long)))
            rot  = rs_s - rs_l
            score = strength_score(rs_s, rs_l, rsi, tr)

            leaders.append({
                "Sector": sector,
                "Ticker": t,
                "Trend": tr,
                "RSI": rsi,
                f"RS({rs_short})": rs_s,
                "Rotation": rot,
                "Strength": score,
                "Meter": _meter(score),
            })

    ldf = pd.DataFrame(leaders)
    if ldf.empty:
        st.info("No leaders found (yfinance returned empty).")
        return

    ldf = ldf.sort_values(["Strength","Rotation"], ascending=[False, False])
    ldf = ldf.groupby("Sector").head(int(leaders_per_sector)).reset_index(drop=True)

    st.dataframe(
        ldf.style.format({
            "RSI": "{:.1f}",
            f"RS({rs_short})": "{:.2%}",
            "Rotation": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
        height=420
    )

    st.markdown("### 📈 Chart Viewer")
    pick = st.selectbox("View chart for:", ldf["Ticker"].tolist(), index=0)
    bars = get_hist(pick)
    if bars.empty:
        st.warning("No data for that ticker.")
        return

    bars = bars.tail(220)
    fig = go.Figure(data=[go.Candlestick(
        x=bars.index,
        open=bars["Open"],
        high=bars["High"],
        low=bars["Low"],
        close=bars["Close"],
        name=pick
    )])
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, title=f"{pick} — Daily Candles")
    st.plotly_chart(fig, use_container_width=True)

