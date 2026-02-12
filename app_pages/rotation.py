import streamlit as st
import pandas as pd
import numpy as np

from app_pages.rotation_engine import (
    build_sector_rotation_table,
    build_rotation_watchlist,
    fetch_close_volume,
    compute_rotation_score,
    compute_early_breakout_score,
    BENCHMARK,
)

def rotation_main():
    st.title("🧭 Sector Rotation (A/B/C)")
    st.caption("Rotation Score (0–100): RS trend + momentum improvement + pullback strength + accumulation volume + breadth proxy.")

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.selectbox("Data window", ["6mo", "9mo", "1y", "2y"], index=1)
    with col2:
        sector_threshold = st.slider("Inflow sector threshold", 50, 85, 70, 1)
    with col3:
        top_sectors = st.slider("Top sectors to scan", 3, 10, 6, 1)

    tabA, tabB, tabC = st.tabs(["A) Heatmap", "B) Auto Watchlist", "C) Early Breakouts"])

    # =========================
    # A) HEATMAP
    # =========================
    with tabA:
        st.subheader("A) Sector Rotation Heatmap")

        sector_df = build_sector_rotation_table(period=period)

        st.dataframe(
            sector_df.style
                .background_gradient(subset=["Rotation Score"], cmap="RdYlGn")
                .background_gradient(subset=["Accel (1w)"], cmap="RdYlGn"),
            use_container_width=True
        )

        st.info("Tip: Rising Rotation Score + positive Accel (1w) = money flowing IN early.")

    # =========================
    # B) WATCHLIST
    # =========================
    with tabB:
        st.subheader("B) Auto Watchlist from Rotation Leaders")

        leaders, tickers = build_rotation_watchlist(
            sector_threshold=sector_threshold,
            top_sectors=top_sectors,
            period=period
        )

        st.write("Top inflow sectors:")
        st.dataframe(leaders, use_container_width=True)

        if not tickers:
            st.warning("No tickers found (baskets empty or threshold too high).")
        else:
            symbols = [BENCHMARK] + tickers
            data = fetch_close_volume(symbols, period=period, interval="1d")
            closes = data["Close"]
            vols = data["Volume"]
            bench = closes[BENCHMARK].dropna()

            rows = []
            for t in tickers:
                if t not in closes.columns:
                    continue
                c = closes[t].dropna()
                v = vols[t].dropna()
                out = compute_rotation_score(c, v, bench)
                if np.isnan(out.get("total", np.nan)):
                    continue
                # small bonus for RSI “early rotation zone”
                bonus = 3.0 if 45 <= out.get("rsi", 0) <= 65 else 0.0
                rows.append({
                    "Ticker": t,
                    "Rotation Score": round(float(out["total"] + bonus), 1),
                    "RS Trend Raw": round(out.get("rs_trend_raw", np.nan), 2),
                    "RSI": round(out.get("rsi", np.nan), 1),
                    "Up/Dn Vol": round(out.get("vol_ratio", np.nan), 2),
                })

            df = pd.DataFrame(rows).sort_values("Rotation Score", ascending=False)
            st.dataframe(
                df.style.background_gradient(subset=["Rotation Score"], cmap="RdYlGn"),
                use_container_width=True
            )

    # =========================
    # C) EARLY BREAKOUTS
    # =========================
    with tabC:
        st.subheader("C) Early Breakouts (Before the obvious breakout candle)")

        leaders, tickers = build_rotation_watchlist(
            sector_threshold=max(55, sector_threshold - 5),
            top_sectors=top_sectors,
            period=period
        )

        if not tickers:
            st.warning("No tickers to scan (baskets empty or threshold too high).")
        else:
            symbols = [BENCHMARK] + tickers
            data = fetch_close_volume(symbols, period=period, interval="1d")
            closes = data["Close"]
            vols = data["Volume"]
            bench = closes[BENCHMARK].dropna()

            rows = []
            for t in tickers:
                if t not in closes.columns:
                    continue
                c = closes[t].dropna()
                v = vols[t].dropna()
                out = compute_early_breakout_score(c, v, bench)
                if np.isnan(out.get("breakout_score", np.nan)):
                    continue
                rows.append({
                    "Ticker": t,
                    "Breakout Score": out["breakout_score"],
                    "RS Trend": out.get("rs_trend", np.nan),
                    "Vol Ratio": out.get("vol_ratio", np.nan),
                    "Compression": out.get("compression", np.nan),
                    "Near 20D High": out.get("near_20d_high", False),
                })

            df = pd.DataFrame(rows).sort_values("Breakout Score", ascending=False).head(50)
            st.dataframe(
                df.style.background_gradient(subset=["Breakout Score"], cmap="RdYlGn"),
                use_container_width=True
            )

            st.caption("High Breakout Score = RS improving + EMA21 rising + volatility compression + accumulation volume + near resistance.")
