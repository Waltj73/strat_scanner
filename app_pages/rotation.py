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

def _score_bucket(score: float) -> str:
    if score >= 80: return "🟩 Strong IN"
    if score >= 60: return "🟢 Building"
    if score >= 40: return "🟡 Neutral"
    if score >= 20: return "🟠 OUT"
    return "🟥 Heavy OUT"

def _accel_arrow(a: float) -> str:
    if a >= 8: return "↑↑"
    if a >= 3: return "↑"
    if a <= -8: return "↓↓"
    if a <= -3: return "↓"
    return "→"

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def rotation_main():
    st.title("🧭 Sector Rotation (A/B/C)")
    st.caption("Rotation Score (0–100): RS trend + momentum improvement + pullback strength + accumulation volume + breadth proxy.")

    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.selectbox("Data window", ["6mo", "9mo", "1y", "2y"], index=1)
    with col2:
        sector_threshold = st.slider("Inflow sector threshold", 50, 85, 70, 1)
    with col3:
        top_sectors = st.slider("Top sectors to scan", 3, 10, 6, 1)

    tabA, tabB, tabC = st.tabs(["A) Heatmap", "B) Auto Watchlist", "C) Early Breakouts"])

    # =========================
    # A) HEATMAP (NO matplotlib)
    # =========================
    with tabA:
        st.subheader("A) Sector Rotation Heatmap")

        sector_df = build_sector_rotation_table(period=period).copy()

        # Add visual columns (no pandas Styler)
        sector_df["Rotation Score"] = sector_df["Rotation Score"].apply(_safe_float)
        sector_df["Accel (1w)"] = sector_df["Accel (1w)"].apply(_safe_float)

        sector_df["Status"] = sector_df["Rotation Score"].apply(lambda x: _score_bucket(x) if pd.notna(x) else "")
        sector_df["Trend"] = sector_df["Accel (1w)"].apply(lambda x: _accel_arrow(x) if pd.notna(x) else "")

        st.data_editor(
            sector_df,
            use_container_width=True,
            disabled=True,
            column_config={
                "Rotation Score": st.column_config.ProgressColumn(
                    "Rotation Score",
                    min_value=0,
                    max_value=100,
                    format="%.1f",
                ),
                "Accel (1w)": st.column_config.NumberColumn("Accel (1w)", format="%.1f"),
                "Status": st.column_config.TextColumn("Status"),
                "Trend": st.column_config.TextColumn("Trend"),
            },
        )

        st.info("Tip: Rotation Score rising + Accel (1w) positive = money flowing IN early.")

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
        st.data_editor(leaders, use_container_width=True, disabled=True)

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

                total_score = float(out["total"] + bonus)

                rows.append({
                    "Ticker": t,
                    "Rotation Score": round(total_score, 1),
                    "Status": _score_bucket(total_score),
                    "RS Trend Raw": round(out.get("rs_trend_raw", np.nan), 2),
                    "RSI": round(out.get("rsi", np.nan), 1),
                    "Up/Dn Vol": round(out.get("vol_ratio", np.nan), 2),
                })

            df = pd.DataFrame(rows).sort_values("Rotation Score", ascending=False)

            st.data_editor(
                df,
                use_container_width=True,
                disabled=True,
                column_config={
                    "Rotation Score": st.column_config.ProgressColumn(
                        "Rotation Score",
                        min_value=0,
                        max_value=100,
                        format="%.1f",
                    ),
                    "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                    "Up/Dn Vol": st.column_config.NumberColumn("Up/Dn Vol", format="%.2f"),
                },
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

                bscore = float(out["breakout_score"])

                rows.append({
                    "Ticker": t,
                    "Breakout Score": bscore,
                    "Status": _score_bucket(bscore),
                    "RS Trend": out.get("rs_trend", np.nan),
                    "Vol Ratio": out.get("vol_ratio", np.nan),
                    "Compression": out.get("compression", np.nan),
                    "Near 20D High": out.get("near_20d_high", False),
                })

            df = pd.DataFrame(rows).sort_values("Breakout Score", ascending=False).head(50)

            st.data_editor(
                df,
                use_container_width=True,
                disabled=True,
                column_config={
                    "Breakout Score": st.column_config.ProgressColumn(
                        "Breakout Score",
                        min_value=0,
                        max_value=100,
                        format="%.1f",
                    ),
                    "Vol Ratio": st.column_config.NumberColumn("Vol Ratio", format="%.2f"),
                    "Compression": st.column_config.NumberColumn("Compression", format="%.4f"),
                },
            )

            st.caption("High Breakout Score = RS improving + EMA21 rising + volatility compression + accumulation volume + near resistance.")
