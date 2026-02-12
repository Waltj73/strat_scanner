from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config.universe import MARKET_ETFS, SECTOR_ETFS, SECTOR_TICKERS
from data.fetch import get_hist
from indicators.rs_rsi import (
    rsi_wilder, rs_vs_spy, trend_label,
    strength_score, strength_trend
)

# ==========================================================
# Helpers
# ==========================================================
def _meter(score: int) -> str:
    if score >= 70:
        return "STRONG"
    if score >= 45:
        return "NEUTRAL"
    return "WEAK"

def _rotation_in_out(df: pd.DataFrame, n: int, bias: str):
    if df.empty:
        return df, df
    if bias == "SHORT":
        rot_in = df.sort_values("Rotation", ascending=True).head(n)
        rot_out = df.sort_values("Rotation", ascending=False).head(n)
    else:
        rot_in = df.sort_values("Rotation", ascending=False).head(n)
        rot_out = df.sort_values("Rotation", ascending=True).head(n)
    return rot_in, rot_out

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _safe_tail(s: pd.Series, n: int) -> pd.Series:
    s = s.dropna()
    return s.iloc[-n:] if len(s) >= n else s

def _pullback_features(
    bars: pd.DataFrame,
    rsi_len: int,
    ema20_len: int,
    ema50_len: int,
    deep_low: float,
    deep_high: float,
    shallow_low: float,
    shallow_high: float,
    near20_pct: float
) -> dict:
    """
    Computes deep/shallow pullback + READY + TRIGGER for the last bar,
    including "yesterday-ready -> trigger today" logic.
    Uses:
      - TrendUp: close > EMA50 and EMA50 rising
      - NearEMA20: abs(close-EMA20)/EMA20 <= near20_pct
      - Deep: RSI in [35,45]
      - Shallow: RSI in [45,55]
      - READY: (Deep/Shallow) AND RSI rising (RSI[t] > RSI[t-2]) AND close > EMA20
      - TRIGGER: READY yesterday AND today close > yesterday high
    """
    if bars is None or bars.empty:
        return {}

    close = bars["Close"].dropna()
    high = bars["High"].dropna()
    if close.empty or high.empty:
        return {}

    # Need enough bars for EMA + RSI rising lookback + trigger logic
    if len(close) < max(ema50_len, ema20_len, rsi_len) + 5:
        return {}

    # Align (in case of NaNs)
    idx = close.index.intersection(high.index)
    close = close.loc[idx]
    high = high.loc[idx]
    if len(close) < max(ema50_len, ema20_len, rsi_len) + 5:
        return {}

    ema20 = _ema(close, int(ema20_len))
    ema50 = _ema(close, int(ema50_len))

    rsi_series = rsi_wilder(close, int(rsi_len)).dropna()
    # ensure alignment after RSI calc
    idx2 = close.index.intersection(rsi_series.index).intersection(ema20.index).intersection(ema50.index).intersection(high.index)
    close = close.loc[idx2]
    high = high.loc[idx2]
    ema20 = ema20.loc[idx2]
    ema50 = ema50.loc[idx2]
    rsi_series = rsi_series.loc[idx2]

    if len(close) < 5:
        return {}

    # last + yesterday
    c0 = float(close.iloc[-1])
    c1 = float(close.iloc[-2])
    h1 = float(high.iloc[-2])

    e20_0 = float(ema20.iloc[-1])
    e20_1 = float(ema20.iloc[-2])

    e50_0 = float(ema50.iloc[-1])
    e50_1 = float(ema50.iloc[-2])
    e50_2 = float(ema50.iloc[-3])

    r0 = float(rsi_series.iloc[-1])
    r1 = float(rsi_series.iloc[-2])

    # RSI rising = today RSI > RSI two bars ago (stabilized)
    rsi_rising_0 = False
    rsi_rising_1 = False
    if len(rsi_series) >= 3:
        rsi_rising_0 = float(rsi_series.iloc[-1]) > float(rsi_series.iloc[-3])
        rsi_rising_1 = float(rsi_series.iloc[-2]) > float(rsi_series.iloc[-4]) if len(rsi_series) >= 4 else False

    # TrendUp today/yesterday (EMA50 rising + price above)
    trend_up_0 = (c0 > e50_0) and (e50_0 > e50_1)
    trend_up_1 = (c1 > e50_1) and (e50_1 > e50_2)

    # Near EMA20 today/yesterday
    near20_0 = (abs(c0 - e20_0) / e20_0) * 100.0 <= float(near20_pct)
    near20_1 = (abs(c1 - e20_1) / e20_1) * 100.0 <= float(near20_pct)

    # Zones today/yesterday
    deep_0 = (r0 >= deep_low) and (r0 <= deep_high)
    shallow_0 = (r0 >= shallow_low) and (r0 <= shallow_high)

    deep_1 = (r1 >= deep_low) and (r1 <= deep_high)
    shallow_1 = (r1 >= shallow_low) and (r1 <= shallow_high)

    # Pullback flags (today/yesterday)
    deep_pb_0 = trend_up_0 and near20_0 and deep_0
    shallow_pb_0 = trend_up_0 and near20_0 and shallow_0

    deep_pb_1 = trend_up_1 and near20_1 and deep_1
    shallow_pb_1 = trend_up_1 and near20_1 and shallow_1

    # READY (your choice B: RSI rising + close > EMA20)
    ready_deep_0 = deep_pb_0 and rsi_rising_0 and (c0 > e20_0)
    ready_shallow_0 = shallow_pb_0 and rsi_rising_0 and (c0 > e20_0)
    ready_any_0 = ready_deep_0 or ready_shallow_0

    ready_deep_1 = deep_pb_1 and rsi_rising_1 and (c1 > e20_1)
    ready_shallow_1 = shallow_pb_1 and rsi_rising_1 and (c1 > e20_1)
    ready_any_1 = ready_deep_1 or ready_shallow_1

    # TRIGGER today (your buy rule): ready yesterday AND close > yesterday high
    trig_deep_0 = bool(ready_deep_1 and (c0 > h1))
    trig_shallow_0 = bool(ready_shallow_1 and (c0 > h1))
    trig_any_0 = bool(trig_deep_0 or trig_shallow_0)

    # labels for display
    if deep_pb_0:
        pb_type = "DEEP"
    elif shallow_pb_0:
        pb_type = "SHALLOW"
    elif trend_up_0 and r0 > 55:
        pb_type = "EXTENDED"
    else:
        pb_type = "—"

    if trig_any_0:
        ready_label = "TRIGGER"
    elif ready_deep_0:
        ready_label = "READY_DEEP"
    elif ready_shallow_0:
        ready_label = "READY_SHALLOW"
    else:
        ready_label = "—"

    return {
        "TrendUp": trend_up_0,
        "EMA20_DistPct": (abs(c0 - e20_0) / e20_0) * 100.0,
        "NearEMA20": near20_0,
        "DeepPB": deep_pb_0,
        "ShallowPB": shallow_pb_0,
        "ReadyDeep": ready_deep_0,
        "ReadyShallow": ready_shallow_0,
        "ReadyAny": ready_any_0,
        "TriggerDeep": trig_deep_0,
        "TriggerShallow": trig_shallow_0,
        "TriggerAny": trig_any_0,
        "PB_Type": pb_type,
        "READY": ready_label,
    }


# ==========================================================
# Main
# ==========================================================
def rsi_dashboard_main():
    st.title("📊 RSI/RS Rotation Dashboard (RSI/RS-only)")

    top = st.columns([1, 1, 6])
    with top[0]:
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    with st.expander("Settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            rs_short = st.selectbox("RS short lookback", [21, 30, 42], index=0)
        with c2:
            rs_long = st.selectbox("RS long lookback", [63, 90, 126], index=0)
        with c3:
            ema_len = st.selectbox("Trend EMA (for label)", [50, 100, 200], index=0)
        with c4:
            # default to RSI 7 (your system), but still selectable
            rsi_len = st.selectbox("RSI length", [7, 14, 21], index=0)

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            top_sectors = st.slider("Top sectors to scan for leaders", 3, 11, 6)
        with c6:
            leaders_per_sector = st.slider("Leaders per sector", 3, 10, 5)
        with c7:
            rotation_n = st.slider("Rotation IN/OUT count", 3, 8, 5)
        with c8:
            strength_week_lookback = st.selectbox("StrengthTrend lookback (bars)", [5, 10], index=0)

        st.markdown("### A+ Leader Rules (editable)")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            aplus_strength = st.slider("Min Strength", 60, 90, 75)
        with a2:
            require_trend_up = st.checkbox("Require Trend UP", value=True)
        with a3:
            require_rot_pos = st.checkbox("Require Rotation > 0", value=True)
        with a4:
            use_rsi_pullback = st.checkbox("Require RSI Pullback Window", value=True)

        pb1, pb2 = st.columns(2)
        with pb1:
            pb_low = st.slider("RSI Pullback Low", 20, 60, 40)
        with pb2:
            pb_high = st.slider("RSI Pullback High", 35, 80, 55)

        st.markdown("---")
        st.markdown("### Pullback Setup Logic (Deep + Shallow) — Dashboard Upgrade")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            deep_low = st.number_input("Deep PB RSI Low", value=35, min_value=10, max_value=60, step=1)
        with p2:
            deep_high = st.number_input("Deep PB RSI High", value=45, min_value=10, max_value=80, step=1)
        with p3:
            shallow_low = st.number_input("Shallow PB RSI Low", value=45, min_value=10, max_value=80, step=1)
        with p4:
            shallow_high = st.number_input("Shallow PB RSI High", value=55, min_value=10, max_value=90, step=1)

        p5, p6, p7 = st.columns(3)
        with p5:
            near20_pct = st.number_input("Near EMA20 threshold (%)", value=1.2, min_value=0.1, max_value=5.0, step=0.1)
        with p6:
            pb_ema20_len = st.selectbox("Pullback EMA (for Near/Ready)", [10, 20, 21], index=1)
        with p7:
            pb_ema50_len = st.selectbox("Trend EMA (for Pullbacks)", [50, 100, 200], index=0)

        st.caption("READY (clean): Pullback flag + RSI rising (t > t-2) + Close > EMA20. TRIGGER: READY yesterday + Close > yesterday High.")

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
        if close.empty or len(close) < (rs_long + 15):
            continue

        rsi = float(rsi_wilder(close, int(rsi_len)).iloc[-1])
        tr = trend_label(close, int(ema_len))
        rs_s = float(rs_vs_spy(close, spy_close, int(rs_short)))
        rs_l = float(rs_vs_spy(close, spy_close, int(rs_long)))
        rot = rs_s - rs_l
        score = strength_score(rs_s, rs_l, rsi, tr)
        s_trend = strength_trend(
            close, spy_close,
            int(rs_short), int(rs_long),
            int(ema_len), int(rsi_len),
            int(strength_week_lookback)
        )

        market_rows.append({
            "Name": name,
            "Ticker": sym,
            "Trend": tr,
            "RSI": rsi,
            f"RS({rs_short})": rs_s,
            f"RS({rs_long})": rs_l,
            "Rotation": rot,
            "Strength": score,
            "StrengthTrend": s_trend,
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
            "StrengthTrend": "{:+.0f}",
        }),
        use_container_width=True,
        hide_index=True
    )

    core = mdf[mdf["Ticker"].isin(["SPY", "QQQ", "IWM", "DIA"])].copy()
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
    st.subheader("Sector Rotation (RS vs SPY + Rotation + Strength + StrengthTrend)")

    sector_rows = []
    for sector, etf in SECTOR_ETFS.items():
        d = get_hist(etf)
        if d.empty:
            continue
        close = d["Close"].dropna()
        if close.empty or len(close) < (rs_long + 15):
            continue

        rsi = float(rsi_wilder(close, int(rsi_len)).iloc[-1])
        tr = trend_label(close, int(ema_len))
        rs_s = float(rs_vs_spy(close, spy_close, int(rs_short)))
        rs_l = float(rs_vs_spy(close, spy_close, int(rs_long)))
        rot = rs_s - rs_l
        score = strength_score(rs_s, rs_l, rsi, tr)
        s_trend = strength_trend(
            close, spy_close,
            int(rs_short), int(rs_long),
            int(ema_len), int(rsi_len),
            int(strength_week_lookback)
        )

        sector_rows.append({
            "Sector": sector,
            "ETF": etf,
            "Trend": tr,
            "RSI": rsi,
            f"RS({rs_short})": rs_s,
            f"RS({rs_long})": rs_l,
            "Rotation": rot,
            "Strength": score,
            "StrengthTrend": s_trend,
            "Meter": _meter(score),
        })

    sdf = pd.DataFrame(sector_rows)
    if sdf.empty:
        st.warning("No sector data returned.")
        return

    sdf = sdf.sort_values(["Strength", "Rotation"], ascending=[False, False])

    # ======================
    # Leaders + A+ Leaders (+ Pullback upgrade)
    # ======================
    st.subheader("✅ Leaders (top tickers from top sectors) + ⭐ A+ Leaders + 🎯 Pullback/Ready/Trigger")

    top_groups = sdf.head(int(top_sectors))["Sector"].tolist()

    leaders = []
    for sector in top_groups:
        names = SECTOR_TICKERS.get(sector, [])
        for t in names[:40]:
            d = get_hist(t)
            if d.empty:
                continue

            close = d["Close"].dropna()
            if close.empty or len(close) < (rs_long + 15):
                continue

            # Core dashboard metrics
            rsi = float(rsi_wilder(close, int(rsi_len)).iloc[-1])
            tr = trend_label(close, int(ema_len))
            rs_s = float(rs_vs_spy(close, spy_close, int(rs_short)))
            rs_l = float(rs_vs_spy(close, spy_close, int(rs_long)))
            rot = rs_s - rs_l
            score = strength_score(rs_s, rs_l, rsi, tr)
            s_trend = strength_trend(
                close, spy_close,
                int(rs_short), int(rs_long),
                int(ema_len), int(rsi_len),
                int(strength_week_lookback)
            )

            # Pullback/Ready/Trigger (uses OHLC)
            pb = _pullback_features(
                d.tail(260),
                rsi_len=int(rsi_len),
                ema20_len=int(pb_ema20_len),
                ema50_len=int(pb_ema50_len),
                deep_low=float(deep_low),
                deep_high=float(deep_high),
                shallow_low=float(shallow_low),
                shallow_high=float(shallow_high),
                near20_pct=float(near20_pct),
            )

            leaders.append({
                "Sector": sector,
                "Ticker": t,
                "Trend": tr,
                "RSI": rsi,
                f"RS({rs_short})": rs_s,
                "Rotation": rot,
                "Strength": score,
                "StrengthTrend": s_trend,
                "Meter": _meter(score),

                # --- Upgrade columns
                "PB": pb.get("PB_Type", "—"),
                "Near20%": pb.get("EMA20_DistPct", np.nan),
                "Ready": pb.get("READY", "—"),
                "Trig": bool(pb.get("TriggerAny", False)),
                "DeepPB": bool(pb.get("DeepPB", False)),
                "ShallowPB": bool(pb.get("ShallowPB", False)),
                "ReadyAny": bool(pb.get("ReadyAny", False)),
                "TriggerAny": bool(pb.get("TriggerAny", False)),
            })

    ldf = pd.DataFrame(leaders)
    if ldf.empty:
        st.info("No leaders found (data returned empty).")
        return

    # Merge setup counts into sector table (for the sectors we scanned)
    sector_counts = (
        ldf.groupby("Sector")[["DeepPB", "ShallowPB", "ReadyAny", "TriggerAny"]]
        .sum()
        .rename(columns={
            "DeepPB": "DeepPB#",
            "ShallowPB": "ShallowPB#",
            "ReadyAny": "Ready#",
            "TriggerAny": "Trigger#",
        })
        .reset_index()
    )

    sdf2 = sdf.merge(sector_counts, on="Sector", how="left")
    for col in ["DeepPB#", "ShallowPB#", "Ready#", "Trigger#"]:
        sdf2[col] = sdf2[col].fillna(0).astype(int)

    st.markdown("### Sector table + Pullback counts (from scanned leaders)")
    st.dataframe(
        sdf2.style.format({
            "RSI": "{:.1f}",
            f"RS({rs_short})": "{:.2%}",
            f"RS({rs_long})": "{:.2%}",
            "Rotation": "{:.2%}",
            "StrengthTrend": "{:+.0f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=420
    )

    rot_in, rot_out = _rotation_in_out(sdf2, int(rotation_n), bias)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔁 Rotation IN")
        for _, r in rot_in.iterrows():
            st.write(
                f"✅ **{r['Sector']}** ({r['ETF']}) — Rot {r['Rotation']:.2%} | Strength {int(r['Strength'])} | Δ {r['StrengthTrend']:+.0f} "
                f"| Deep {int(r.get('DeepPB#', 0))} | Shallow {int(r.get('ShallowPB#', 0))} | Ready {int(r.get('Ready#', 0))} | Trig {int(r.get('Trigger#', 0))}"
            )
    with c2:
        st.markdown("### 🔁 Rotation OUT")
        for _, r in rot_out.iterrows():
            st.write(
                f"❌ **{r['Sector']}** ({r['ETF']}) — Rot {r['Rotation']:.2%} | Strength {int(r['Strength'])} | Δ {r['StrengthTrend']:+.0f} "
                f"| Deep {int(r.get('DeepPB#', 0))} | Shallow {int(r.get('ShallowPB#', 0))} | Ready {int(r.get('Ready#', 0))} | Trig {int(r.get('Trigger#', 0))}"
            )

    # Leaders table
    ldf_sorted = ldf.sort_values(["Strength", "Rotation"], ascending=[False, False])
    leaders_df = ldf_sorted.groupby("Sector").head(int(leaders_per_sector)).reset_index(drop=True)

    st.markdown("### Leaders (by sector) — now includes Pullback/Ready/Trigger")
    st.dataframe(
        leaders_df.style.format({
            "RSI": "{:.1f}",
            f"RS({rs_short})": "{:.2%}",
            "Rotation": "{:.2%}",
            "StrengthTrend": "{:+.0f}",
            "Near20%": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=420
    )

    # A+ detector (same as before)
    def _aplus_mask(df: pd.DataFrame) -> pd.Series:
        m = (df["Strength"] >= float(aplus_strength))
        if require_trend_up:
            m = m & (df["Trend"] == "UP")
        if require_rot_pos:
            m = m & (df["Rotation"] > 0)
        if use_rsi_pullback:
            m = m & (df["RSI"] >= float(pb_low)) & (df["RSI"] <= float(pb_high))
        return m

    aplus = ldf_sorted[_aplus_mask(ldf_sorted)].copy()
    aplus = aplus.sort_values(["Strength", "StrengthTrend", "Rotation"], ascending=[False, False, False]).head(15)

    st.markdown("### ⭐ Top 15 A+ Leaders (market-wide, from scanned sectors)")
    if aplus.empty:
        st.info("No A+ leaders under current rules. Try lowering Min Strength or widening RSI window.")
    else:
        st.dataframe(
            aplus.style.format({
                "RSI": "{:.1f}",
                f"RS({rs_short})": "{:.2%}",
                "Rotation": "{:.2%}",
                "StrengthTrend": "{:+.0f}",
                "Near20%": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
            height=420
        )

    # Quick filters for action
    st.markdown("### 🎯 Trade Queue (Ready / Trigger)")
    q1, q2, q3 = st.columns([2, 2, 2])
    with q1:
        show_only_ready = st.checkbox("Show only READY", value=True)
    with q2:
        show_only_trigger = st.checkbox("Show only TRIGGER (today)", value=False)
    with q3:
        queue_limit = st.slider("Queue size", 10, 60, 25)

    queue = ldf_sorted.copy()
    if show_only_ready:
        queue = queue[queue["ReadyAny"] == True]
    if show_only_trigger:
        queue = queue[queue["TriggerAny"] == True]

    queue = queue.sort_values(["TriggerAny", "Strength", "StrengthTrend", "Rotation"], ascending=[False, False, False, False]).head(int(queue_limit))

    if queue.empty:
        st.info("No names in queue under current filters.")
    else:
        st.dataframe(
            queue[["Sector", "Ticker", "PB", "Near20%", "Ready", "Trig", "Strength", "StrengthTrend", "Rotation", "RSI"]].style.format({
                "Near20%": "{:.2f}",
                "Rotation": "{:.2%}",
                "RSI": "{:.1f}",
                "StrengthTrend": "{:+.0f}",
            }),
            use_container_width=True,
            hide_index=True,
            height=420
        )

    # ======================
    # Chart viewer
    # ======================
    st.markdown("### 📈 Chart Viewer")
    chart_list = []
    if not queue.empty:
        chart_list = queue["Ticker"].tolist()
    elif not aplus.empty:
        chart_list = aplus["Ticker"].tolist()
    elif not leaders_df.empty:
        chart_list = leaders_df["Ticker"].tolist()
    else:
        chart_list = ldf_sorted["Ticker"].tolist()

    pick = st.selectbox("View chart for:", chart_list, index=0)
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

    st.caption(
        "StrengthTrend = Strength(today) − Strength(~1 week ago). "
        "Pullback logic: Deep = RSI 35–45, Shallow = RSI 45–55, Near EMA20 within threshold. "
        "READY = Pullback + RSI rising (t > t-2) + Close > EMA20. "
        "TRIGGER = READY yesterday + Close > yesterday High."
    )
