# pages/dashboard.py — Market Dashboard (Sentiment • Rotation • Strength • Watchlist • Writeups)
# UI only. Logic uses data.py + indicators.py + strat.py + engine.py

from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from data import get_hist
from indicators import (
    rsi_wilder,
    total_return,
    rs_vs_spy,
    trend_label,
    clamp_rs,
    strength_meter,
    strength_label,
    meter_style,
    strength_style,
    pullback_zone_ok,
    RS_CAP,
    ROT_CAP,
)
from engine import writeup_block
from engine import analyze_ticker as analyze_basic
from strat import compute_flags, tf_frames, best_trigger


# -------------------------
# Internal: explainable analyzer (RS/Rotation/Strength + STRAT trigger)
# -------------------------
def analyze_ticker_dashboard(
    ticker: str,
    spy_close: pd.Series,
    rs_short: int,
    rs_long: int,
    ema_trend_len: int,
    rsi_len: int,
) -> dict | None:
    d = get_hist(ticker)
    if d.empty:
        return None

    close = d["Close"].dropna()
    if close.empty or len(close) < max(rs_long, 80) + 10:
        return None

    tr = trend_label(close, ema_trend_len)
    rsi_v = float(rsi_wilder(close, rsi_len).iloc[-1])

    rs_s = float(rs_vs_spy(close, spy_close, rs_short).iloc[-1])
    rs_l = float(rs_vs_spy(close, spy_close, rs_long).iloc[-1])

    rs_s_c = clamp_rs(rs_s, -RS_CAP, RS_CAP)
    rs_l_c = clamp_rs(rs_l, -RS_CAP, RS_CAP)

    rot = rs_s - rs_l
    rot_c = clamp_rs(rot, -ROT_CAP, ROT_CAP)

    score = strength_meter(rs_s_c, rot_c, tr)
    meter = strength_label(score)

    # STRAT context for triggers
    d_tf, w_tf, m_tf = tf_frames(d)
    flags = compute_flags(d_tf, w_tf, m_tf)
    tf, entry, stop = best_trigger("LONG", d_tf, w_tf)
    trigger_status = "READY" if (flags.get("W_Inside") or flags.get("D_Inside")) else "WAIT (No Inside Bar)"

    explain = [
        f"Trend = {tr} (price vs EMA({ema_trend_len}) + EMA slope)",
        f"RSI({rsi_len}) = {rsi_v:.1f}",
        f"RS vs SPY short ({rs_short}) = {rs_s*100:.1f}% (capped {rs_s_c*100:.1f}%)",
        f"RS vs SPY long ({rs_long}) = {rs_l*100:.1f}% (capped {rs_l_c*100:.1f}%)",
        f"Rotation = RS short − RS long = {rot*100:.1f}% (capped {rot_c*100:.1f}%)",
        f"Strength = {score}/100 ({meter})",
    ]

    return {
        "Ticker": ticker.upper(),
        "Trend": tr,
        "RSI": rsi_v,
        "RS_short": rs_s,
        "RS_long": rs_l,
        "Rotation": rot,
        "Strength": score,
        "Meter": meter,
        "TriggerStatus": trigger_status,
        "TF": tf,
        "Entry": None if entry is None else round(float(entry), 2),
        "Stop": None if stop is None else round(float(stop), 2),
        "Explain": explain,
    }


def show_dashboard(
    MARKET_ETFS: Dict[str, str],
    SECTOR_ETFS: Dict[str, str],
    SECTOR_TICKERS: Dict[str, List[str]],
):
    st.title("📊 Market Dashboard (Sentiment • Rotation • Strength • Watchlist)")

    # =========================
    # SETTINGS
    # =========================
    with st.expander("Dashboard Settings", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.2])
        with c1:
            rs_short = st.selectbox("RS Lookback (short)", [21, 30, 42], index=0)
        with c2:
            rs_long = st.selectbox("RS Lookback (long)", [63, 90, 126], index=0)
        with c3:
            ema_trend_len = st.selectbox("Trend EMA", [50, 100, 200], index=0)
        with c4:
            rsi_len = st.selectbox("RSI Length", [7, 14, 21], index=1)
        with c5:
            if st.button("Refresh data"):
                st.cache_data.clear()
                st.rerun()

    with st.expander("Today Watchlist Settings", expanded=True):
        w1, w2, w3, w4, w5 = st.columns([1, 1, 1, 1, 1.4])
        with w1:
            top_sectors_in = st.slider("Top Groups IN", 1, 6, 3)
        with w2:
            leaders_per_sector = st.slider("Leaders per group", 3, 10, 5)
        with w3:
            pb_low = st.slider("RSI Pullback Low (UP trend)", 25, 60, 40)
        with w4:
            pb_high = st.slider("RSI Pullback High (UP trend)", 35, 75, 55)
        with w5:
            strict_pullback = st.checkbox("Strict pullback filter (only RSI-in-zone)", value=False)

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # =========================
    # MARKET SENTIMENT ROW
    # =========================
    st.subheader("Overall Market Sentiment")
    market_syms = list(MARKET_ETFS.values()) + ["^VIX"]
    cols = st.columns(len(market_syms))

    for i, sym in enumerate(market_syms):
        d = get_hist(sym)
        if d.empty:
            with cols[i]:
                st.metric(sym, "n/a", "n/a")
            continue

        close = d["Close"].dropna()
        if close.empty or len(close) < 10:
            with cols[i]:
                st.metric(sym, "n/a", "n/a")
            continue

        tr = trend_label(close, int(ema_trend_len))
        r = float(rsi_wilder(close, int(rsi_len)).iloc[-1])
        ret = float(total_return(close, int(rs_short)).iloc[-1]) if len(close) > rs_short else np.nan

        with cols[i]:
            st.metric(sym, f"{close.iloc[-1]:.2f}", f"{(ret*100):.1f}%" if np.isfinite(ret) else "n/a")
            st.write(f"Trend: **{tr}**")
            st.write(f"RSI: **{r:.1f}**")

    # =========================
    # SPY BASELINE
    # =========================
    spy_df = get_hist("SPY")
    if spy_df.empty:
        st.warning("SPY data unavailable; cannot compute RS vs SPY.")
        return

    spy = spy_df["Close"].dropna()
    if len(spy) < (rs_long + 10):
        st.warning("Not enough SPY history for these lookbacks.")
        return

    # =========================
    # SECTOR ROTATION TABLE
    # =========================
    st.subheader("Sector / Metals Rotation + Strength (Relative Strength vs SPY)")

    sector_rows = []
    for name, etf in SECTOR_ETFS.items():
        d = get_hist(etf)
        if d.empty:
            continue

        close = d["Close"].dropna()
        if len(close) < (rs_long + 10):
            continue

        rs_s = float(rs_vs_spy(close, spy, int(rs_short)).iloc[-1])
        rs_l = float(rs_vs_spy(close, spy, int(rs_long)).iloc[-1])

        rs_s_c = clamp_rs(rs_s, -RS_CAP, RS_CAP)
        rs_l_c = clamp_rs(rs_l, -RS_CAP, RS_CAP)

        rot = rs_s - rs_l
        rot_c = clamp_rs(rot, -ROT_CAP, ROT_CAP)

        tr = trend_label(close, int(ema_trend_len))
        rsi_v = float(rsi_wilder(close, int(rsi_len)).iloc[-1])

        # ✅ IMPORTANT: pass capped values (prevents “everything = 100”)
        score = strength_meter(rs_s_c, rot_c, tr)

        sector_rows.append({
            "Group": name,
            "ETF": etf,
            "Strength": int(score),
            "Meter": strength_label(int(score)),
            f"RS vs SPY ({rs_short})": rs_s,
            f"RS vs SPY ({rs_long})": rs_l,
            "Rotation (RS short - RS long)": rot,
            "Trend": tr,
            "RSI": rsi_v
        })

    sectors = pd.DataFrame(sector_rows)
    if sectors.empty:
        st.warning("Sector data unavailable right now (yfinance returned empty). Try Refresh.")
        return

    sectors = sectors.sort_values(["Strength", "Rotation (RS short - RS long)"], ascending=[False, False])

    styled = (
        sectors
        .style
        .format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            f"RS vs SPY ({rs_long})": "{:.2%}",
            "Rotation (RS short - RS long)": "{:.2%}",
            "RSI": "{:.1f}"
        })
        .applymap(meter_style, subset=["Meter"])
        .applymap(strength_style, subset=["Strength"])
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

    # =========================
    # TODAY WATCHLIST
    # =========================
    st.subheader("✅ Today Watchlist (Auto-built from Rotation IN + Leaders)")

    top_groups = sectors.head(int(top_sectors_in))[["Group", "ETF", "Strength", "Meter"]].to_dict("records")
    st.write(
        "**Top Groups IN:** " +
        ", ".join([f"{g['Group']}({g['ETF']}) {g['Meter']} {g['Strength']}/100" for g in top_groups])
    )

    watchlist = []
    for g in top_groups:
        group_name = g["Group"]
        names = SECTOR_TICKERS.get(group_name, [])
        if not names:
            continue

        infos = []
        for sym in names[:min(30, len(names))]:
            info = analyze_ticker_dashboard(sym, spy, int(rs_short), int(rs_long), int(ema_trend_len), int(rsi_len))
            if info is not None:
                info["Group"] = group_name
                infos.append(info)

        if not infos:
            continue

        infos = sorted(infos, key=lambda x: (x["Strength"], x["Rotation"], x["RS_short"]), reverse=True)

        if strict_pullback:
            infos = [x for x in infos if pullback_zone_ok(x["Trend"], x["RSI"], pb_low, pb_high)]

        pick = infos[:int(leaders_per_sector)]
        watchlist.extend(pick)

    if not watchlist:
        st.warning("Watchlist is empty under current settings. Loosen pullback filter or increase scan sizes.")
        return

    wdf = pd.DataFrame([{
        "Group": x["Group"],
        "Ticker": x["Ticker"],
        "Strength": x["Strength"],
        "Meter": x["Meter"],
        "Trend": x["Trend"],
        "RSI": x["RSI"],
        f"RS vs SPY ({rs_short})": x["RS_short"],
        "Rotation": x["Rotation"],
        "Trigger": x["TriggerStatus"],
        "TF": x["TF"],
        "Entry": x["Entry"],
        "Stop": x["Stop"],
    } for x in watchlist]).sort_values(["Strength", "Rotation"], ascending=[False, False])

    wstyled = (
        wdf.style
        .format({
            f"RS vs SPY ({rs_short})": "{:.2%}",
            "Rotation": "{:.2%}",
            "RSI": "{:.1f}"
        })
        .applymap(meter_style, subset=["Meter"])
        .applymap(strength_style, subset=["Strength"])
    )

    st.dataframe(wstyled, use_container_width=True, hide_index=True, height=420)

    st.write("### 📌 Watchlist Write-ups (click to expand)")
    for rec in wdf.head(25).to_dict("records"):
        t = rec["Ticker"]
        with st.expander(f"{rec['Group']} — {t} | {rec['Meter']} {rec['Strength']}/100 | {rec['Trigger']}"):
            info = analyze_ticker_dashboard(t, spy, int(rs_short), int(rs_long), int(ema_trend_len), int(rsi_len))
            if info is None:
                st.warning("No data returned.")
                continue

            # Quick read
            st.markdown(f"#### {t} — {info['Meter']} ({info['Strength']}/100)")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.write(f"**Trend:** {info['Trend']}")
            with c2: st.write(f"**RSI:** {info['RSI']:.1f}")
            with c3: st.write(f"**RS short:** {info['RS_short']*100:.1f}%")
            with c4: st.write(f"**Rotation:** {info['Rotation']*100:.1f}%")

            pb_ok = pullback_zone_ok(info["Trend"], info["RSI"], pb_low, pb_high)
            st.write(f"**Pullback Zone ({pb_low}-{pb_high}) OK?** {'✅ YES' if pb_ok else '❌ NO'}")

            st.write(
                f"**Trigger:** {info['TriggerStatus']}"
                + (f" | TF: **{info['TF']}** | Entry: **{info['Entry']}** | Stop: **{info['Stop']}**" if info["Entry"] else "")
            )

            st.markdown("##### Why it scores this way")
            for line in info["Explain"]:
                st.write(f"- {line}")

    # =========================
    # QUICK TICKER SEARCH
    # =========================
    st.subheader("🔎 Quick Ticker Search (Explain why it’s a candidate)")
    q = st.text_input("Type a ticker:", value="AAPL")
    if q:
        info = analyze_ticker_dashboard(q.strip().upper(), spy, int(rs_short), int(rs_long), int(ema_trend_len), int(rsi_len))
        if info is None:
            st.warning("No data returned (bad ticker or yfinance empty). Try another symbol.")
        else:
            st.markdown(f"### {info['Ticker']} — {info['Meter']} ({info['Strength']}/100)")
            st.write(f"Trend: **{info['Trend']}** | RSI: **{info['RSI']:.1f}** | Rotation: **{info['Rotation']*100:.1f}%**")
            st.write(f"Trigger: **{info['TriggerStatus']}**")
            for line in info["Explain"]:
                st.write(f"- {line}")
