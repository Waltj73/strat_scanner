# pages/scanner.py — STRAT Scanner page (regime + sector ranking + drilldown)
# This file is UI only. All logic lives in strat.py / engine.py / indicators.py / data.py

from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from data import get_hist
from strat import (
    tf_frames,
    compute_flags,
    score_regime,
    market_bias_and_strength,
    best_trigger,
)
from engine import magnitude_metrics, calc_scores


# -------------------------
# Local helper
# -------------------------
def alignment_ok(bias: str, flags: Dict[str, bool]) -> bool:
    """
    Alignment filter used by the scanner:
    LONG requires Monthly OR Weekly Bull
    SHORT requires Monthly OR Weekly Bear
    """
    if bias == "LONG":
        return flags.get("M_Bull", False) or flags.get("W_Bull", False)
    if bias == "SHORT":
        return flags.get("M_Bear", False) or flags.get("W_Bear", False)
    return False


def show_scanner(
    MARKET_ETFS: Dict[str, str],
    SECTOR_ETFS: Dict[str, str],
    SECTOR_TICKERS: Dict[str, List[str]],
):
    st.title("🧭 STRAT Regime Scanner (Auto LONG/SHORT + Magnitude)")
    st.caption("Market bias → sector ranking → drilldown into names → inside-bar triggers + RR/ATR% scoring.")

    # =========================
    # FILTERS
    # =========================
    with st.expander("Filters", expanded=True):
        colA, colB, colC, colD = st.columns([1.1, 1.2, 1.6, 1.1])

        with colA:
            only_inside = st.checkbox("ONLY Inside Bars (D or W)", value=False)
        with colB:
            only_212 = st.checkbox("ONLY 2-1-2 forming (bias direction)", value=False)
        with colC:
            require_alignment = st.checkbox("Require Monthly OR Weekly alignment (bias direction)", value=True)
        with colD:
            top_k = st.slider("Top Picks count", min_value=3, max_value=12, value=6)

        r1, _ = st.columns([1, 3])
        with r1:
            if st.button("Refresh data"):
                st.cache_data.clear()
                st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # =========================
    # MARKET REGIME
    # =========================
    market_rows: List[Dict] = []
    for name, etf in MARKET_ETFS.items():
        d = get_hist(etf)
        if d.empty:
            flags = {k: False for k in [
                "D_Bull","W_Bull","M_Bull","D_Bear","W_Bear","M_Bear",
                "D_Inside","W_Inside","M_Inside","D_212Up","W_212Up","D_212Dn","W_212Dn"
            ]}
            bull, bear = 0, 0
        else:
            d_tf, w_tf, m_tf = tf_frames(d)
            flags = compute_flags(d_tf, w_tf, m_tf)
            bull, bear = score_regime(flags)

        row = {"Market": name, "ETF": etf, "BullScore": bull, "BearScore": bear}
        row.update(flags)
        market_rows.append(row)

    bias, strength, bull_bear_diff = market_bias_and_strength(market_rows)

    st.subheader("Market Regime (SPY / QQQ / IWM / DIA) — Bull vs Bear")
    st.write(f"**Bias:** {bias}  |  **Strength:** {strength}/100  |  **Bull–Bear diff:** {bull_bear_diff}")

    market_df = pd.DataFrame(market_rows)[[
        "Market","ETF","BullScore","BearScore",
        "M_Bull","W_Bull","D_Bull",
        "M_Bear","W_Bear","D_Bear",
        "W_212Up","D_212Up","W_212Dn","D_212Dn",
        "W_Inside","D_Inside"
    ]]
    st.dataframe(market_df, use_container_width=True, hide_index=True)

    # =========================
    # SECTOR TABLE (STRAT based)
    # =========================
    sector_rows: List[Dict] = []
    for sector, etf in SECTOR_ETFS.items():
        d = get_hist(etf)
        if d.empty:
            flags = {k: False for k in [
                "D_Bull","W_Bull","M_Bull","D_Bear","W_Bear","M_Bear",
                "D_Inside","W_Inside","M_Inside","D_212Up","W_212Up","D_212Dn","W_212Dn"
            ]}
            bull, bear = 0, 0
        else:
            d_tf, w_tf, m_tf = tf_frames(d)
            flags = compute_flags(d_tf, w_tf, m_tf)
            bull, bear = score_regime(flags)

        row = {"Sector": sector, "ETF": etf, "BullScore": bull, "BearScore": bear}
        row.update(flags)
        sector_rows.append(row)

    sectors_df = pd.DataFrame(sector_rows)

    if sectors_df.empty:
        st.warning("Sector data unavailable right now (yfinance returned empty). Try Refresh.")
        return

    if bias == "LONG":
        sectors_df = sectors_df.sort_values(["BullScore", "BearScore"], ascending=[False, True])
    elif bias == "SHORT":
        sectors_df = sectors_df.sort_values(["BearScore", "BullScore"], ascending=[False, True])
    else:
        sectors_df["Dominance"] = (sectors_df["BullScore"] - sectors_df["BearScore"]).abs()
        sectors_df = sectors_df.sort_values("Dominance", ascending=False)

    st.subheader("Sectors + Metals — ranked after bias is known (STRAT regime)")
    st.dataframe(
        sectors_df[[
            "Sector","ETF","BullScore","BearScore",
            "M_Bull","W_Bull","D_Bull",
            "M_Bear","W_Bear","D_Bear",
            "W_212Up","D_212Up","W_212Dn","D_212Dn",
            "W_Inside","D_Inside"
        ]],
        use_container_width=True,
        hide_index=True,
        height=420
    )

    # =========================
    # DRILLDOWN
    # =========================
    st.subheader("Drill into a group (ranks names by Setup + Magnitude)")
    group_names = list(SECTOR_TICKERS.keys())
    if not group_names:
        st.warning("No SECTOR_TICKERS configured.")
        return

    sector_choice = st.selectbox("Choose a group:", options=group_names, index=0)
    tickers = SECTOR_TICKERS.get(sector_choice, [])

    if not tickers:
        st.info("No tickers in this group list.")
        return

    st.write(f"Selected: **{sector_choice}** ({SECTOR_ETFS.get(sector_choice, '')}) — list size: **{len(tickers)}**")

    scan_n = st.slider("How many tickers to scan", min_value=1, max_value=len(tickers), value=min(20, len(tickers)))
    scan_list = tickers[:scan_n]

    eff_bias = bias if bias in ("LONG", "SHORT") else "LONG"
    cand_rows: List[Dict] = []

    for t in scan_list:
        d = get_hist(t)
        if d.empty:
            continue

        d_tf, w_tf, m_tf = tf_frames(d)
        flags = compute_flags(d_tf, w_tf, m_tf)

        # Optional alignment requirement
        if require_alignment and eff_bias in ("LONG", "SHORT") and not alignment_ok(eff_bias, flags):
            continue

        # Optional inside-bar requirement
        if only_inside and not (flags.get("D_Inside") or flags.get("W_Inside")):
            continue

        # Optional 2-1-2 requirement in bias direction
        if only_212:
            if eff_bias == "LONG" and not (flags.get("D_212Up") or flags.get("W_212Up")):
                continue
            if eff_bias == "SHORT" and not (flags.get("D_212Dn") or flags.get("W_212Dn")):
                continue

        tf, entry, stop = best_trigger(eff_bias, d_tf, w_tf)
        rr, atrp, room, compression = magnitude_metrics(eff_bias, d_tf, entry, stop)
        setup_score, mag_score, total_score = calc_scores(
            eff_bias, flags, rr, atrp, compression, entry, stop
        )

        trigger_status = "READY" if (flags.get("W_Inside") or flags.get("D_Inside")) else "WAIT (No Inside Bar)"

        cand = {
            "Ticker": t,
            "Trigger": trigger_status,
            "SetupScore": setup_score,
            "MagScore": mag_score,
            "TotalScore": total_score,
            "TF": tf,
            "Entry": None if entry is None else round(float(entry), 2),
            "Stop": None if stop is None else round(float(stop), 2),
            "Room": None if room is None else round(float(room), 2),
            "RR": None if rr is None else round(float(rr), 2),
            "ATR%": None if atrp is None else round(float(atrp), 2),
            "W_Inside": bool(flags.get("W_Inside")),
            "D_Inside": bool(flags.get("D_Inside")),
            "W_212Up": bool(flags.get("W_212Up")),
            "D_212Up": bool(flags.get("D_212Up")),
            "W_212Dn": bool(flags.get("W_212Dn")),
            "D_212Dn": bool(flags.get("D_212Dn")),
            "M_Bull": bool(flags.get("M_Bull")),
            "W_Bull": bool(flags.get("W_Bull")),
            "D_Bull": bool(flags.get("D_Bull")),
            "M_Bear": bool(flags.get("M_Bear")),
            "W_Bear": bool(flags.get("W_Bear")),
            "D_Bear": bool(flags.get("D_Bear")),
        }
        cand_rows.append(cand)

    cand_df = pd.DataFrame(cand_rows)

    if cand_df.empty:
        st.info("No matches under current filters. Loosen filters or pick another group.")
        return

    cand_df = cand_df.sort_values("TotalScore", ascending=False)

    st.markdown(f"### Top Trade Ideas — Bias: **{eff_bias}** (Top {top_k})")
    st.dataframe(
        cand_df.head(top_k),
        use_container_width=True,
        hide_index=True,
        height=320
    )

    st.markdown("### 🎯 Trade of the Day (best score + valid trigger + RR≥2)")
    valid = cand_df.dropna(subset=["Entry", "Stop", "RR"]).copy()
    valid = valid[valid["RR"] >= 2.0]
    if valid.empty:
        st.warning("No valid trigger found (needs Inside Bar levels + RR≥2).")
    else:
        best = valid.iloc[0]
        st.success(
            f"**{best['Ticker']}** | Bias: **{eff_bias}** | TF: **{best['TF']}** | "
            f"Entry: **{best['Entry']}** | Stop: **{best['Stop']}** | "
            f"RR: **{best['RR']}** | ATR%: **{best['ATR%']}**"
        )

    st.markdown("### All Matches (ranked by TotalScore)")
    st.dataframe(
        cand_df,
        use_container_width=True,
        hide_index=True,
        height=520
    )

    # =========================
    # QUICK MARKET READ (Scanner-only rotation list)
    # =========================
    st.subheader("Quick Market Read (Scanner)")
    if eff_bias == "LONG":
        rotation_in = [f"{r.Sector}({r.ETF})" for r in sectors_df.head(3).itertuples(index=False)]
        rotation_out = [f"{r.Sector}({r.ETF})" for r in sectors_df.tail(3).itertuples(index=False)]
        badge = "🟢"
        plan = "Plan: LONG only. Focus strong groups with triggers."
    elif eff_bias == "SHORT":
        rotation_in = [f"{r.Sector}({r.ETF})" for r in sectors_df.head(3).itertuples(index=False)]
        rotation_out = [f"{r.Sector}({r.ETF})" for r in sectors_df.tail(3).itertuples(index=False)]
        badge = "🔴"
        plan = "Plan: SHORT only. Focus weak groups with triggers."
    else:
        rotation_in, rotation_out = [], []
        badge = "🟠"
        plan = "Plan: Defensive. Trade smaller or wait for A+ triggers."

    st.write(f"Bias: **{eff_bias}** {badge} | Strength: **{strength}/100** | Bull–Bear diff: **{bull_bear_diff}**")

    if rotation_in:
        st.write("### Rotation IN (STRAT regime ranking)")
        st.write(", ".join(rotation_in))
    if rotation_out:
        st.write("### Rotation OUT (STRAT regime ranking)")
        st.write(", ".join(rotation_out))

    st.success(plan)
    st.caption("Trigger logic: weekly inside bar preferred. No inside bar = no trigger = wait.")
