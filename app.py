    st.subheader("Overall Market Sentiment")

    # --- Helper: Market grade from SPY/QQQ/IWM/DIA (ignore VIX for grading) ---
    def market_grade(trend_map: Dict[str, str], rsi_map: Dict[str, float], ret_map: Dict[str, float]) -> Tuple[str, str, int]:
        core = ["SPY", "QQQ", "IWM", "DIA"]
        ups = sum(1 for t in core if trend_map.get(t) == "UP")
        dns = sum(1 for t in core if trend_map.get(t) != "UP")

        rsi_vals = [rsi_map.get(t, 50.0) for t in core if np.isfinite(rsi_map.get(t, np.nan))]
        ret_vals = [ret_map.get(t, 0.0) for t in core if np.isfinite(ret_map.get(t, np.nan))]

        avg_rsi = float(np.mean(rsi_vals)) if rsi_vals else 50.0
        avg_ret = float(np.mean(ret_vals)) if ret_vals else 0.0

        # Strength score: simple, stable, intuitive
        strength = int(np.clip(50 + (ups - dns) * 12 + (avg_rsi - 50) * 0.8 + (avg_ret * 100) * 0.6, 0, 100))

        # Trend label
        if ups >= 3:
            trend = "UP"
        elif dns >= 3:
            trend = "DOWN"
        else:
            trend = "MIXED"

        # Grade logic (simple)
        if trend == "UP" and strength >= 70:
            grade = "A"
        elif trend == "DOWN" and strength >= 70:
            grade = "A"
        elif strength >= 55:
            grade = "B"
        else:
            grade = "C"

        return trend, grade, strength

    market_syms = list(MARKET_ETFS.values()) + ["^VIX"]

    # Collect stats so we can compute the grade box
    trend_map: Dict[str, str] = {}
    rsi_map: Dict[str, float] = {}
    ret_map: Dict[str, float] = {}

    # Layout: left = metrics grid, right = sentiment grade box
    left, right = st.columns([3, 1])

    with left:
        mcols = st.columns(len(market_syms))
        for i, sym in enumerate(market_syms):
            d = get_hist(sym)
            if d.empty:
                with mcols[i]:
                    st.metric(sym, "n/a", "n/a")
                continue

            close = d["Close"].dropna()
            if close.empty or len(close) < 10:
                with mcols[i]:
                    st.metric(sym, "n/a", "n/a")
                continue

            tr = trend_label(close, int(ema_trend_len))
            r = float(rsi_wilder(close, int(rsi_len)).iloc[-1])
            ret = float(total_return(close, int(rs_short)).iloc[-1]) if len(close) > rs_short else np.nan

            trend_map[sym] = tr
            rsi_map[sym] = r
            ret_map[sym] = ret

            with mcols[i]:
                st.metric(sym, f"{close.iloc[-1]:.2f}", f"{(ret*100):.1f}%" if np.isfinite(ret) else "n/a")
                st.write(f"Trend: **{tr}**")
                st.write(f"RSI: **{r:.1f}**")

    with right:
        # Compute grade from SPY/QQQ/IWM/DIA only
        trend, grade, strength = market_grade(trend_map, rsi_map, ret_map)

        badge = "🟢" if trend == "UP" else "🔴" if trend == "DOWN" else "🟠"

        st.markdown("### Market Grade")
        st.write(f"**Trend:** {badge} **{trend}**")
        st.write(f"**Grade:** **{grade}**")
        st.write(f"**Strength:** **{strength}/100**")

        if trend == "UP":
            st.success("Lean LONG: focus leaders + triggers.")
        elif trend == "DOWN":
            st.error("Risk OFF: be defensive (or shorts only if you enable short logic).")
        else:
            st.warning("Mixed: trade smaller, wait for A+ triggers.")
