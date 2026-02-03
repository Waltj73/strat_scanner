from typing import Dict
import streamlit as st

def writeup_block(info: Dict, pb_low: float, pb_high: float):
    st.markdown(f"### {info['Ticker']} — {info['Meter']} ({info['Strength']}/100)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trend", info["Trend"])
    c2.metric("RSI", f"{info['RSI']:.1f}")
    c3.metric("RS short", f"{info['RS_short']:.2%}")
    c4.metric("Rotation", f"{info['Rotation']:.2%}")

    st.write(f"Trigger: **{info['TriggerStatus']}**  | TF: **{info['TF']}**")
    st.write(f"Entry (guide): **{info['Entry']:.2f}**")
    st.write(f"Stop (guide): **{info['Stop']:.2f}**")

    if info["Trend"] == "UP" and (pb_low <= info["RSI"] <= pb_high):
        st.success(f"Pullback zone OK (RSI between {pb_low}–{pb_high})")
    else:
        st.info("Pullback zone not confirmed.")
