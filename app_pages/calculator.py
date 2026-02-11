from __future__ import annotations
import streamlit as st

def calculator_main():
    st.title("🧮 Share / Risk Calculator")

    c1,c2,c3 = st.columns(3)
    with c1:
        entry = st.number_input("Entry", value=100.0, step=0.1)
    with c2:
        stop = st.number_input("Stop", value=95.0, step=0.1)
    with c3:
        risk = st.number_input("$ Risk", value=200.0, step=10.0)

    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        st.warning("Entry and Stop must be different.")
        return

    shares = int(risk // risk_per_share)
    position_cost = shares * entry

    st.subheader("Result")
    st.metric("Risk per share", f"${risk_per_share:.2f}")
    st.metric("Shares", f"{shares}")
    st.metric("Position cost", f"${position_cost:,.2f}")

    st.caption("This is a basic position-sizing calculator (shares).")

