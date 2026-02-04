# app.py — STRAT Scanner (Modular Build)
# Routes pages:
# - Scanner
# - Market Dashboard
# - Ticker Analyzer
# - User Guide

from datetime import datetime, timezone
import streamlit as st

# Import page render functions
from pages.scanner import show_scanner
from pages.dashboard import show_dashboard
from pages.analyzer import show_analyzer
from pages.guide import show_guide


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="STRAT Scanner", layout="wide")


# =========================
# SIDEBAR NAV
# =========================
st.sidebar.title("Navigation")
pages = [
    "🧭 Scanner",
    "📊 Market Dashboard",
    "🔎 Ticker Analyzer",
    "📘 User Guide",
]
page = st.sidebar.radio("Go to", pages)

st.sidebar.caption(f"UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")

# Optional: global refresh button
if st.sidebar.button("🔄 Refresh data (clear cache)"):
    st.cache_data.clear()
    st.rerun()


# =========================
# ROUTING
# =========================
if page == "📘 User Guide":
    show_guide()
elif page == "📊 Market Dashboard":
    show_dashboard()
elif page == "🔎 Ticker Analyzer":
    show_analyzer()
else:
    show_scanner()
