# app.py — STRAT Scanner (Modular)
# Version: 1.4.x Modular Build
# Purpose: routing only (sidebar nav + page calls)

from datetime import datetime, timezone

import streamlit as st

# Page modules (we'll create these next)
from pages.scanner import show_scanner
from pages.dashboard import show_market_dashboard
from pages.analyzer import show_ticker_analyzer
from pages.guide import show_user_guide


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="STRAT Scanner", layout="wide")


# =========================
# SIDEBAR NAV
# =========================
st.sidebar.title("Navigation")

pages = [
    "Scanner",
    "📊 Market Dashboard",
    "🔎 Ticker Analyzer",
    "📘 User Guide",
]

page = st.sidebar.radio("Go to", pages)

st.sidebar.caption(
    f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
)


# =========================
# ROUTING
# =========================
if page == "📘 User Guide":
    show_user_guide()
elif page == "📊 Market Dashboard":
    show_market_dashboard()
elif page == "🔎 Ticker Analyzer":
    show_ticker_analyzer()
else:
    show_scanner()
