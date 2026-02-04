# app.py — STRAT Scanner Suite (Modular, Streamlit Cloud-safe)
# Version: 1.0.0 (clean baseline)
# NOTE: Uses custom sidebar navigation (NOT Streamlit multipage folder behavior)

from datetime import datetime, timezone
import streamlit as st

from strat_scanner.pages.dashboard import show_dashboard
from strat_scanner.pages.scanner import show_scanner
from strat_scanner.pages.analyzer import show_analyzer
from strat_scanner.pages.guide import show_guide

st.set_page_config(page_title="STRAT Scanner", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Scanner", "Market Dashboard", "Ticker Analyzer", "User Guide"],
    index=0
)

st.sidebar.caption(f"UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")

if page == "Scanner":
    show_scanner()
elif page == "Market Dashboard":
    show_dashboard()
elif page == "Ticker Analyzer":
    show_analyzer()
else:
    show_guide()
