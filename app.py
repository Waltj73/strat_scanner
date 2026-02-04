# app.py
import streamlit as st

from strat_scanner.pages.scanner import show_scanner
from strat_scanner.pages.dashboard import show_dashboard
from strat_scanner.pages.analyzer import show_analyzer
from strat_scanner.pages.guide import show_guide

st.set_page_config(page_title="STRAT Scanner", layout="wide")

PAGES = {
    "Scanner": show_scanner,
    "Dashboard": show_dashboard,
    "Ticker Analyzer": show_analyzer,
    "Guide": show_guide,
}

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[choice]()
