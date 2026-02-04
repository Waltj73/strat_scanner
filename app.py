import streamlit as st

from strat_scanner.pages.scanner import show_scanner
from strat_scanner.pages.dashboard import show_dashboard
from strat_scanner.pages.analyzer import show_analyzer
from strat_scanner.pages.guide import show_guide


st.set_page_config(
    page_title="STRAT Scanner",
    page_icon="📌",
    layout="wide"
)

st.sidebar.title("Navigation")
choice = st.sidebar.radio(
    "Go to",
    ["Scanner", "Market Dashboard", "Ticker Analyzer", "User Guide"],
    index=0
)

PAGES = {
    "Scanner": show_scanner,
    "Market Dashboard": show_dashboard,
    "Ticker Analyzer": show_analyzer,
    "User Guide": show_guide,
}

PAGES[choice]()
