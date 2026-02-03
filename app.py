import streamlit as st

# Import page renderers
from strat_scanner.pages.scanner import show_scanner
from strat_scanner.pages.dashboard import show_dashboard
from strat_scanner.pages.analyzer import show_analyzer
from strat_scanner.pages.guide import show_guide


# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="STRAT Regime Scanner",
    layout="wide"
)

st.title("STRAT Regime Scanner")

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Scanner",
        "Market Dashboard",
        "Ticker Analyzer",
        "User Guide"
    ]
)

# ---------------------------
# Page Routing
# ---------------------------
if page == "Scanner":
    show_scanner()

elif page == "Market Dashboard":
    show_dashboard()

elif page == "Ticker Analyzer":
    show_analyzer()

else:
    show_guide()

