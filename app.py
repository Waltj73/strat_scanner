# app.py (repo root)
import streamlit as st

st.set_page_config(page_title="STRAT Scanner", layout="wide")

st.sidebar.title("Navigation")

PAGES = {}

def safe_import(name, importer):
    try:
        PAGES[name] = importer
    except Exception as e:
        def fail():
            st.error(f"Page failed to load: {name}")
            st.exception(e)
        PAGES[name] = fail

safe_import("Scanner", lambda: __import__("strat_scanner.pages.scanner", fromlist=["show_scanner"]).show_scanner())
safe_import("Dashboard", lambda: __import__("strat_scanner.pages.dashboard", fromlist=["show_dashboard"]).show_dashboard())
safe_import("Ticker Analyzer", lambda: __import__("strat_scanner.pages.analyzer", fromlist=["show_analyzer"]).show_analyzer())
safe_import("User Guide", lambda: __import__("strat_scanner.pages.guide", fromlist=["show_guide"]).show_guide())

choice = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[choice]()
