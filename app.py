import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from app_pages.rsi_dashboard import rsi_dashboard_main
from app_pages.analyzer import analyzer_main
from app_pages.calculator import calculator_main
from app_pages.guide import guide_main

# NEW
from app_pages.rotation import rotation_main

st.set_page_config(page_title="RSI/RS Rotation Dashboard", layout="wide")

st.sidebar.title("Navigation")

pages = {
    "📊 RSI/RS Dashboard": rsi_dashboard_main,
    "🧭 Sector Rotation (A/B/C)": rotation_main,   # NEW
    "🔎 Ticker Analyzer": analyzer_main,
    "🧮 Share Calculator": calculator_main,
    "📘 Guide": guide_main,
}

choice = st.sidebar.radio("Go to", list(pages.keys()), index=0)
pages[choice]()
