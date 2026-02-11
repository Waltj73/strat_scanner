import streamlit as st

def guide_main():
    st.title("📘 Guide — RSI/RS Rotation App")

    st.markdown("""
## What this app is for
This is **rotation + leadership** using:
- **Relative Strength vs SPY (RS)**
- **RSI**
- **Rotation** (RS short − RS long)
- **Trend** (price vs EMA + EMA slope)
- **Strength score (0–100)**

## How to use it (fast)
1) Check **Overall Bias** (LONG / MIXED / SHORT)
2) Read **Rotation IN / OUT**
3) Click the strongest sectors
4) Pull top tickers into a watchlist
5) Verify in **Ticker Analyzer** + size with **Calculator**
""")

