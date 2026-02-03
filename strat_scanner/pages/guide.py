import streamlit as st

def show_guide():
    st.title("📘 User Guide")
    st.write("""
### What this app does
- **Scanner**: ranks sectors/metals by RS vs SPY + rotation + trend.
- **Dashboard**: shows market sentiment + top groups IN + an auto-built watchlist + write-ups.
- **Analyzer**: deep-dives a single ticker and explains why it’s showing up.

### How to use it for trade ideas (simple workflow)
1) **Dashboard** → Find the top “Rotation IN” groups.
2) Review the **watchlist** to see which tickers lead those groups.
3) Open **Analyzer** on the ticker → confirm:
   - Trend = UP (or avoid)
   - Strength is high
   - RSI is in your pullback zone (if you’re pullback trading)
4) Use Entry/Stop as a *guide* and apply your own chart/STRAT rules.

If you want, next we can expand this guide into a full “playbook” section-by-section.
""")
