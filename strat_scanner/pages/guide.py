import streamlit as st


def show_guide():
    st.title("📘 STRAT Scanner — Complete User Guide")

    st.markdown("""
### What this app does
This app helps you turn “market noise” into a short list of **trade-ready candidates** by combining:

1) **Market sentiment**
2) **Sector / metals rotation**
3) **Relative Strength vs SPY**
4) **STRAT bar logic** (1 / 2U / 2D / 3 + trigger status)
5) A repeatable workflow to build trade ideas

---

## 1) Scanner Page (Rotation • Leaders • Drilldown)
### What you’re looking at
**Sector / Metals Rotation + Strength**
- **RS vs SPY (short)**: Who’s outperforming SPY lately
- **RS vs SPY (long)**: Who’s outperforming over the longer window
- **Rotation (short - long)**: Positive means momentum improving; negative means fading
- **Trend**: UP vs DOWN/CHOP (price vs EMA)
- **Strength meter**: 0–100 blend of RS + rotation + trend bias

### Drilldown
Pick a sector group → you’ll see the leaders ranked by:
- Strength
- Rotation
- RS short
- STRAT trigger status

### STRAT in this app (simple + practical)
- **1** = inside bar (compression)
- **2U** = breaks up only (bull expansion)
- **2D** = breaks down only (bear expansion)
- **3** = breaks both sides (wild/outside)

**TriggerStatus**
- READY (2U) = bullish trigger present
- READY (2D) = bearish trigger present
- WAIT (inside bar) = needs break
- READY (3 close high/low) = outside bar with conviction
- WAIT (3 bar) = outside bar but messy

---

## 2) Market Dashboard Page (Sentiment • Rotation • Leaders • Watchlist)
### Market Sentiment Panel
Shows SPY / QQQ / IWM / DIA / VIX with:
- Price
- Short-term return
- Trend
- RSI

Use this to decide:
- Are we in “risk-on” mode (SPY/QQQ trend UP)?
- Is VIX elevated (tighter sizing / more caution)?

### Watchlist Builder
Takes the “top groups IN” and pulls leader tickers to build a daily list.
Then it generates write-ups so you can quickly decide:
- Is this a candidate?
- What is the bias?
- What does STRAT say right now?

---

## 3) Ticker Analyzer (Deep Dive)
Type any ticker → get:
- Candle chart
- Strength + trend + RS + rotation
- STRAT bar types and trigger status
- A plain-English “what to do next” view

---

## 4) How to use this to develop trade ideas (workflow)
### Step A — Decide market mode
Dashboard → look at SPY/QQQ trend + VIX.
- If trend UP and VIX stable: lean long ideas
- If trend DOWN/CHOP or VIX rising: tighten risk or lean short

### Step B — Pick groups rotating IN
Scanner/Dashboard → sort by Strength + Rotation.
- Top 3 groups IN is a good daily focus

### Step C — Drilldown leaders
Scanner drilldown → pick the leaders with:
- Strength 70+
- Rotation positive
- Trend UP (for longs) or DOWN/CHOP (for shorts)
- STRAT trigger = READY (2U/2D) or WAIT (inside bar) near breakout

### Step D — Use STRAT to time it
- If WAIT (inside bar): watch for break
- If READY (2U): use high break / continuation logic
- If READY (2D): use low break / continuation logic

### Step E — Confirm with pullback behavior
If trend UP, you often want RSI in a “pullback zone” before entry.
That’s why the writeups call out pullback confirmation.

---

## 5) Common issues / notes
- yfinance can sometimes return empty data briefly → hit Refresh.
- Indices can have strange volume → volume is not used for scoring.
- STRAT signals are not “buy/sell” — they are **timing context**.

---

If you want, next we can add:
- A true A/B/C grade later (without breaking the app)
- ATR-based stop sizing
- A “trigger watchlist” of inside bars ready to break
""")
