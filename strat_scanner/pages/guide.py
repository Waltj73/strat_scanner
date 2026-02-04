# pages/guide.py — Detailed User Guide
# Shows a long-form, readable guide inside the app (not just bullets).

import streamlit as st


def show_guide():
    st.title("📘 STRAT Scanner — Complete User Guide")
    st.caption("How to use the Dashboard, Watchlist, Scanner, and Analyzer together as a single swing-trading workflow.")

    st.markdown("""
## What This App Is Built For

Most traders do it backwards:
they pick a ticker first, then try to justify it.

This app forces a professional decision pipeline:

**Market → Sector → Stock → Trigger**

That’s the whole edge:
you stop hunting random charts and start trading where money is actually flowing.

---

## The 4 Pages (What Each One Does)

### 1) 📊 Market Dashboard
The Dashboard answers:

**“Where is money flowing right now?”**

It helps you *focus* your attention on the best areas so you’re not scanning 500 charts.

You’ll see:
- **Overall market snapshot** (SPY / QQQ / IWM / DIA + optional VIX if you enabled it)
- **Sector + metals table**
- **Relative Strength vs SPY**
- **Rotation (improving vs deteriorating leadership)**
- **Strength score (0–100) with color meter**
- **Auto-built Watchlist** (top groups → top leaders)

Think of the Dashboard as your **macro filter**.

---

### 2) ✅ Today Watchlist (inside Dashboard)
This is your **daily short list**.

How it is built:
1) The app selects the **top sectors rotating IN**
2) From each sector, it scans your pre-defined “leader list”
3) It ranks those leaders by:
   - Strength (0–100)
   - Rotation (improving RS)
   - Short RS vs SPY
4) It shows **trigger status** (READY vs WAIT)
5) You can expand each name to get a full write-up (why it scores the way it does)

The Watchlist answers:

**“If I only look at 10 charts today… what should they be?”**

---

### 3) 🧭 Scanner
The Scanner answers:

**“What is actionable right now?”**

It’s more execution-focused than the Dashboard.

The Scanner:
- Builds a **market regime bias** (LONG / SHORT / MIXED)
- Ranks **sectors** based on that bias
- Lets you drill down into a sector/metals group
- Ranks tickers by:
  - Setup quality (STRAT alignment)
  - Magnitude (RR + ATR% + compression)
- Prints Entry/Stop only when there is an Inside Bar trigger

If Entry/Stop are blank:
that’s not a bug.

That means: **no trigger → no trade**.

---

### 4) 🔎 Ticker Analyzer
The Analyzer answers:

**“Why does this ticker score the way it does?”**

Type any ticker and it will show:
- Trend condition
- RSI state
- RS vs SPY (short and long)
- Rotation (short RS minus long RS)
- Strength score + label
- STRAT context (M/W/D bull, inside bar, 2-1-2)
- Trigger status + levels (when present)

Use this when:
- You want to sanity-check a watchlist name
- You want to investigate a ticker you already follow
- You want to understand “why it ranked high”

---

## Strength Score (0–100)

The Strength Score is designed to be a simple “how likely is follow-through?” number.

It blends:
- **RS vs SPY (short lookback)** → leadership today
- **Rotation (RS short − RS long)** → is leadership improving?
- **Trend** (price vs EMA + EMA slope) → wind at back or not

Interpretation:
- **70–100** → Strong leader (best follow-through)
- **45–69** → Neutral (mixed conditions)
- **0–44** → Weak (avoid for longs)

### Why we cap RS / Rotation
Without caps, one extreme move can dominate the score.

Capping keeps the model tradable and stable.

---

## Rotation IN vs Rotation OUT (What It Really Means)

Rotation is NOT “already strong.”
Rotation is “becoming strong.”

We compute:
**Rotation = RS(short) − RS(long)**

- Positive rotation → money flowing IN now (improving leadership)
- Negative rotation → money flowing OUT (leadership deteriorating)

A sector can be “strong” but rotation can be negative.
That means: it’s still up, but it may be losing leadership.

---

## RSI Pullback Zone (Continuation Entries)

This tool uses RSI as a “pullback quality” filter inside an uptrend.

Default pullback zone for long continuation:
**RSI between 40 and 55**

Meaning:
- Trend is still intact
- Momentum cooled off (no longer extended)
- Continuation becomes more likely

How to read it:
- RSI > 55 → often extended / late
- RSI 40–55 → pullback zone (ideal)
- RSI < 40 → risk of trend damage / deeper mean reversion

This is not a buy signal by itself.
It’s a **context filter**.

---

## STRAT Trigger Logic (How Entry/Stop Is Generated)

When an Inside Bar exists, the app prints actionable levels.

**LONG**
- Entry = break of Inside Bar high
- Stop = below Inside Bar low

**SHORT (Scanner supports this)**
- Entry = break of Inside Bar low
- Stop = above Inside Bar high

Important:
- **Weekly Inside Bar triggers > Daily triggers**
- Daily is fine when the ticker is a true leader, but weekly usually has cleaner follow-through.

---

## “Ready” vs “Wait”
- **READY** → there is a Daily or Weekly Inside Bar so Entry/Stop can be defined.
- **WAIT** → no inside bar trigger printed; you are early (or it’s messy).

The system is designed so you don’t “force” trades.

No trigger = no trade.

---

## When the Dashboard and Scanner Disagree
This is normal and expected.

- Dashboard is using **RS/Rotation/Trend** to determine *where to focus*
- Scanner is using **STRAT triggers** to determine *when to enter*

A sector can be rotating IN on Dashboard…
but have no clean Inside Bars yet in the Scanner.

In that case:
**Focus there — but wait for triggers.**

---

## Recommended Daily Workflow (2–5 minutes)

### Step 1 — Dashboard
- Check overall market trend
- Identify top sectors rotating IN
- Look at the watchlist leaders (top 10–20)

### Step 2 — Watchlist write-ups
- Expand the A/B candidates
- Note who is:
  - Strong (70+)
  - Trend UP
  - Rotation positive
  - RSI in pullback zone (optional)
  - Trigger READY (ideal)

### Step 3 — Scanner (Execution)
- Drill into the strongest sector
- Confirm which tickers have valid entry/stop
- Place stop orders and let price confirm

Goal:
**Let price take you in.**
No trigger = no entry.

---

## Best Default Settings (Recommended)
If you don’t know what to use, these are stable swing settings:

- RS short: **21**
- RS long: **63**
- Trend EMA: **50**
- RSI: **14**
- Pullback zone: **40–55**

---

## Final Notes (How to Actually Win With This)
This system is not built for “more trades.”
It’s built for **fewer, higher-quality trades**.

The only job each day is:
1) Find where money is flowing
2) Pick leaders
3) Wait for a clean trigger
4) Execute with discipline

Consistency comes from the pipeline — not prediction.
""")


# This lets app.py import and display it as a page.
__all__ = ["show_guide"]
