# pages/guide.py — Comprehensive User Guide
from datetime import datetime, timezone
import streamlit as st


def show_guide():
    st.title("📘 STRAT Regime Scanner — Complete User Guide (Comprehensive)")
    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    st.markdown("""
This guide shows you how to use **all parts of the app** (Scanner + Dashboard + Ticker Analyzer)
to consistently develop **high-quality trade ideas** with a repeatable process.

The core philosophy is simple:

### ✅ Market → Sector → Stock → Trigger → Plan
Most traders do it backwards (pick a ticker first, then justify it).
This app forces you to do it in the *highest-probability* order.
""")

    st.divider()

    st.markdown("## 🧭 The 3 Parts of the App (What each one is for)")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
### 1) Scanner
**Purpose:** Find *actionable* setups ranked by quality + magnitude.  
You use this to answer: **“What can I trade *today* with a clean trigger?”**
""")
    with c2:
        st.markdown("""
### 2) Market Dashboard
**Purpose:** Find where money is flowing + build a daily watchlist.  
You use this to answer: **“What groups are strongest / rotating IN?”**
""")
    with c3:
        st.markdown("""
### 3) Ticker Analyzer
**Purpose:** Explain WHY a ticker scores well + give a trade plan.  
You use this to answer: **“Is this real leadership, or noise?”**
""")

    st.divider()

    # =========================
    # MARKET DASHBOARD
    # =========================
    with st.expander("📊 Market Dashboard — Full walkthrough", expanded=True):
        st.markdown("""
### What the Dashboard does
The dashboard is your **market radar**.

It helps you:
- See overall market health (SPY/QQQ/IWM/DIA/VIX)
- Identify **top sectors / metals rotating IN**
- Generate a **Today Watchlist** automatically
- Expand writeups to see an immediate **trade plan** per name
- Type any ticker into Quick Search to get a full “why” breakdown

---

### Section A — Overall Market Sentiment (SPY/QQQ/IWM/DIA + VIX)
You’re looking for:
- **Trend UP** on the main indexes = tailwind for longs
- **Trend DOWN/CHOP** = be more selective and smaller
- **VIX rising hard** = expect wider ranges, faster stops

**Rule of thumb:**
- If most indexes show **UP**, your best trades will be LONG continuation.
- If indexes are mixed/choppy, focus only **A setups** with clean triggers.

---

### Section B — Sector / Metals Rotation + Strength table
This table is the heart of the dashboard.

It shows:
- **RS vs SPY (short)** → who is winning *recently*
- **RS vs SPY (long)** → who has been winning *for a while*
- **Rotation = (RS short − RS long)** → who is *improving right now*
- **Strength (0–100)** → blended leadership score (RS + rotation + trend)

#### Rotation IN / OUT (simple meaning)
- **Rotation IN (positive)**: money is *entering* the sector lately
- **Rotation OUT (negative)**: money is *leaving* the sector lately

This is how you stop trading random tickers and start trading “where the money is going.”

---

### Section C — Today Watchlist (Auto-built)
This is your daily short list.

How it’s built:
1) It selects the **Top Sectors IN**
2) It ranks the **leaders inside those sectors**
3) Optionally filters by **pullback quality** (RSI zone)
4) Shows **Trigger Status**
5) Gives writeups for the best names

#### What “Trigger Status” means
- **READY** = there’s an Inside Bar trigger (entry/stop available)
- **WAIT** = no Inside Bar → do not force the trade

---

### Section D — Watchlist Writeups (expanders)
Each writeup gives:
- Strength + trend + rotation
- Pullback zone check
- Trigger info (Entry/Stop)
- Grade A/B/C
- Targets + invalidation
- What would make it better

Use this to quickly decide:
✅ actionable today  
✅ needs waiting  
❌ avoid

---

### Section E — Quick Ticker Search
Type any ticker and instantly see:
- Whether it belongs on the watchlist
- Why the score is high/low
- Whether it has a trigger
- How to plan the trade

This is your “sanity-check” tool.
""")

    # =========================
    # TICKER ANALYZER
    # =========================
    with st.expander("🔎 Ticker Analyzer — How to use it like a pro", expanded=False):
        st.markdown("""
### What the Analyzer does
The Analyzer is where you verify if a ticker is a **real trade candidate**.

It answers:
- Is trend actually strong?
- Is RS real, or just a short pop?
- Is rotation improving?
- Is there a clean trigger?
- What is the plan?

---

### How to read the Analyzer output

#### 1) Trend
- **UP** is required for clean long swing continuation trades.
- **DOWN/CHOP** means either:
  - avoid longs, or
  - wait until trend flips.

#### 2) RSI + Pullback Zone
Pullback zone is a continuation filter for uptrends.

Default: **40–55**  
Meaning:
- Momentum cooled
- Trend intact
- Continuation becomes likely

#### 3) RS vs SPY (short + long)
- Short RS = recent outperformance
- Long RS = sustained outperformance

#### 4) Rotation
Rotation = RS(short) − RS(long)

- Positive rotation = improving leadership
- Negative rotation = fading leadership

#### 5) Trigger (Inside Bar)
If you see READY:
- Entry = break above Inside Bar high
- Stop = below Inside Bar low
Prefer Weekly inside bars over Daily.

#### 6) Trade Plan Notes
This is the “game plan” section:
- Grade (A/B/C)
- Play type (breakout / pullback / wait / avoid)
- Targets (20d + 63d extremes)
- Invalidation (stop)
- Sizing hint (ATR% based)

Use Analyzer to confirm the trade idea before execution.
""")

    # =========================
    # SCANNER
    # =========================
    with st.expander("🧠 Scanner — How to produce trade ideas from it", expanded=False):
        st.markdown("""
### What the Scanner does
The scanner is the **execution engine**.

It:
- Determines market bias (LONG / SHORT / MIXED)
- Ranks sectors based on that bias
- Lets you drill into a sector and scan tickers
- Ranks tickers by:
  - Setup quality (STRAT alignment)
  - Magnitude (RR + ATR% + compression)

---

### Step-by-step scanner workflow

#### Step 1 — Read market regime
Look at the Market Regime table.
This produces:
- **Bias** (LONG / SHORT / MIXED)
- **Strength** (0–100)
- Bull–Bear diff

**Interpretation**
- LONG + strong → be aggressive on A/B setups
- MIXED/weak → smaller size, only A setups, expect chop

---

#### Step 2 — Use sector ranking
Sectors are ranked after bias is known.

If bias is LONG:
- you care most about sectors with high bull scores
If bias is SHORT:
- you care most about sectors with high bear scores

---

#### Step 3 — Drill into one group
Pick a top sector and scan its tickers.

Use filters:
- ONLY Inside Bars → only actionable triggers
- ONLY 2-1-2 → only developing continuations
- Require alignment → avoids counter-trend junk

---

#### Step 4 — Use “Top Trade Ideas”
This gives you your highest ranked candidates.

**Important:**
If Entry/Stop are blank → there is no trigger → WAIT.

---

#### Step 5 — Confirm in Analyzer (recommended)
Before placing the trade:
- Type ticker into Analyzer
- Confirm leadership + plan

---

### Magnitude Metrics (what they mean)
These are designed to stop you from taking trades that “look good” but have no room.

- **RR**: reward/risk to target
- **ATR%**: volatility (affects sizing + stop tolerance)
- **Compression**: inside bar range vs ATR (tightness)

Ideal:
- RR ≥ 2
- ATR% normal/high but not insane
- compression low (tight range) = explosive potential
""")

    st.divider()

    # =========================
    # FULL DAILY PLAYBOOK
    # =========================
    with st.expander("✅ Daily Playbook (2–5 min): How to build trade ideas consistently", expanded=True):
        st.markdown("""
### Your daily workflow (simple + repeatable)

#### 1) Start on the Dashboard
- Confirm market is trending UP (or not)
- Identify Top Sectors rotating IN
- Note any sectors that are STRONG with positive rotation

#### 2) Review Today Watchlist
- Expand writeups for top candidates
- Flag any tickers that are:
  - Trend UP
  - Strength ≥ 70
  - Rotation positive
  - READY trigger

#### 3) Go to the Scanner for execution confirmation
- Choose the same sector
- Confirm your ticker ranks near the top
- Confirm Entry/Stop exists

#### 4) Final check in Analyzer
- Confirm trade plan notes and RR makes sense
- Make sure you are not chasing extension

#### 5) Place orders
- Use stop order at entry (Inside Bar break)
- Stop at the stop level
- Size based on ATR% (high ATR% = smaller size)

---

### What you should do when...
✅ **Everything is strong and READY**
- Execute A/B setups with confidence

⏳ **Sector is strong but no triggers exist**
- Add to watchlist, WAIT for inside bar

🟠 **Market is MIXED**
- Trade smaller and only A setups
- Prefer Weekly inside bars

🔴 **Trend is down/chop on candidate**
- Skip it, don’t force it
""")

    st.divider()

    # =========================
    # TRADE IDEA TEMPLATE
    # =========================
    with st.expander("📝 Trade Idea Template (copy/paste)", expanded=False):
        st.code(
            """TICKER:
SECTOR:
BIAS (LONG/SHORT/MIXED):
STRENGTH SCORE:
TREND:
RS SHORT / RS LONG:
ROTATION:
TRIGGER (READY/WAIT):
TIMEFRAME (D/W):
ENTRY:
STOP:
RISK UNIT:
TARGETS (T1/T2):
RR ESTIMATE:
PLAN (breakout/pullback/wait):
INVALIDATION:
NOTES:""",
            language="text"
        )

    st.divider()

    with st.expander("⚠️ Common mistakes this tool helps you avoid", expanded=False):
        st.markdown("""
- Picking random tickers without market/sector context
- Trading extended names without pullback/trigger
- Forcing entries when triggers don’t exist
- Ignoring rotation (buying the “old leaders” as money exits)
- Sizing too big when ATR% is high
""")

    st.success("If you want, I can also add a “Quick Start” one-page cheat sheet at the top + a FAQ section.")


# For Streamlit multipage imports
def render():
    show_guide()
