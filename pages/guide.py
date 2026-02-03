import streamlit as st

def show_guide():
    st.title("📘 User Guide (Baseline)")
    st.markdown("""
### Use the app in this order:
**Market → Sector → Stock → Trigger**

**Dashboard** tells you where money is flowing (rotation + strength).  
**Scanner** tells you what is actionable (regime + triggers + magnitude).  
**Analyzer** explains why a name scores the way it does.

### What “Rotation” means
Rotation = RS(short) − RS(long).  
Positive = improving now. Negative = fading now.

### Why you were seeing everything at 100
That happens when RS inputs aren’t capped. This baseline caps:
- RS to ±10%
- Rotation to ±8%
So the strength meter stays realistic.
""")
