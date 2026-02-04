from __future__ import annotations

def meter_style(val: str) -> str:
    if val == "STRONG":
        return "background-color: #114b2b; color: white;"
    if val == "NEUTRAL":
        return "background-color: #5a4b11; color: white;"
    return "background-color: #5a1111; color: white;"
