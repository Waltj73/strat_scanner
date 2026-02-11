from __future__ import annotations
from typing import Dict, List

MARKET_ETFS: Dict[str, str] = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Russell 2000": "IWM",
    "Dow Jones": "DIA",
    "Volatility (VIX)": "^VIX",
}

SECTOR_ETFS: Dict[str, str] = {
    "Energy": "XLE",
    "Comm Services": "XLC",
    "Staples": "XLP",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Discretionary": "XLY",
    "Utilities": "XLU",
    "Financials": "XLF",
    "Technology": "XLK",
    "Health Care": "XLV",
}

# Keep your lists simple; you can expand later
SECTOR_TICKERS: Dict[str, List[str]] = {
    "Energy": ["XOM","CVX","COP","EOG","SLB","HAL"],
    "Comm Services": ["GOOGL","META","NFLX","TMUS","DIS","CMCSA"],
    "Staples": ["PG","KO","PEP","WMT","COST","MDLZ"],
    "Materials": ["LIN","APD","NUE","FCX","NEM","ALB"],
    "Industrials": ["CAT","DE","HON","GE","LMT","RTX"],
    "Real Estate": ["PLD","AMT","EQIX","O","WELL","DLR"],
    "Discretionary": ["AMZN","TSLA","HD","MCD","NKE","LOW"],
    "Utilities": ["NEE","DUK","SO","AEP","EXC","XEL"],
    "Financials": ["JPM","BAC","WFC","GS","MS","C"],
    "Technology": ["AAPL","MSFT","NVDA","AVGO","AMD","ORCL"],
    "Health Care": ["UNH","JNJ","LLY","PFE","MRK","ABBV"],
}
