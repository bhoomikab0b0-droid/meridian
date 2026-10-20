"""
Basic company fundamentals, shown as read-only context on the dashboard.

These are deliberately kept OUT of the model's feature set: fundamentals
like market cap, P/E, or sector are current-point-in-time facts, and naively
attaching today's P/E to every historical training row would leak
information the model would never have actually had on those past dates
(a classic and easy-to-miss source of backtest overfitting). If you want to
use fundamentals in the model properly, you'd need a point-in-time
fundamentals feed (values as they were known/reported on each historical
date), which is a materially bigger data-engineering lift than what's here.

So: informational only, clearly labeled as such in the UI.
"""
from __future__ import annotations


def fetch_fundamentals(ticker: str) -> dict:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        if not info or len(info) < 3:
            return _unavailable()
        return {
            "available": True,
            "long_name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return _unavailable()


def _unavailable() -> dict:
    return {
        "available": False,
        "note": "Fundamentals unavailable in this environment (no route to a market "
                "data provider). These are informational only and are never fed into "
                "the model -- see src/fundamentals.py for why.",
    }
