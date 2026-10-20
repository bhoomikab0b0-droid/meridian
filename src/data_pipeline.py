"""
Data acquisition layer.

Tries to pull real OHLCV data via yfinance. If there is no network route to
Yahoo Finance (e.g. running in a sandboxed environment with an egress
allowlist), it transparently falls back to a realistic synthetic OHLCV
generator so the rest of the pipeline (features -> model -> backtest -> UI)
can still be exercised end to end. The UI always makes it clear which mode
produced the data on screen.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_SECONDS = 60 * 30


def _cache_path(ticker: str, period: str) -> str:
    key = hashlib.md5(f"{ticker}_{period}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.parquet")


def fetch_ohlcv(ticker: str, period: str = "3y") -> tuple[pd.DataFrame, str]:
    """
    Returns (dataframe, source) where source is 'live' or 'synthetic-demo'.
    """
    cpath = _cache_path(ticker, period)
    if os.path.exists(cpath) and (time.time() - os.path.getmtime(cpath) < CACHE_TTL_SECONDS):
        df = pd.read_parquet(cpath)
        source = df.attrs.get("source", "live")
        return df, source

    df, source = _try_live_fetch(ticker, period)
    if df is None or df.empty:
        df = _synthetic_ohlcv(ticker, period)
        source = "synthetic-demo"

    df.attrs["source"] = source
    try:
        df.to_parquet(cpath)
    except Exception:
        pass
    return df, source


def _try_live_fetch(ticker: str, period: str):
    try:
        import yfinance as yf
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return None, None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return data, "live"
    except Exception:
        return None, None


def _synthetic_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    """
    Realistic-ish synthetic price series: regime-switching drift/volatility
    (bull / bear / choppy) plus GARCH-like volatility clustering, so
    technical indicators and the model produce non-degenerate signals.
    Deterministic per-ticker seed so results are stable across calls.
    """
    days_map = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260, "10y": 2520}
    n = days_map.get(period, 756)
    seed = int(hashlib.md5(ticker.encode()).hexdigest(), 16) % (2 ** 32)
    rng = np.random.default_rng(seed)

    regimes = [
        (0.00035, 0.011),   # mild bull
        (-0.00025, 0.020),  # bear / stress
        (0.00005, 0.009),   # choppy / sideways
        (0.00055, 0.015),   # strong bull
    ]
    n_segments = max(3, n // 90)
    # Split [0, n) into n_segments contiguous chunks of pseudo-random length
    cut_points = np.sort(rng.choice(np.arange(10, n - 10), size=n_segments - 1, replace=False))
    boundaries = np.concatenate([[0], cut_points, [n]])
    seg_lengths = np.diff(boundaries).astype(int)

    drift = np.zeros(n)
    vol = np.zeros(n)
    pos = 0
    for seg_len in seg_lengths:
        if pos >= n:
            break
        seg_len = min(seg_len, n - pos)
        mu, sigma = regimes[rng.integers(0, len(regimes))]
        drift[pos:pos + seg_len] = mu
        vol[pos:pos + seg_len] = sigma
        pos += seg_len

    # GARCH(1,1)-like volatility clustering on top of regime base vol
    shocks = rng.standard_normal(n)
    garch_vol = np.zeros(n)
    garch_vol[0] = vol[0]
    for t in range(1, n):
        garch_vol[t] = np.sqrt(0.05 * vol[t] ** 2 + 0.9 * garch_vol[t - 1] ** 2 + 0.05 * (garch_vol[t - 1] * shocks[t - 1]) ** 2)

    log_returns = drift + garch_vol * shocks
    base_price = 50 + (seed % 200)
    close = base_price * np.exp(np.cumsum(log_returns))

    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = (rng.lognormal(mean=14, sigma=0.4, size=n) * (1 + np.abs(shocks) * 0.5)).astype(np.int64)

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)
    return df
