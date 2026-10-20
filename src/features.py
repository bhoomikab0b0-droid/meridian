"""
Feature engineering for the prediction pipeline.

Builds a rich technical-indicator feature set on top of OHLCV data.
All indicators are computed causally (no look-ahead) so every row's
features only use information available up to and including that bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have columns: Open, High, Low, Close, Volume (indexed by date).
    Returns a new dataframe with engineered features appended.
    """
    out = df.copy()
    close, high, low, vol, open_ = (
        out["Close"], out["High"], out["Low"], out["Volume"], out["Open"]
    )

    # --- Returns / momentum at multiple horizons ---
    for lag in (1, 2, 3, 5, 10, 20):
        out[f"ret_{lag}d"] = close.pct_change(lag)
        out[f"vol_chg_{lag}d"] = vol.pct_change(lag)

    out["log_ret_1d"] = np.log(close / close.shift(1))
    out["overnight_gap"] = _safe_div(open_ - close.shift(1), close.shift(1))
    out["intraday_range"] = _safe_div(high - low, close)
    out["close_position"] = _safe_div(close - low, (high - low).replace(0, np.nan))

    # --- Trend indicators ---
    for w in (5, 10, 20, 50, 100, 200):
        sma = ta.trend.sma_indicator(close, window=w)
        out[f"sma_{w}"] = _safe_div(close, sma) - 1  # normalized distance from MA
    ema12 = ta.trend.ema_indicator(close, window=12)
    ema26 = ta.trend.ema_indicator(close, window=26)
    out["ema_12_26_spread"] = _safe_div(ema12, ema26) - 1

    macd = ta.trend.MACD(close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    adx = ta.trend.ADXIndicator(high, low, close)
    out["adx"] = adx.adx()
    out["adx_pos"] = adx.adx_pos()
    out["adx_neg"] = adx.adx_neg()

    ichi = ta.trend.IchimokuIndicator(high, low)
    out["ichimoku_conv"] = _safe_div(close, ichi.ichimoku_conversion_line()) - 1
    out["ichimoku_base"] = _safe_div(close, ichi.ichimoku_base_line()) - 1

    # --- Momentum oscillators ---
    out["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    out["rsi_7"] = ta.momentum.RSIIndicator(close, window=7).rsi()
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()
    out["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
    out["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc()
    out["tsi"] = ta.momentum.TSIIndicator(close).tsi()
    out["awesome_osc"] = ta.momentum.AwesomeOscillatorIndicator(high, low).awesome_oscillator()

    # --- Volatility ---
    bb = ta.volatility.BollingerBands(close)
    out["bb_pctb"] = bb.bollinger_pband()
    out["bb_width"] = bb.bollinger_wband()
    atr = ta.volatility.AverageTrueRange(high, low, close)
    out["atr_pct"] = _safe_div(atr.average_true_range(), close)
    out["realized_vol_10"] = out["log_ret_1d"].rolling(10).std() * np.sqrt(252)
    out["realized_vol_20"] = out["log_ret_1d"].rolling(20).std() * np.sqrt(252)
    keltner = ta.volatility.KeltnerChannel(high, low, close)
    out["keltner_pct"] = _safe_div(close - keltner.keltner_channel_lband(),
                                    (keltner.keltner_channel_hband() - keltner.keltner_channel_lband()).replace(0, np.nan))

    # --- Volume ---
    out["obv"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    out["obv_chg_10"] = out["obv"].pct_change(10)
    out["mfi"] = ta.volume.MFIIndicator(high, low, close, vol).money_flow_index()
    out["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, vol).chaikin_money_flow()
    out["vwap_dist"] = _safe_div(close, ta.volume.VolumeWeightedAveragePrice(high, low, close, vol).volume_weighted_average_price()) - 1
    out["vol_zscore_20"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()

    # --- Regime / structure ---
    out["dist_52w_high"] = _safe_div(close, close.rolling(252, min_periods=20).max()) - 1
    out["dist_52w_low"] = _safe_div(close, close.rolling(252, min_periods=20).min()) - 1
    out["trend_strength_50"] = _safe_div(close.rolling(50).mean(), close.rolling(200, min_periods=50).mean()) - 1

    # Clean up infinities produced by near-zero denominators
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


FEATURE_COLUMNS = None  # populated dynamically after build_features() at train time
