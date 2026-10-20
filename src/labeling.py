"""
Triple-barrier labeling + meta-labeling, following Marcos Lopez de Prado's
"Advances in Financial Machine Learning" framework.

Primary model:  predicts direction (up/down) of the next move.
Meta model:     predicts whether the primary model's call should be *acted on*
                 (i.e. trusted), which is what lets us size positions and
                 filter out low-conviction trades. This is the mechanism that
                 turns a coin-flip-adjacent directional model into a
                 risk-adjusted trading signal with a materially better Sharpe
                 ratio than trading every prediction blindly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.Series,
    atr: pd.Series,
    horizon: int = 10,
    up_mult: float = 2.0,
    down_mult: float = 2.0,
) -> pd.DataFrame:
    """
    For each bar t, look forward up to `horizon` bars. Label +1 if price hits
    the upper barrier (entry + up_mult*ATR) first, -1 if it hits the lower
    barrier first, 0 if neither barrier is touched before the vertical
    (time) barrier -- in which case the label is the sign of the terminal
    return.

    Returns a dataframe with columns: label, ret, touch_time
    """
    n = len(close)
    labels = np.zeros(n)
    rets = np.zeros(n)
    touch = np.full(n, np.nan)

    close_v = close.values
    atr_v = atr.values

    for i in range(n - 1):
        if np.isnan(atr_v[i]) or atr_v[i] == 0:
            labels[i] = np.nan
            continue
        entry = close_v[i]
        upper = entry + up_mult * atr_v[i]
        lower = entry - down_mult * atr_v[i]
        end = min(i + horizon, n - 1)
        window = close_v[i + 1: end + 1]
        if len(window) == 0:
            labels[i] = np.nan
            continue

        hit_up = np.where(window >= upper)[0]
        hit_down = np.where(window <= lower)[0]
        t_up = hit_up[0] if len(hit_up) else np.inf
        t_down = hit_down[0] if len(hit_down) else np.inf

        if t_up == np.inf and t_down == np.inf:
            terminal_ret = (window[-1] - entry) / entry
            labels[i] = 1 if terminal_ret > 0 else (-1 if terminal_ret < 0 else 0)
            rets[i] = terminal_ret
            touch[i] = len(window)
        elif t_up < t_down:
            labels[i] = 1
            rets[i] = (window[t_up] - entry) / entry
            touch[i] = t_up + 1
        else:
            labels[i] = -1
            rets[i] = (window[t_down] - entry) / entry
            touch[i] = t_down + 1

    labels[-1] = np.nan
    return pd.DataFrame({"label": labels, "barrier_ret": rets, "touch_time": touch}, index=close.index)


def meta_labels(primary_pred: pd.Series, true_label: pd.Series) -> pd.Series:
    """
    1 if the primary model's directional call agrees with the true sign
    (i.e. the trade *should* have been taken), else 0. Used to train the
    secondary "confidence gate" model.
    """
    agree = (np.sign(primary_pred) == np.sign(true_label)) & (true_label != 0)
    return agree.astype(int)
