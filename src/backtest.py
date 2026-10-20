"""
Walk-forward backtest.

Unlike a single train/test split, this retrains the stacked model on a
rolling/expanding window and only ever predicts on unseen future bars,
which gives an honest (non-overfit) estimate of directional accuracy and
risk-adjusted return. Position size is scaled by the meta-labeling
confidence gate, and trades below a minimum confidence threshold are
skipped entirely -- this is what lets the strategy trade less often but
more accurately, improving the Sharpe ratio relative to "trade every bar".

Two things this version adds on top of a plain backtest:

1. Transaction costs, slippage, and a starting-capital P&L trace, so the
   reported numbers reflect what a real account would have experienced
   rather than a frictionless paper return. This is applied per trade as
   a single round-trip basis-point charge (commission + estimated
   slippage combined) -- simple, but it means "net" figures are always
   worse than (or equal to) the "gross" ones, never the other way around.
2. Each rolling window is independent of the others (it only depends on
   history up to its own train slice), so the windows are fit in
   parallel via a process pool when more than one CPU core is available.
   On a single-core sandbox this transparently falls back to running
   sequentially -- correctness doesn't depend on core count, only speed.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .model import StackedDirectionalModel

DEFAULT_TRANSACTION_COST_BPS = 10.0  # 0.10% round-trip: commission + estimated slippage
DEFAULT_STARTING_CAPITAL = 100_000.0


def _run_one_window(features, labels, forward_returns, start, train_window, test_window,
                     confidence_threshold, n_jobs_per_model):
    """
    Fit on [start-train_window, start), predict on [start, start+test_window).
    Pure function of its inputs (no shared state) so it's safe to run in a
    separate process. Returns None if the window can't be fit.
    """
    train_slice = slice(max(0, start - train_window), start)
    test_slice = slice(start, start + test_window)

    X_train = features.iloc[train_slice]
    y_train = labels.iloc[train_slice]
    if X_train.notna().all(axis=1).sum() < 100:
        return None

    model = StackedDirectionalModel(n_jobs=n_jobs_per_model)
    try:
        model.fit(X_train, y_train)
    except Exception:
        return None

    X_test = features.iloc[test_slice]
    fwd_test = forward_returns.iloc[test_slice]
    valid_test = X_test.notna().all(axis=1) & fwd_test.notna()
    if valid_test.sum() == 0:
        return None

    preds = model.predict(X_test.loc[valid_test])
    fwd_ret = fwd_test.loc[valid_test]

    direction = preds["direction"]
    confidence = preds["trade_confidence"]
    position = np.where(confidence >= confidence_threshold, direction * confidence, 0.0)

    rows = []
    for i, date in enumerate(fwd_ret.index):
        rows.append({
            "date": str(date.date()) if hasattr(date, "date") else str(date),
            "direction": int(direction[i]),
            "confidence": float(confidence[i]),
            "position": float(position[i]),
            "actual_return": float(fwd_ret.values[i]),
        })
    return rows


def walk_forward_backtest(
    features: pd.DataFrame,
    labels: pd.Series,
    forward_returns: pd.Series,
    train_window: int = 500,
    test_window: int = 60,
    step: int = 60,
    confidence_threshold: float = 0.55,
    trading_days_per_year: int = 252,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
    parallel: bool = True,
) -> dict:
    n = len(features)

    # Adaptive fallback: if the requested window sizes don't fit at all
    # (e.g. a shorter history than expected), shrink train_window first,
    # then test_window, down to sane floors, rather than just erroring out.
    tw, sw = train_window, test_window
    starts = []
    for _ in range(6):
        start = tw
        starts = []
        step_used = sw  # step shrinks in lockstep with test_window on retries
        while start + sw <= n:
            starts.append(start)
            start += step_used
        if starts:
            break
        if tw > 120:
            tw = max(120, tw - 60)
        elif sw > 15:
            sw = max(15, sw - 10)
        else:
            break
    train_window, test_window = tw, sw

    if not starts:
        return {"error": "Not enough data to backtest with the given window sizes -- try a longer history period."}

    max_workers = min(len(starts), os.cpu_count() or 1)
    predictions_log = []
    ran_parallel = False

    if parallel and max_workers > 1:
        # Each base learner is capped to 1 internal thread so that running
        # several windows in separate processes doesn't oversubscribe the
        # machine (N processes x N internal threads each).
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(_run_one_window, features, labels, forward_returns, s,
                              train_window, test_window, confidence_threshold, 1): s
                    for s in starts
                }
                results_by_start = {}
                for fut in as_completed(futures):
                    s = futures[fut]
                    try:
                        results_by_start[s] = fut.result()
                    except Exception:
                        results_by_start[s] = None
            for s in starts:  # preserve chronological order regardless of completion order
                rows = results_by_start.get(s)
                if rows:
                    predictions_log.extend(rows)
            ran_parallel = True
        except Exception:
            # Process pool unavailable in this environment (e.g. restricted
            # sandbox) -- fall back to sequential execution transparently.
            predictions_log = []

    if not ran_parallel:
        for s in starts:
            rows = _run_one_window(features, labels, forward_returns, s,
                                    train_window, test_window, confidence_threshold, -1)
            if rows:
                predictions_log.extend(rows)

    if not predictions_log:
        return {"error": "No window produced a usable prediction -- try a longer history or smaller windows."}

    result = _compute_metrics(predictions_log, trading_days_per_year, transaction_cost_bps, starting_capital)
    result["ran_parallel"] = ran_parallel
    result["windows_evaluated"] = len(starts)
    return result


def _compute_metrics(predictions_log, tdpy, cost_bps, starting_capital) -> dict:
    positions = np.array([p["position"] for p in predictions_log])
    directions = np.array([p["direction"] for p in predictions_log])
    actual_returns = np.array([p["actual_return"] for p in predictions_log])

    finite = np.isfinite(positions) & np.isfinite(actual_returns)
    positions, directions, actual_returns = positions[finite], directions[finite], actual_returns[finite]
    predictions_log = [p for p, keep in zip(predictions_log, finite) if keep]

    if len(positions) == 0:
        return {"error": "Not enough data to backtest with the given window sizes."}

    gross_strat_returns = positions * actual_returns
    bench_returns = actual_returns  # buy & hold: always fully invested

    # --- Transaction costs & slippage ---
    # Charged once per trade, proportional to position size (a smaller,
    # lower-confidence position also risks less capital and so pays less
    # cost -- consistent with sizing by confidence rather than trading
    # a fixed lot every time).
    cost_rate = cost_bps / 10_000.0
    trade_cost = np.abs(positions) * cost_rate
    net_strat_returns = gross_strat_returns - trade_cost

    traded_mask = positions != 0
    n_traded = int(traded_mask.sum())

    if n_traded > 0:
        traded_dir = directions[traded_mask]
        traded_actual = actual_returns[traded_mask]
        directional_accuracy = float((np.sign(traded_dir) == np.sign(traded_actual)).mean())
    else:
        directional_accuracy = float("nan")

    def _sharpe(rets):
        mean_ret = rets.mean()
        std_ret = rets.std(ddof=1) if len(rets) > 1 else np.nan
        return (mean_ret / std_ret) * np.sqrt(tdpy) if std_ret and std_ret > 0 else float("nan")

    def _sortino(rets):
        mean_ret = rets.mean()
        downside = rets[rets < 0]
        downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
        return (mean_ret / downside_std) * np.sqrt(tdpy) if downside_std and downside_std > 0 else float("nan")

    sharpe_net = _sharpe(net_strat_returns)
    sharpe_gross = _sharpe(gross_strat_returns)
    sortino_net = _sortino(net_strat_returns)

    equity_curve = np.cumprod(1 + net_strat_returns)
    gross_equity_curve = np.cumprod(1 + gross_strat_returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / running_max - 1
    max_drawdown = float(drawdown.min())

    bench_equity = np.cumprod(1 + bench_returns)

    wins = net_strat_returns[net_strat_returns > 0]
    losses = net_strat_returns[net_strat_returns < 0]
    win_rate = float(len(wins) / n_traded) if n_traded > 0 else float("nan")
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("nan")

    total_return_net = float(equity_curve[-1] - 1) if len(equity_curve) else float("nan")
    total_return_gross = float(gross_equity_curve[-1] - 1) if len(gross_equity_curve) else float("nan")
    bench_total_return = float(bench_equity[-1] - 1) if len(bench_equity) else float("nan")
    total_cost_drag = (total_return_gross - total_return_net) if (total_return_gross == total_return_gross and total_return_net == total_return_net) else None

    calmar = (total_return_net / abs(max_drawdown)) if max_drawdown != 0 else float("nan")

    ending_capital = starting_capital * (1 + total_return_net) if total_return_net == total_return_net else None
    total_cost_paid_dollars = float(trade_cost.sum() * starting_capital) if starting_capital else None

    return {
        "sharpe_ratio": round(sharpe_net, 3) if sharpe_net == sharpe_net else None,
        "sharpe_ratio_gross": round(sharpe_gross, 3) if sharpe_gross == sharpe_gross else None,
        "sortino_ratio": round(sortino_net, 3) if sortino_net == sortino_net else None,
        "calmar_ratio": round(calmar, 3) if calmar == calmar else None,
        "directional_accuracy": round(directional_accuracy, 4) if directional_accuracy == directional_accuracy else None,
        "win_rate": round(win_rate, 4) if win_rate == win_rate else None,
        "profit_factor": round(profit_factor, 3) if profit_factor == profit_factor else None,
        "max_drawdown": round(max_drawdown, 4),
        "total_return": round(total_return_net, 4) if total_return_net == total_return_net else None,
        "total_return_gross": round(total_return_gross, 4) if total_return_gross == total_return_gross else None,
        "cost_drag": round(total_cost_drag, 4) if total_cost_drag is not None else None,
        "buy_hold_return": round(bench_total_return, 4) if bench_total_return == bench_total_return else None,
        "transaction_cost_bps": cost_bps,
        "starting_capital": starting_capital,
        "ending_capital": round(ending_capital, 2) if ending_capital is not None else None,
        "total_cost_paid_dollars": round(total_cost_paid_dollars, 2) if total_cost_paid_dollars is not None else None,
        "n_periods_tested": int(len(net_strat_returns)),
        "n_trades_taken": int(n_traded),
        "trade_participation_rate": round(float(n_traded / len(net_strat_returns)), 4),
        "equity_curve": equity_curve.tolist(),
        "gross_equity_curve": gross_equity_curve.tolist(),
        "benchmark_curve": bench_equity.tolist(),
        "predictions_log": predictions_log[-120:],  # cap payload size for the UI
    }
