import sys
import os
import time
import json
import uuid
import hashlib
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
import ta

from src.data_pipeline import fetch_ohlcv
from src.features import build_features
from src.labeling import triple_barrier_labels
from src.model import StackedDirectionalModel
from src.backtest import walk_forward_backtest, DEFAULT_TRANSACTION_COST_BPS, DEFAULT_STARTING_CAPITAL
from src.sentiment import score_headlines
from src.fundamentals import fetch_fundamentals

app = Flask(__name__)

RESULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "results")
os.makedirs(RESULT_CACHE_DIR, exist_ok=True)
RESULT_CACHE_TTL = 60 * 20  # 20 minutes -- analysis is compute-heavy, so cache full responses

DEFAULT_HEADLINES_BY_TICKER = {
    "default": [
        "Company reports quarterly earnings in line with analyst expectations",
        "Shares steady as management reaffirms full-year guidance",
        "Analysts split on outlook amid mixed sector signals",
    ]
}

# ----------------------------------------------------------------------
# Lightweight in-process job queue.
#
# A single walk-forward analysis takes ~15-30s of CPU-bound work. Handling
# that synchronously inside one Flask request means a second visitor asking
# about a different ticker just queues up behind the first with the dev
# server's default single worker. Instead, /api/analyze/submit hands the
# work to a small thread pool and returns immediately with a job id; the
# client polls /api/analyze/status/<job_id> for progress and the final
# result. This is not a substitute for a real broker (Celery + Redis, RQ,
# etc.) if you need this to survive process restarts or scale across
# machines -- but it removes the "one ticker blocks everyone else" problem
# with zero extra infrastructure, which covers most single-instance
# portfolio/demo deployments.
# ----------------------------------------------------------------------
JOB_WORKERS = max(2, (os.cpu_count() or 1))
_executor = ThreadPoolExecutor(max_workers=JOB_WORKERS, thread_name_prefix="analysis")
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
JOB_TTL_SECONDS = 60 * 30


def _result_cache_path(ticker, period, horizon, confidence, cost_bps, capital):
    key = hashlib.md5(f"{ticker}_{period}_{horizon}_{confidence}_{cost_bps}_{capital}".encode()).hexdigest()
    return os.path.join(RESULT_CACHE_DIR, f"{key}.json")


def _run_analysis(params: dict) -> dict:
    """The actual analysis pipeline. Pure function of `params` -> response
    dict (or a dict with an "error" key), safe to call from a worker thread."""
    ticker = params["ticker"]
    period = params["period"]
    horizon = params["horizon"]
    conf_threshold = params["confidence"]
    cost_bps = params["transaction_cost_bps"]
    starting_capital = params["starting_capital"]
    parallel_windows = params["parallel"]

    t0 = time.time()

    cache_path = _result_cache_path(ticker, period, horizon, conf_threshold, cost_bps, starting_capital)
    if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path) < RESULT_CACHE_TTL):
        with open(cache_path) as f:
            cached = json.load(f)
        cached["cached"] = True
        return cached

    df, source = fetch_ohlcv(ticker, period=period)
    if df is None or len(df) < 400:
        return {"error": "Not enough price history for this ticker to build reliable features (needs roughly 2 years+ -- some indicators, like the 200-day moving average, need ~200 bars of warm-up before the walk-forward windows even start). Try a longer period."}

    feat = build_features(df)
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    labels_df = triple_barrier_labels(df["Close"], atr, horizon=horizon, up_mult=1.5, down_mult=1.5)
    fwd_ret = df["Close"].pct_change(horizon).shift(-horizon)

    # --- Walk-forward backtest (honest, non-overfit performance estimate),
    #     with per-trade transaction cost/slippage and a $ P&L trace, and
    #     windows fit in parallel processes when more than one core exists.
    # Window sizes scale with however much history we actually have -- a
    # fixed 500/60 split (fine for 3-5y of data) would leave 0 usable
    # windows for a 1Y pull, since a 200-bar warm-up (e.g. the 200d SMA
    # feature) plus a 500-bar train window doesn't fit in ~252 bars.
    n = len(feat)
    train_window = min(400, max(150, n // 2))
    test_window = max(30, n // 8)
    backtest_res = walk_forward_backtest(
        feat, labels_df["label"], fwd_ret,
        train_window=train_window, test_window=test_window, step=test_window,
        confidence_threshold=conf_threshold,
        transaction_cost_bps=cost_bps,
        starting_capital=starting_capital,
        parallel=parallel_windows,
    )

    # --- Final model trained on full history, for "today's" live signal ---
    live_model = StackedDirectionalModel()
    valid_mask = feat.notna().all(axis=1) & labels_df["label"].notna()
    live_model.fit(feat.loc[valid_mask], labels_df["label"].loc[valid_mask])

    latest_row = feat.iloc[[-1]]
    if latest_row.notna().all(axis=1).iloc[0]:
        live_pred = live_model.predict(latest_row)
        current_signal = {
            "direction": int(live_pred["direction"][0]),
            "up_probability": float(round(live_pred["up_probability"][0], 4)),
            "trade_confidence": float(round(live_pred["trade_confidence"][0], 4)),
            "model_agreement": float(round(live_pred["agreement"][0], 4)),
            "actionable": bool(live_pred["trade_confidence"][0] >= conf_threshold),
        }
    else:
        current_signal = {"error": "Most recent bar has missing indicator values (needs more warm-up history)."}

    importance = live_model.feature_importance(12)
    importance_payload = [
        {"feature": idx, "importance": round(float(row["avg"]), 4)}
        for idx, row in importance.iterrows()
    ] if not importance.empty else []

    headlines = DEFAULT_HEADLINES_BY_TICKER.get(ticker, DEFAULT_HEADLINES_BY_TICKER["default"])
    sentiment = score_headlines(headlines)
    fundamentals = fetch_fundamentals(ticker)

    price_payload = {
        "dates": [str(d.date()) for d in df.index[-250:]],
        "close": [round(float(x), 2) for x in df["Close"].iloc[-250:]],
        "volume": [int(x) for x in df["Volume"].iloc[-250:]],
        "sma_20": [None if pd.isna(x) else round(float(x), 2) for x in ta.trend.sma_indicator(df["Close"], 20).iloc[-250:]],
        "sma_50": [None if pd.isna(x) else round(float(x), 2) for x in ta.trend.sma_indicator(df["Close"], 50).iloc[-250:]],
    }

    indicator_payload = {
        "rsi_14": None if pd.isna(feat["rsi_14"].iloc[-1]) else round(float(feat["rsi_14"].iloc[-1]), 2),
        "macd_hist": None if pd.isna(feat["macd_hist"].iloc[-1]) else round(float(feat["macd_hist"].iloc[-1]), 4),
        "adx": None if pd.isna(feat["adx"].iloc[-1]) else round(float(feat["adx"].iloc[-1]), 2),
        "bb_pctb": None if pd.isna(feat["bb_pctb"].iloc[-1]) else round(float(feat["bb_pctb"].iloc[-1]), 3),
        "atr_pct": None if pd.isna(feat["atr_pct"].iloc[-1]) else round(float(feat["atr_pct"].iloc[-1]) * 100, 2),
        "realized_vol_20": None if pd.isna(feat["realized_vol_20"].iloc[-1]) else round(float(feat["realized_vol_20"].iloc[-1]) * 100, 2),
    }

    response = {
        "ticker": ticker,
        "data_source": source,
        "period": period,
        "horizon_days": horizon,
        "last_price": round(float(df["Close"].iloc[-1]), 2),
        "price": price_payload,
        "indicators": indicator_payload,
        "current_signal": current_signal,
        "backtest": backtest_res,
        "feature_importance": importance_payload,
        "sentiment": sentiment,
        "fundamentals": fundamentals,
        "compute_seconds": round(time.time() - t0, 1),
        "cached": False,
    }
    try:
        with open(cache_path, "w") as f:
            json.dump(response, f)
    except Exception:
        pass
    return response


def _parse_params(args) -> dict:
    return {
        "ticker": args.get("ticker", "AAPL").upper().strip(),
        "period": args.get("period", "3y"),
        "horizon": int(args.get("horizon", 10)),
        "confidence": float(args.get("confidence", 0.55)),
        "transaction_cost_bps": float(args.get("transaction_cost_bps", DEFAULT_TRANSACTION_COST_BPS)),
        "starting_capital": float(args.get("starting_capital", DEFAULT_STARTING_CAPITAL)),
        "parallel": str(args.get("parallel", "true")).lower() != "false",
    }


def _job_worker(job_id: str, params: dict):
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = time.time()
    try:
        result = _run_analysis(params)
        status = "error" if "error" in result else "done"
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {"error": f"Analysis failed: {str(e)}"}
        status = "error"
    with _jobs_lock:
        _jobs[job_id]["status"] = status
        _jobs[job_id]["result"] = result
        _jobs[job_id]["finished_at"] = time.time()


def _cleanup_old_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    with _jobs_lock:
        stale = [jid for jid, j in _jobs.items() if j.get("submitted_at", 0) < cutoff]
        for jid in stale:
            _jobs.pop(jid, None)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze")
def api_analyze():
    """Synchronous endpoint -- kept for simple/scripted use. Blocks until
    the full pipeline finishes. For the dashboard UI, prefer the async
    submit/status pair below so multiple tickers can run concurrently."""
    params = _parse_params(request.args)
    try:
        result = _run_analysis(params)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/analyze/submit", methods=["POST", "GET"])
def api_analyze_submit():
    """Enqueue an analysis job and return immediately with a job id. Accepts
    the same params as /api/analyze (query string or JSON body)."""
    args = request.get_json(silent=True) or request.args
    params = _parse_params(args)

    _cleanup_old_jobs()
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "submitted_at": time.time(), "params": params}
    _executor.submit(_job_worker, job_id, params)
    return jsonify({"job_id": job_id, "status": "queued", "queue_workers": JOB_WORKERS})


@app.route("/api/analyze/status/<job_id>")
def api_analyze_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown or expired job id."}), 404
    payload = {"status": job["status"]}
    if job["status"] in ("done", "error"):
        payload["result"] = job["result"]
    return jsonify(payload)


@app.route("/health")
def health():
    with _jobs_lock:
        active = sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))
    return jsonify({"status": "ok", "active_jobs": active, "job_workers": JOB_WORKERS})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True, use_reloader=False, threaded=True)
