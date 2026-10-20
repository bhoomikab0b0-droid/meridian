# Meridian — Quantitative Signal Terminal

An upgraded stock direction/prediction pipeline and dashboard, built on top of
the idea in your original resume bullet (RandomForest + Stacking + meta-labeling,
Flask dashboard) but with a deeper ensemble, a real confidence-gating layer,
a pretrained financial-language model for sentiment, and a from-scratch
"quant terminal" UI.

## What's actually different from the original system

| | Original | This version |
|---|---|---|
| Base learners | Random Forest | XGBoost + LightGBM + Random Forest + Extra Trees |
| Stacking | Stacking Regressor | Proper leak-free stacking via walk-forward out-of-fold predictions → logistic meta-learner |
| Meta-labeling | mentioned, not detailed | Full López de Prado triple-barrier labeling + a dedicated LightGBM confidence gate that decides whether a signal is worth acting on, and sizes the position by that confidence |
| Sentiment | none | Pretrained FinBERT (`ProsusAI/finbert`) financial-language transformer scores headlines as an added feature, with a lexicon-based fallback if the weights can't be downloaded |
| Backtest | unspecified | Walk-forward (rolling retrain, only ever scores unseen future bars) — not a single train/test split, so the numbers it reports are an honest out-of-sample estimate |
| Metrics reported | directional accuracy, Sharpe | + Sortino, Calmar, win rate, profit factor, max drawdown, trade participation rate |
| UI | Flask dashboard | Custom-designed dark "quant terminal" dashboard: live equity curve vs. buy & hold, indicator gauges, feature-importance bars, an architecture diagram of the model itself, sentiment panel |

## Important — read before you trust any number this prints

Directional accuracy and Sharpe ratio are **empirical facts about a specific
dataset**, not properties of an algorithm. I did not fabricate the 58–62%
accuracy / 1.10–1.25 Sharpe range from your original bullet, and I'm not
going to hand you fabricated "better" numbers either — I built the improved
pipeline and it computes real metrics from whatever data you point it at.

Two things follow from that:

1. **Run it on real data before citing any metric anywhere** (a resume, a
   report, a pitch). The number you get depends on the ticker, the date
   range, the labeling parameters (horizon, barrier width), and the
   confidence threshold you choose. Don't reuse a number from one ticker/
   period as if it's a general property of the system.
2. **This sandbox could not reach Yahoo Finance or Hugging Face** (its
   network egress is locked to package registries only), so everything I
   personally ran here used a synthetic, regime-switching price generator
   as a stand-in for real quotes — clearly labeled `synthetic-demo` in the
   UI whenever it's active. On that synthetic data the honest walk-forward
   result came back near a coin flip (accuracy ~37–50%, negative Sharpe),
   which is *expected and correct*: the synthetic series has randomly
   injected regime shifts with no genuine leading signal in the features,
   so a well-behaved model shouldn't be able to predict it well. It's proof
   the backtest isn't silently overfitting, not a verdict on the real
   system's real-market performance.

Once you run this with `yfinance` pointed at real tickers (works immediately
outside this sandbox), you'll get real, citable numbers. Try a handful of
liquid, trending large-caps and a handful of choppy/sideways ones — the
model's edge (if any) will look very different across regimes, which is
itself worth reporting honestly.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5050`.

- `transformers` + `torch` are optional. Without them (or without a route
  to huggingface.co), sentiment scoring automatically falls back to a small
  finance-word lexicon and says so in the UI. First FinBERT run downloads
  ~440MB of weights, cached afterward.
- First analysis of a ticker fetches and caches price history for 30
  minutes (`cache/`); full pipeline results are cached for 20 minutes
  (`cache/results/`) since a full walk-forward backtest is compute-heavy
  (typically 15–30s per ticker on a laptop CPU).

## Tuning knobs worth knowing about

All accepted as query params (or JSON body on the async endpoint):

- `horizon` (default 10) — prediction horizon in trading days
- `confidence` (default 0.55) — minimum meta-labeling confidence to take a
  trade at all; raising this trades less often but (usually) more
  selectively
- `period` — how much history to pull (`2y`/`3y`/`5y`/`10y`; shorter than
  ~2 years doesn't leave enough bars for the 200-day-SMA feature's warm-up
  plus a walk-forward train/test split, so it's rejected with an
  explanatory error rather than silently degrading that indicator)
- `transaction_cost_bps` (default 10) — round-trip commission + slippage
  charge per trade, in basis points
- `starting_capital` (default 100000) — used to turn the % return series
  into a dollar P&L trace (`ending_capital`, `total_cost_paid_dollars`)
- `parallel` (default true) — fit walk-forward windows in separate
  processes when more than one CPU core is available

Barrier width (`up_mult`/`down_mult` in `src/labeling.py`, default 1.5×ATR
each) controls how far price has to move to count as a "win" — wider
barriers mean fewer, larger, higher-conviction labels; narrower barriers
mean more frequent, noisier ones.

## API

- `GET /api/analyze?ticker=AAPL&period=3y&...` — synchronous, blocks until
  the pipeline finishes. Fine for scripts/curl; the dashboard doesn't use
  it directly.
- `POST /api/analyze/submit?ticker=AAPL&...` — enqueues the analysis, returns
  `{"job_id": "...", "status": "queued", "queue_workers": N}` immediately.
- `GET /api/analyze/status/<job_id>` — returns `{"status": "queued" |
  "running" | "done" | "error", "result": {...}}` (`result` present once
  `done`/`error`). Poll this until `status` is terminal.
- `GET /health` — `{"status": "ok", "active_jobs": N, "job_workers": N}`

## Project layout

```
app.py                  Flask routes + job queue
src/data_pipeline.py     yfinance fetch + cache + synthetic offline fallback
src/features.py          59 engineered technical indicators
src/labeling.py          triple-barrier + meta-labeling
src/model.py             4-model stack -> logistic meta-learner -> confidence gate
src/backtest.py          walk-forward backtest, parallel windows, costs/capital
src/sentiment.py         pretrained FinBERT + lexicon fallback
src/fundamentals.py      informational company fundamentals (not a model input)
templates/index.html     dashboard markup
static/css/style.css     "quant terminal" design system
static/js/dashboard.js   async job submission + polling, Chart.js rendering
```

## Honest limitations / next steps

- No live intraday data or options flow — end-of-day OHLCV + sentiment +
  informational fundamentals only.
- ~~No fundamentals~~ **Added**: `src/fundamentals.py` pulls sector, market
  cap, P/E, beta, etc. via yfinance and shows them as read-only context on
  the dashboard. They're deliberately **not** fed into the model — today's
  P/E attached to historical training rows would leak information the
  model never actually had on those past dates. Using fundamentals in the
  model properly needs a point-in-time feed, which is a bigger data lift.
- The synthetic fallback is for demoing the pipeline mechanics only; it
  has no claim to resembling real market microstructure.
- ~~Walk-forward retraining is CPU-only and not parallelized~~ **Added**:
  each rolling window is now independent and fit in a separate process via
  `ProcessPoolExecutor` (`src/backtest.py`, `parallel=True` by default).
  Each worker process caps its own base learners to 1 thread so N windows
  in N processes don't oversubscribe the machine. On a single-core box
  this correctly falls back to running sequentially — you'll see
  `"ran_parallel": false` in the response, which is expected, not a bug.
- ~~For many tickers at once you'd want a job queue~~ **Added**: a small
  in-process job queue (`ThreadPoolExecutor` + an in-memory job table).
  `POST /api/analyze/submit` returns a `job_id` immediately;
  `GET /api/analyze/status/<job_id>` is polled for progress and the final
  result. The dashboard UI uses this by default so two tickers (or two
  people) can run analyses concurrently instead of blocking each other.
  This is a single-process queue, not Celery/RQ — it won't survive a
  process restart or scale across machines. If you need that, swap
  `_executor`/`_jobs` in `app.py` for a real broker; the request/response
  shape (`job_id` in, `{status, result}` out) is designed to make that
  swap mostly mechanical.
- ~~No transaction costs/slippage/capital constraints~~ **Added**:
  `src/backtest.py` now charges a per-trade basis-point cost (commission +
  estimated slippage combined, configurable, default 10bps round-trip)
  proportional to position size, and tracks a dollar P&L against a
  configurable starting capital. The dashboard shows gross vs. net Sharpe,
  the return "cost drag," and ending capital side by side so the effect of
  costs is visible rather than hidden in a single blended number. This is
  still a simplified model — no bid/ask spread dynamics, no market-impact
  scaling with order size, no borrow cost for short positions — so treat
  it as a more honest approximation, not a guarantee of real fill prices.
- This remains a research/portfolio tool, not a trading system. It does
  not place real orders, has no risk management beyond the confidence gate
  and position sizing described above, and live results will still likely
  be worse than the backtest (regime change, real slippage exceeding the
  assumed bps, latency, etc.).
