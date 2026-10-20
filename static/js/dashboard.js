const form = document.getElementById('analyzeForm');
const tickerInput = document.getElementById('tickerInput');
const periodSelect = document.getElementById('periodSelect');
const costInput = document.getElementById('costInput');
const capitalInput = document.getElementById('capitalInput');
const confInput = document.getElementById('confInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingBlock = document.getElementById('loadingBlock');
const loadingText = document.getElementById('loadingText');
const errorBlock = document.getElementById('errorBlock');
const resultsRoot = document.getElementById('resultsRoot');

let priceChart, equityChart;
let pollTimer = null;

// ---- clock ----
function tickClock() {
  const el = document.getElementById('clock');
  const now = new Date();
  el.textContent = now.toTimeString().slice(0, 8) + ' local';
}
setInterval(tickClock, 1000);
tickClock();

// ---- loading message rotation ----
const LOADING_MESSAGES = [
  'Pulling price history…',
  'Engineering technical indicators…',
  'Fitting stacked ensemble on walk-forward windows…',
  'Training meta-labeling confidence gate…',
  'Pricing in transaction costs & slippage…',
  'Scoring news sentiment…',
  'Assembling the readout…',
];
let loadingInterval;
function startLoadingMessages() {
  let i = 0;
  loadingText.textContent = LOADING_MESSAGES[0];
  loadingInterval = setInterval(() => {
    i = (i + 1) % LOADING_MESSAGES.length;
    loadingText.textContent = LOADING_MESSAGES[i];
  }, 1400);
}
function stopLoadingMessages() {
  clearInterval(loadingInterval);
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  runAnalysis();
});

async function runAnalysis() {
  const ticker = tickerInput.value.trim().toUpperCase() || 'AAPL';
  const period = periodSelect.value;
  const cost = costInput.value || 10;
  const capital = capitalInput.value || 100000;
  const conf = confInput.value || 0.55;

  errorBlock.hidden = true;
  resultsRoot.hidden = true;
  loadingBlock.hidden = false;
  analyzeBtn.disabled = true;
  startLoadingMessages();
  if (pollTimer) clearTimeout(pollTimer);

  const qs = new URLSearchParams({
    ticker, period,
    transaction_cost_bps: cost,
    starting_capital: capital,
    confidence: conf,
  }).toString();

  try {
    // Job-queue flow: submit returns immediately, then poll status. This is
    // what lets two people (or two tickers) run analyses at the same time
    // instead of one request blocking the next behind it.
    const submitRes = await fetch(`/api/analyze/submit?${qs}`, { method: 'POST' });
    const submitData = await submitRes.json();
    if (!submitRes.ok) throw new Error(submitData.error || 'Could not queue analysis.');

    const result = await pollJob(submitData.job_id);
    render(result);
  } catch (err) {
    errorBlock.textContent = `⚠ ${err.message}`;
    errorBlock.hidden = false;
  } finally {
    stopLoadingMessages();
    loadingBlock.hidden = true;
    analyzeBtn.disabled = false;
  }
}

function pollJob(jobId, intervalMs = 1500, timeoutMs = 180000) {
  const startedAt = Date.now();
  return new Promise((resolve, reject) => {
    async function tick() {
      try {
        const res = await fetch(`/api/analyze/status/${jobId}`);
        const data = await res.json();
        if (!res.ok) return reject(new Error(data.error || 'Job lookup failed.'));

        if (data.status === 'done') return resolve(data.result);
        if (data.status === 'error') return reject(new Error(data.result?.error || 'Analysis failed.'));

        if (Date.now() - startedAt > timeoutMs) return reject(new Error('Analysis timed out.'));
        pollTimer = setTimeout(tick, intervalMs);
      } catch (e) {
        reject(e);
      }
    }
    tick();
  });
}

function render(data) {
  resultsRoot.hidden = false;

  // data source banner
  const banner = document.getElementById('dataBanner');
  if (data.data_source === 'synthetic-demo') {
    banner.innerHTML = `<span class="flag-synthetic">● demo mode</span> — no live route to a market data feed from this environment, so this run uses a regime-switching synthetic price series for ${data.ticker} instead of real quotes. The full pipeline (features → stacked ensemble → meta-labeling → walk-forward backtest) is running for real; only the input prices are simulated. Point <code>src/data_pipeline.py</code> at a live feed (yfinance works out of the box with normal internet access) to get real results.`;
  } else {
    const cachedTag = data.cached ? ' · served from cache' : '';
    banner.innerHTML = `<span class="flag-live">● live data</span> — ${data.ticker}, ${data.period} history, ${data.price.dates.length} most recent sessions shown. Computed in ${data.compute_seconds}s${cachedTag}.`;
  }

  renderSignal(data);
  renderMetrics(data.backtest);
  renderPriceChart(data);
  renderEquityChart(data.backtest);
  renderGauges(data.indicators);
  renderImportance(data.feature_importance);
  renderSentiment(data.sentiment);
  renderExecution(data.backtest);
  renderFundamentals(data.fundamentals);

  document.getElementById('priceMeta').textContent =
    `last close $${data.last_price} · ${data.horizon_days}-day prediction horizon`;
}

function renderSignal(data) {
  document.getElementById('sigTicker').textContent = data.ticker;
  const sig = data.current_signal;
  const dirEl = document.getElementById('sigDirection');
  const probEl = document.getElementById('sigProb');
  const fill = document.getElementById('confidenceFill');
  const confVal = document.getElementById('confidenceVal');
  const actionable = document.getElementById('actionableTag');

  if (sig.error) {
    dirEl.textContent = 'N/A';
    dirEl.className = 'signal-big';
    probEl.textContent = sig.error;
    fill.style.width = '0%';
    confVal.textContent = '—';
    actionable.textContent = '';
    return;
  }

  const isUp = sig.direction === 1;
  dirEl.textContent = isUp ? '▲ UP' : '▼ DOWN';
  dirEl.className = 'signal-big ' + (isUp ? 'up' : 'down');
  probEl.textContent = `${(sig.up_probability * 100).toFixed(1)}%`;
  fill.style.width = `${Math.round(sig.trade_confidence * 100)}%`;
  confVal.textContent = `${(sig.trade_confidence * 100).toFixed(1)}%`;
  actionable.textContent = sig.actionable
    ? 'above threshold — model would take this trade'
    : 'below threshold — model would sit this one out';
  actionable.style.color = sig.actionable ? 'var(--positive)' : 'var(--text-faint)';
}

function fmtPct(x) { return x === null || x === undefined ? '—' : (x * 100).toFixed(1) + '%'; }
function fmtNum(x) { return x === null || x === undefined ? '—' : x.toFixed(2); }
function fmtUsd(x) { return x === null || x === undefined ? '—' : '$' + Number(x).toLocaleString(undefined, { maximumFractionDigits: 0 }); }

function renderMetrics(bt) {
  const grid = document.getElementById('metricsGrid');
  if (bt.error) {
    grid.innerHTML = `<div class="metric-cell"><div class="metric-label">backtest</div><div class="metric-value">${bt.error}</div></div>`;
    return;
  }
  const cells = [
    { label: 'sharpe (net of costs)', value: fmtNum(bt.sharpe_ratio), cls: cls(bt.sharpe_ratio) },
    { label: 'sortino (net)', value: fmtNum(bt.sortino_ratio), cls: cls(bt.sortino_ratio) },
    { label: 'directional accuracy', value: fmtPct(bt.directional_accuracy), cls: cls((bt.directional_accuracy ?? 0.5) - 0.5) },
    { label: 'win rate', value: fmtPct(bt.win_rate), cls: cls((bt.win_rate ?? 0.5) - 0.5) },
    { label: 'max drawdown', value: fmtPct(bt.max_drawdown), cls: 'neg' },
    { label: 'strategy return (net)', value: fmtPct(bt.total_return), cls: cls(bt.total_return) },
    { label: 'buy & hold return', value: fmtPct(bt.buy_hold_return), cls: cls(bt.buy_hold_return) },
    { label: 'trade participation', value: fmtPct(bt.trade_participation_rate), cls: '' },
    { label: 'periods tested', value: bt.n_periods_tested, cls: '' },
  ];
  grid.innerHTML = cells.map(c => `
    <div class="metric-cell">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value ${c.cls}">${c.value}</div>
    </div>
  `).join('');
}
function cls(v) {
  if (v === null || v === undefined || isNaN(v)) return '';
  return v > 0 ? 'pos' : (v < 0 ? 'neg' : '');
}

function renderExecution(bt) {
  const meta = document.getElementById('executionMeta');
  const grid = document.getElementById('executionGrid');
  if (bt.error) { grid.innerHTML = ''; meta.textContent = ''; return; }

  meta.textContent = bt.ran_parallel
    ? `${bt.windows_evaluated} windows, fit in parallel`
    : `${bt.windows_evaluated} windows, sequential${bt.windows_evaluated > 1 ? ' (single core available)' : ''}`;

  const items = [
    { label: 'sharpe, before costs', value: fmtNum(bt.sharpe_ratio_gross) },
    { label: 'return, before costs', value: fmtPct(bt.total_return_gross) },
    { label: 'cost drag on return', value: fmtPct(bt.cost_drag) },
    { label: `cost per trade (bps)`, value: bt.transaction_cost_bps },
    { label: 'starting capital', value: fmtUsd(bt.starting_capital) },
    { label: 'ending capital (net)', value: fmtUsd(bt.ending_capital) },
    { label: 'total costs paid', value: fmtUsd(bt.total_cost_paid_dollars) },
    { label: 'trades taken', value: bt.n_trades_taken },
  ];
  grid.innerHTML = items.map(i => `
    <div class="gauge">
      <div class="gauge-label">${i.label}</div>
      <div class="gauge-value">${i.value === null || i.value === undefined ? '—' : i.value}</div>
    </div>
  `).join('');
}

function renderFundamentals(f) {
  const grid = document.getElementById('fundamentalsGrid');
  const note = document.getElementById('fundamentalsNote');
  if (!f || !f.available) {
    grid.innerHTML = '';
    note.textContent = f?.note || 'Fundamentals unavailable.';
    return;
  }
  note.textContent = 'Shown for context only — never fed into the model (see src/fundamentals.py for why).';
  const fmtCap = (v) => v ? '$' + (v / 1e9).toFixed(1) + 'B' : '—';
  const items = [
    { label: 'name', value: f.long_name || '—' },
    { label: 'sector', value: f.sector || '—' },
    { label: 'industry', value: f.industry || '—' },
    { label: 'market cap', value: fmtCap(f.market_cap) },
    { label: 'trailing P/E', value: f.trailing_pe ? f.trailing_pe.toFixed(1) : '—' },
    { label: 'forward P/E', value: f.forward_pe ? f.forward_pe.toFixed(1) : '—' },
    { label: 'dividend yield', value: f.dividend_yield ? (f.dividend_yield * 100).toFixed(2) + '%' : '—' },
    { label: 'beta', value: f.beta ? f.beta.toFixed(2) : '—' },
  ];
  grid.innerHTML = items.map(i => `
    <div class="gauge">
      <div class="gauge-label">${i.label}</div>
      <div class="gauge-value" style="font-size:14px">${i.value}</div>
    </div>
  `).join('');
}

function renderPriceChart(data) {
  const ctx = document.getElementById('priceChart');
  if (priceChart) priceChart.destroy();
  priceChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.price.dates,
      datasets: [
        { label: 'close', data: data.price.close, borderColor: '#E8A33D', borderWidth: 1.6, pointRadius: 0, tension: 0.15 },
        { label: 'sma 20', data: data.price.sma_20, borderColor: '#3DDC97', borderWidth: 1, pointRadius: 0, tension: 0.1 },
        { label: 'sma 50', data: data.price.sma_50, borderColor: '#7C8494', borderWidth: 1, pointRadius: 0, tension: 0.1 },
      ],
    },
    options: chartOptions(true),
  });
}

function renderEquityChart(bt) {
  const ctx = document.getElementById('equityChart');
  if (equityChart) equityChart.destroy();
  if (bt.error || !bt.equity_curve) return;
  const labels = bt.equity_curve.map((_, i) => i);
  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'strategy (net)', data: bt.equity_curve, borderColor: '#E8A33D', borderWidth: 1.8, pointRadius: 0 },
        { label: 'strategy (gross)', data: bt.gross_equity_curve, borderColor: '#C9862A', borderWidth: 1, pointRadius: 0, borderDash: [4, 3] },
        { label: 'buy & hold', data: bt.benchmark_curve, borderColor: '#4C5566', borderWidth: 1.2, pointRadius: 0 },
      ],
    },
    options: chartOptions(false),
  });
}

function chartOptions(showX) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { intersect: false, mode: 'index' },
    plugins: {
      legend: { labels: { color: '#7C8494', font: { family: 'IBM Plex Mono', size: 11 }, boxWidth: 12 } },
      tooltip: { backgroundColor: '#161D27', borderColor: '#232A35', borderWidth: 1, titleColor: '#E7E9EC', bodyColor: '#E7E9EC', bodyFont: { family: 'IBM Plex Mono', size: 11 } },
    },
    scales: {
      x: { display: showX, ticks: { color: '#4C5566', maxTicksLimit: 8, font: { family: 'IBM Plex Mono', size: 10 } }, grid: { color: '#1A212B' } },
      y: { ticks: { color: '#4C5566', font: { family: 'IBM Plex Mono', size: 10 } }, grid: { color: '#1A212B' } },
    },
  };
}

function renderGauges(ind) {
  const grid = document.getElementById('gaugeGrid');
  const items = [
    { label: 'RSI (14)', value: ind.rsi_14 },
    { label: 'MACD histogram', value: ind.macd_hist },
    { label: 'ADX (trend strength)', value: ind.adx },
    { label: 'Bollinger %B', value: ind.bb_pctb },
    { label: 'ATR (% of price)', value: ind.atr_pct !== null ? ind.atr_pct + '%' : null },
    { label: '20d realized vol (ann.)', value: ind.realized_vol_20 !== null ? ind.realized_vol_20 + '%' : null },
  ];
  grid.innerHTML = items.map(i => `
    <div class="gauge">
      <div class="gauge-label">${i.label}</div>
      <div class="gauge-value">${i.value === null || i.value === undefined ? '—' : i.value}</div>
    </div>
  `).join('');
}

function renderImportance(items) {
  const wrap = document.getElementById('importanceBars');
  if (!items || items.length === 0) {
    wrap.innerHTML = '<div class="panel-note">Not enough data to compute feature importance.</div>';
    return;
  }
  const max = Math.max(...items.map(i => i.importance));
  wrap.innerHTML = items.map(i => `
    <div class="bar-row">
      <div class="bar-name">${i.feature}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(i.importance / max * 100).toFixed(1)}%"></div></div>
      <div class="bar-pct">${(i.importance * 100).toFixed(1)}%</div>
    </div>
  `).join('');
}

function renderSentiment(s) {
  document.getElementById('sentimentEngine').textContent = s.engine || '';
  const body = document.getElementById('sentimentBody');
  const avg = s.avg_score ?? 0;
  const cls = avg > 0.1 ? 'pos' : (avg < -0.1 ? 'neg' : 'neu');
  const label = avg > 0.1 ? 'net positive' : (avg < -0.1 ? 'net negative' : 'net neutral');
  body.innerHTML = `
    <div class="sentiment-score-row">
      <div class="sentiment-score ${cls}">${avg.toFixed(2)}</div>
      <div class="panel-meta">${label}, scored −1 to +1 across recent headlines</div>
    </div>
    ${s.note ? `<div class="sentiment-note">${s.note}</div>` : ''}
    ${!s.note ? `<div class="sentiment-note">Scored with a pretrained financial-language transformer (FinBERT), used as an additional model feature alongside the technical indicators.</div>` : ''}
  `;
}

// run once on load with default ticker
window.addEventListener('DOMContentLoaded', () => {
  runAnalysis();
});
