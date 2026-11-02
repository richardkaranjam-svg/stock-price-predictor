const form = document.querySelector('#prediction-form');
const tickerInput = document.querySelector('#ticker');
const button = document.querySelector('#predict-button');
const status = document.querySelector('#form-status');
const resultCard = document.querySelector('#result-card');
const autoStart = document.querySelector('#auto-start');
const autoStop = document.querySelector('#auto-stop');
const autoStatus = document.querySelector('#auto-status');

const currency = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
});

function formatNewsSentiment(value) {
  if (value == null) return 'neutral';
  if (value > 0.05) return 'positive';
  if (value < -0.05) return 'negative';
  return 'neutral';
}

// Fetch and display model metadata dynamically
async function loadModelMetadata() {
  try {
    const resp = await fetch('/model/metadata');
    if (!resp.ok) return;
    const meta = await resp.json();
    const versionBadge = document.getElementById('model-version-badge');
    if (versionBadge) versionBadge.textContent = meta.model_version || 'V3+';
    const footerBar = document.getElementById('footer-bar-info');
    if (footerBar) {
      const horizon = meta.horizon_bars || 12;
      footerBar.textContent = 'Bar Interval: 20 minutes · Horizon: ' + horizon +
        ' bars · Training: ' + (meta.training_bars || 'N/A') + ' bars';
    }
    const footerModel = document.getElementById('footer-model-info');
    if (footerModel) {
      const tp = meta.tp_atr_mult || 1.5;
      const sl = meta.sl_atr_mult || 0.75;
      footerModel.textContent = 'LightGBM + RF ' + (meta.model_version || 'V3+') +
        ' · ' + tp + '×ATR TP / ' + sl + '×ATR SL · 1R Ratchet · Educational use only';
    }
  } catch (e) { /* silently ignore */ }
}
loadModelMetadata();

function renderResult(data) {
  let changeClass = '';
  let expectedText = 'expected to chop/sideways';

  if (data.predicted_class === 1) {
    changeClass = ' positive';
    expectedText = 'expected UP';
  } else if (data.predicted_class === -1) {
    changeClass = ' negative';
    expectedText = 'expected DOWN';
  }

  const confidencePct = (data.confidence * 100).toFixed(1);
  const modelVersion = data.model_version || 'V2';
  const horizonBars = data.horizon_bars || 12;

  const dirAccuracy = data.metrics && data.metrics.directional_accuracy != null
    ? (data.metrics.directional_accuracy * 100).toFixed(1) + '%'
    : 'Run training';

  const tpLabel = data.tp_atr_mult
    ? data.tp_atr_mult + '×ATR TP'
    : '+0.4% TP';
  const slLabel = data.sl_atr_mult
    ? data.sl_atr_mult + '×ATR SL'
    : '-0.2% SL';

  const newsNote = data.news_used_by_model
    ? 'News signal used: ' + formatNewsSentiment(data.news && data.news.sentiment)
    : 'News shown for context only; retrain with historical data to use it.';

  // Market closed banner
  let marketNotice = '';
  if (data.market_closed) {
    const reason = (data.market_status && data.market_status.reason) || 'US market is currently closed';
    marketNotice = '<div style="background:rgba(245,166,35,0.12);border:1px solid #f5a623;border-radius:6px;padding:0.6rem 0.85rem;margin-bottom:1rem;color:#f5a623;font-size:0.85rem;line-height:1.4;">' +
      '⏸ <strong>Market Closed</strong>: ' + reason + '.<br>' +
      '<span style="color:var(--text-muted);font-size:0.8rem;">Displaying institutional forecast based on latest closed market bars.</span>' +
      '</div>';
  }

  // V3+ model staleness warning
  let stalenessNote = '';
  if (data.model_status && data.model_status.retrain_recommended_in_cloud) {
    stalenessNote = '<p class="news-note" style="color:#f5a623;">⚠ Model is ' +
      (data.model_status.model_age_days || '?') +
      ' days old. Consider retraining in Colab.</p>';
  }

  // ATR barrier info
  let atrInfo = '';
  if (data.atr_at_entry != null) {
    atrInfo = '<div><span class="meta-label">ATR at Entry</span><span class="meta-value">$' +
      data.atr_at_entry.toFixed(2) + '</span></div>';
  }
  if (data.take_profit_price != null && data.stop_loss_price != null) {
    atrInfo += '<div><span class="meta-label">TP / SL</span><span class="meta-value">' +
      currency.format(data.take_profit_price) + ' / ' +
      currency.format(data.stop_loss_price) + '</span></div>';
  }
  if (data.ev_atr_value != null) {
    atrInfo += '<div><span class="meta-label">Net EV (ATR)</span><span class="meta-value">' +
      data.ev_atr_value.toFixed(4) + '</span></div>';
  }

  resultCard.innerHTML =
    '<div class="result-content">' +
      marketNotice +
      '<p class="result-label">Model forecast <span style="opacity:0.5;font-size:0.8em;">' + modelVersion + '</span></p>' +
      '<h2 class="result-ticker">' + data.ticker + ' <span aria-hidden="true">/</span> next ' + horizonBars + ' bars</h2>' +
      '<p class="result-price">' + currency.format(data.current_price) + '</p>' +
      '<p class="result-change' + changeClass + '">' + expectedText + ' (' + confidencePct + '% confidence)</p>' +
      '<div class="result-meta">' +
        '<div><span class="meta-label">Current price</span><span class="meta-value">' + currency.format(data.current_price) + '</span></div>' +
        '<div><span class="meta-label">Horizon</span><span class="meta-value">' + horizonBars + ' bars</span></div>' +
        '<div><span class="meta-label">Model Engine</span><span class="meta-value">LightGBM + RF ' + modelVersion + '</span></div>' +
        '<div><span class="meta-label">Risk Guard</span><span class="meta-value">' + tpLabel + ' / ' + slLabel + '</span></div>' +
        '<div><span class="meta-label">Direction hit rate</span><span class="meta-value">' + dirAccuracy + '</span></div>' +
        '<div><span class="meta-label">Training bars</span><span class="meta-value">' + (data.training_bars || 'N/A') + '</span></div>' +
        atrInfo +
        '<div><span class="meta-label">Current headlines</span><span class="meta-value">' + ((data.news && data.news.headline_count) || 0) + ' found</span></div>' +
      '</div>' +
      '<p class="news-note">' + newsNote + '</p>' +
      stalenessNote +
      '<div class="paper-controls">' +
        '<button type="button" class="paper-button buy" data-side="buy" data-ticker="' + data.ticker + '">Paper buy 1 share</button>' +
        '<button type="button" class="paper-button sell" data-side="sell" data-ticker="' + data.ticker + '">Paper sell 1 share</button>' +
      '</div>' +
      '<p class="paper-status" role="status" aria-live="polite"></p>' +
    '</div>';
}

async function submitPaperOrder(btn) {
  const controls = btn.closest('.result-content');
  const paperStatus = controls.querySelector('.paper-status');
  controls.querySelectorAll('.paper-button').forEach(function(item) { item.disabled = true; });
  paperStatus.textContent = 'Submitting to paper account...';
  try {
    const response = await fetch('/paper/order', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker: btn.dataset.ticker, side: btn.dataset.side, quantity: 1 }),
    });
    const responseText = await response.text();
    let data;
    try {
      data = JSON.parse(responseText);
    } catch (e) {
      throw new Error('Paper broker returned an unexpected response (' + response.status + ').');
    }
    if (!response.ok) throw new Error(data.error || 'Paper order failed.');
    paperStatus.textContent = data.side.toUpperCase() + ' order ' + data.status + ': ' + data.quantity + ' share of ' + data.ticker + '.';
  } catch (error) {
    paperStatus.textContent = error.message;
  } finally {
    controls.querySelectorAll('.paper-button').forEach(function(item) { item.disabled = false; });
  }
}

resultCard.addEventListener('click', function(event) {
  const btn = event.target.closest('.paper-button');
  if (btn) submitPaperOrder(btn);
});

async function setAutoTrading(path, message) {
  autoStart.disabled = true;
  autoStop.disabled = true;
  try {
    const response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker: tickerInput.value.trim().toUpperCase() || 'AAPL' }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not update autonomous mode.');
    autoStatus.textContent = data.running
      ? message + ' Trading ' + data.ticker + ' in paper mode.'
      : 'Autonomous mode stopped.';
  } catch (error) {
    autoStatus.textContent = error.message;
  } finally {
    refreshAutoStatus();
  }
}

autoStart.addEventListener('click', function() { setAutoTrading('/paper/auto/start', 'Model control enabled.'); });
autoStop.addEventListener('click', function() { setAutoTrading('/paper/auto/stop', ''); });

async function refreshAutoStatus() {
  try {
    const response = await fetch('/paper/auto/status');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Status unavailable.');

    var decisionStr = '';
    if (data.running && data.latest_decision) {
      var act = data.latest_decision.action.toUpperCase();
      var rsn = data.latest_decision.reason || data.latest_decision.error || '';
      decisionStr = ' | LAST ' + act + (rsn ? ' (' + rsn + ')' : '');
    }

    var perfStr = '';
    if (data.performance) {
      var sign = data.performance.profit_loss >= 0 ? '+' : '';
      perfStr = ' | P/L ' + sign + currency.format(data.performance.profit_loss) +
                ' | equity ' + currency.format(data.performance.equity);
    }

    var intervalStr = data.interval_seconds < 60
      ? data.interval_seconds + ' seconds'
      : (data.interval_seconds / 60) + ' minutes';

    autoStatus.textContent = data.running
      ? 'RUNNING - checking ' + data.ticker + ' every ' + intervalStr + decisionStr + perfStr + '.'
      : 'STOPPED - autonomous trading is idle.' + perfStr;

    autoStatus.classList.toggle('running', data.running);
    autoStatus.classList.toggle('stopped', !data.running);
    autoStart.disabled = data.running;
    autoStop.disabled = !data.running;
  } catch (error) {
    autoStatus.textContent = 'Status unavailable: ' + error.message;
    autoStatus.classList.remove('running', 'stopped');
  }
}

refreshAutoStatus();
let autoRefreshTimer = setInterval(refreshAutoStatus, 15000);

const chartSymbolBadge = document.querySelector('#chart-symbol-badge');

function updateLiveChart(symbol) {
  if (!symbol) return;
  const cleanTicker = symbol.trim().toUpperCase();
  if (chartSymbolBadge) chartSymbolBadge.textContent = cleanTicker;

  const oldScript = document.getElementById('tv-widget-script');
  if (oldScript) oldScript.remove();

  const chartDiv = document.getElementById('tradingview_chart');
  if (!chartDiv) return;
  chartDiv.innerHTML = '';

  const script = document.createElement('script');
  script.id = 'tv-widget-script';
  script.type = 'text/javascript';
  script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
  script.async = true;
  script.innerHTML = JSON.stringify({
    autosize: true,
    symbol: cleanTicker,
    interval: 'D',
    timezone: 'Etc/UTC',
    theme: 'dark',
    style: '1',
    locale: 'en',
    allow_symbol_change: true,
    calendar: false,
    support_host: 'https://www.tradingview.com'
  });
  chartDiv.appendChild(script);
}

form.addEventListener('submit', async function(event) {
  event.preventDefault();
  const ticker = tickerInput.value.trim().toUpperCase();
  if (!ticker) return;

  updateLiveChart(ticker);

  button.disabled = true;
  resultCard.setAttribute('aria-busy', 'true');
  status.textContent = 'Pulling latest ' + ticker + ' market data...';

  try {
    const response = await fetch('/predict?ticker=' + encodeURIComponent(ticker));
    const data = await response.json();

    if (!response.ok) throw new Error(data.error || 'The forecast could not be generated.');
    renderResult(data);
    resultCard.setAttribute('aria-busy', 'false');
    if (data.market_closed) {
      status.textContent = 'Forecast loaded (Market is currently closed — showing latest close data).';
    } else {
      status.textContent = 'Forecast updated for ' + ticker + '.';
    }
  } catch (error) {
    resultCard.setAttribute('aria-busy', 'false');
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

// Load the default AAPL chart immediately on page open
updateLiveChart(tickerInput.value.trim() || 'AAPL');

const shutdownBtn = document.querySelector('#server-shutdown-btn');
if (shutdownBtn) {
  shutdownBtn.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to stop the background server?')) return;
    if (autoRefreshTimer) {
      clearInterval(autoRefreshTimer);
      autoRefreshTimer = null;
    }
    shutdownBtn.disabled = true;
    shutdownBtn.textContent = 'Stopping...';
    try {
      await fetch('/shutdown', { method: 'POST' });
      document.body.innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100vh;font-family:sans-serif;flex-direction:column;"><h2>Server stopped</h2><p style="color:#666;">The background server has been closed. You can restart it using your desktop shortcut.</p></div>';
    } catch (e) {
      document.body.innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100vh;font-family:sans-serif;flex-direction:column;"><h2>Server stopped</h2><p style="color:#666;">The background server has been closed.</p></div>';
    }
  });
}


// Fetch and render recent signals from SQLite
async function loadRecentSignals() {
  const container = document.getElementById('signals-table-container');
  const badge = document.getElementById('signals-count-badge');
  if (!container) return;
  try {
    const res = await fetch('/paper/signals?limit=10');
    if (!res.ok) return;
    const data = await res.json();
    if (badge) badge.textContent = `${data.count} signals logged`;
    if (!data.signals || data.signals.length === 0) {
      container.innerHTML = '<p style="color:var(--text-muted); padding:4px 0;">No signals recorded yet.</p>';
      return;
    }
    let html = '<table style="width:100%; border-collapse:collapse; text-align:left;">' +
      '<thead><tr style="border-bottom:1px solid #374151; color:#9ca3af;">' +
      '<th style="padding:4px;">Time</th><th>Side</th><th>Price</th><th>Conf</th><th>Hash</th>' +
      '</tr></thead><tbody>';
    for (const s of data.signals) {
      const timeStr = s.timestamp_utc ? s.timestamp_utc.substring(11, 19) : '--:--:--';
      const sideColor = s.side === 'long' ? '#34d399' : (s.side === 'short' ? '#f87171' : '#9ca3af');
      const hashShort = s.features_hash ? s.features_hash.substring(0, 8) + '...' : '-';
      const sideText = s.side ? s.side.toUpperCase() : '-';
      html += `<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
        <td style="padding:4px;">${timeStr}</td>
        <td style="color:${sideColor}; font-weight:bold;">${sideText}</td>
        <td>$${s.price ? Number(s.price).toFixed(2) : '-'}</td>
        <td>${s.confidence != null ? (Number(s.confidence) * 100).toFixed(0) : '-'}%</td>
        <td style="color:#6b7280;" title="${s.features_hash || ''}">${hashShort}</td>
      </tr>`;
    }
    html += '</tbody></table>';
    container.innerHTML = html;
  } catch (e) {
    container.innerHTML = '<p style="color:var(--text-muted); padding:4px 0;">Signals feed idle.</p>';
  }
}

loadRecentSignals();
setInterval(loadRecentSignals, 15000);
