// ═══════════════════════════════════════════════════════
//  results.js — renders prediction output into results.html
// ═══════════════════════════════════════════════════════

/**
 * Called by upload.js after a successful /api/predict response.
 * d = { prediction, confidence, top5, metrics, log, mode,
 *       dataset_present, model_trained, charts:{...}, preprocessed_img }
 */
function renderResults(d) {
  document.getElementById('resultsEmpty').style.display = 'none';
  document.getElementById('resultsBody').style.display  = 'block';

  renderLog(d.log);
  renderPredictCard(d);
  renderTop5(d.top5);
  renderMetrics(d.metrics);
  renderCharts(d.charts);
}

// ── Pipeline log ─────────────────────────────────────────────
function renderLog(log) {
  const panel = document.getElementById('pipelineLog');
  if (!log || !log.length) { panel.style.display = 'none'; return; }

  panel.innerHTML = log.map(([status, msg]) => {
    const icon = status === 'err' ? '✗' : '✓';
    const cls  = status === 'err' ? 'log-line err' : 'log-line ok';
    return `<div class="${cls}"><span class="log-ico">${icon}</span>${escapeHtml(msg)}</div>`;
  }).join('');
  panel.style.display = 'block';
}

// ── Prediction card ──────────────────────────────────────────
function renderPredictCard(d) {
  const modeEl = document.getElementById('pcMode');
  modeEl.textContent = d.mode === 'real' ? 'Real model prediction' : 'Simulated (demo) prediction';
  modeEl.classList.toggle('mode-real', d.mode === 'real');
  modeEl.classList.toggle('mode-sim',  d.mode !== 'real');

  document.getElementById('pcPrediction').textContent = d.prediction;
  document.getElementById('pcConfNum').textContent = d.confidence + '%';

  const ring = document.getElementById('pcRing');
  ring.style.setProperty('--pct', d.confidence);

  const img = document.getElementById('pcProcessedImg');
  if (d.preprocessed_img) {
    img.src = 'data:image/png;base64,' + d.preprocessed_img;
  }
}

// ── Top 5 list ───────────────────────────────────────────────
function renderTop5(top5) {
  const list = document.getElementById('top5List');
  if (!top5 || !top5.length) { list.innerHTML = ''; return; }

  list.innerHTML = top5.map(item => {
    const pct = (item.prob * 100).toFixed(1);
    return `
      <div class="top5-row">
        <span class="t5-name">${escapeHtml(item.name)}</span>
        <div class="t5-bar-track">
          <div class="t5-bar-fill" style="width:${pct}%"></div>
        </div>
        <span class="t5-pct">${pct}%</span>
      </div>`;
  }).join('');
}

// ── Metrics ──────────────────────────────────────────────────
function renderMetrics(m) {
  if (!m) return;
  document.getElementById('mAcc').textContent  = m.acc  + '%';
  document.getElementById('mPrec').textContent = m.prec + '%';
  document.getElementById('mRec').textContent  = m.rec  + '%';
  document.getElementById('mF1').textContent   = m.f1   + '%';
}

// ── Charts ───────────────────────────────────────────────────
function renderCharts(charts) {
  if (!charts) return;
  const map = {
    distribution:    'chartDistInline', // not used directly, dataset chart uses cDist
    pixel:           'chartPixel',
    model_compare:   'chartModelCompare',
    preproc_compare: 'chartPreprocCompare',
    training:        'chartTraining',
  };
  Object.entries(map).forEach(([key, id]) => {
    const el = document.getElementById(id);
    if (el && charts[key]) {
      el.src = 'data:image/png;base64,' + charts[key];
    }
  });
  // Dataset distribution chart can refresh too (same source as /api/distribution)
  const distEl = document.getElementById('cDist');
  if (distEl && charts.distribution) {
    distEl.src = 'data:image/png;base64,' + charts.distribution;
  }
}

// ── Util ─────────────────────────────────────────────────────
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}