// ═══════════════════════════════════════════════════════
//  Upload & Run pipeline
// ═══════════════════════════════════════════════════════
let selectedFile = null;

const fileInput = document.getElementById('fileInput');
const dropzone  = document.getElementById('dropzone');
const runBtn    = document.getElementById('runBtn');
const origPrev  = document.getElementById('origPrev');

// ── File selection ───────────────────────────────────────────
if (fileInput) {
  fileInput.addEventListener('change', e => handleFile(e.target.files[0]));
}

if (dropzone) {
  dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.classList.add('drag');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('drag');
    handleFile(e.dataTransfer.files[0]);
  });
}

function handleFile(f) {
  if (!f) return;
  selectedFile = f;
  if (origPrev) origPrev.src = URL.createObjectURL(f);
  const prevWrap = document.getElementById('prevWrap');
  if (prevWrap) prevWrap.style.display = 'block';
  if (runBtn) runBtn.disabled = false;
  showInfo(`${f.name} · ${(f.size / 1024).toFixed(1)} KB`);
}

// ── Run pipeline ─────────────────────────────────────────────
if (runBtn) {
  runBtn.addEventListener('click', async () => {
    if (!selectedFile) { showErr('Please select an image first.'); return; }
    setLoading(true);
    clearAlerts();
    clearLog();

    const fd = new FormData();
    fd.append('file',          selectedFile);
    fd.append('model',         getVal('model',         'cnn'));
    fd.append('enhancement',   getVal('enhancement',   'clahe'));
    fd.append('normalization', getVal('normalization', 'minmax'));
    fd.append('aug_flip',      isChecked('aug_flip')   ? '1' : '0');
    fd.append('aug_rotate',    isChecked('aug_rotate') ? '1' : '0');
    fd.append('aug_zoom',      isChecked('aug_zoom')   ? '1' : '0');

    try {
      const r = await fetch(API + '/api/predict', { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok) { showErr(d.error || 'Server error'); return; }
      if (typeof renderResults === 'function') renderResults(d);
    } catch (e) {
      showErr('Network error: ' + e.message);
    } finally {
      setLoading(false);
    }
  });
}

// ── Helpers ──────────────────────────────────────────────────
function getVal(id, fallback) {
  const el = document.getElementById(id);
  return el ? el.value : fallback;
}
function isChecked(id) {
  const el = document.getElementById(id);
  return !!(el && el.checked);
}
function setLoading(on) {
  if (!runBtn) return;
  runBtn.disabled = on;
  runBtn.classList.toggle('loading', on);
}
function showErr(msg) {
  const el = document.getElementById('errAlert');
  if (!el) return;
  el.textContent = '⚠ ' + msg;
  el.style.display = 'block';
}
function showInfo(msg) {
  const el = document.getElementById('infoAlert');
  if (!el) return;
  el.textContent = msg;
  el.style.display = 'block';
}
function clearAlerts() {
  const e1 = document.getElementById('errAlert');
  const e2 = document.getElementById('infoAlert');
  if (e1) e1.style.display = 'none';
  if (e2) e2.style.display = 'none';
}
function clearLog() {
  const lp = document.getElementById('logPanel');
  if (!lp) return;
  lp.innerHTML = '';
  lp.style.display = 'none';
}