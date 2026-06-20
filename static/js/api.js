// ═══════════════════════════════════════════════════════
//  API base — points to Railway backend
// ═══════════════════════════════════════════════════════
const API = "";

// ═══════════════════════════════════════════════════════
//  Status check
// ═══════════════════════════════════════════════════════
async function checkStatus() {
  try {
    const d = await (await fetch(API + '/api/status')).json();

    const setDot = (id, ok) => {
      document.getElementById(id).className = 'sdot ' + (ok ? 'g' : 'r');
    };

    setDot('sdDataset', d.dataset);
    document.getElementById('slDataset').textContent =
      d.dataset ? 'Dataset found' : 'No dataset (demo)';

    ['CNN', 'SVM', 'RF', 'KNN'].forEach(m => {
      const k  = m.toLowerCase();
      const ok = !!d.trained[k];
      setDot('sd' + m, ok);
      document.getElementById('sl' + m).textContent =
        ok ? `${m} · F1 ${d.trained[k].f1}%` : `${m} · not trained`;
    });
  } catch (_) {}
}

// Run on page load
checkStatus();

// Pre-load distribution chart on page load
window.addEventListener('load', async () => {
  try {
    const d = await (await fetch(API + '/api/distribution')).json();
    if (d.chart) {
      document.getElementById('cDist').src = 'data:image/png;base64,' + d.chart;
    }
  } catch (_) {}
});