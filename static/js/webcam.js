// ═══════════════════════════════════════════════════════
//  webcam.js — real-time simulated input stream (Step 4c)
//  Captures a frame every N seconds, runs it through the
//  same /api/predict pipeline, updates the live panel.
// ═══════════════════════════════════════════════════════

(function () {
  const video    = document.getElementById('webcamVideo');
  const canvas   = document.getElementById('webcamCanvas');
  const startBtn = document.getElementById('wcStartBtn');
  const stopBtn  = document.getElementById('wcStopBtn');
  const liveDot  = document.getElementById('wcLiveDot');
  const liveLbl  = document.getElementById('wcLiveLbl');
  const errAlert = document.getElementById('wcErrAlert');
  const streamLog= document.getElementById('wcStreamLog');
  const intervalSelect = document.getElementById('wcInterval');

  if (!video || !startBtn) return; // webcam.html not on this page

  let mediaStream = null;
  let loopHandle  = null;
  let busy        = false; // avoid overlapping requests if a predict call is slow

  startBtn.addEventListener('click', startCamera);
  stopBtn.addEventListener('click', stopCamera);

  async function startCamera() {
    hideWcErr();
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      });
      video.srcObject = mediaStream;
      setLiveState(true);
      startBtn.disabled = true;
      stopBtn.disabled  = false;

      const seconds = parseInt(intervalSelect ? intervalSelect.value : '3', 10);
      logStream('ok', `Camera started — capturing every ${seconds}s`);
      loopHandle = setInterval(captureAndPredict, seconds * 1000);
      // Capture one immediately too
      captureAndPredict();
    } catch (e) {
      showWcErr('Could not access camera: ' + e.message);
    }
  }

  function stopCamera() {
    if (loopHandle) { clearInterval(loopHandle); loopHandle = null; }
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
    video.srcObject = null;
    setLiveState(false);
    startBtn.disabled = false;
    stopBtn.disabled  = true;
    logStream('ok', 'Camera stopped');
  }

  async function captureAndPredict() {
    if (busy || !mediaStream) return;
    if (!video.videoWidth) return; // not ready yet
    busy = true;

    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) { busy = false; return; }
      try {
        const fd = new FormData();
        fd.append('file', blob, 'frame.jpg');
        fd.append('model',         getVal('model',         'cnn'));
        fd.append('enhancement',   getVal('enhancement',   'clahe'));
        fd.append('normalization', getVal('normalization', 'minmax'));
        fd.append('aug_flip',   '0');
        fd.append('aug_rotate', '0');
        fd.append('aug_zoom',   '0');

        const r = await fetch(API + '/api/predict', { method: 'POST', body: fd });
        const d = await r.json();
        if (!r.ok) {
          logStream('err', d.error || 'Predict failed');
        } else {
          renderWebcamResult(d);
          logStream('ok', `${d.prediction} (${d.confidence}%) — ${d.mode}`);
        }
      } catch (e) {
        logStream('err', 'Network error: ' + e.message);
      } finally {
        busy = false;
      }
    }, 'image/jpeg', 0.85);
  }

  function renderWebcamResult(d) {
    const modeEl = document.getElementById('wcMode');
    if (modeEl) {
      modeEl.textContent = d.mode === 'real' ? 'Real model prediction' : 'Simulated (demo) prediction';
      modeEl.classList.toggle('mode-real', d.mode === 'real');
      modeEl.classList.toggle('mode-sim',  d.mode !== 'real');
    }
    setText('wcPrediction', d.prediction);
    setText('wcConfNum', d.confidence + '%');

    const ring = document.getElementById('wcRing');
    if (ring) ring.style.setProperty('--pct', d.confidence);

    const list = document.getElementById('wcTop5List');
    if (list && d.top5) {
      list.innerHTML = d.top5.map(item => {
        const pct = (item.prob * 100).toFixed(1);
        return `
          <div class="top5-row">
            <span class="t5-name">${escapeHtmlWc(item.name)}</span>
            <div class="t5-bar-track">
              <div class="t5-bar-fill" style="width:${pct}%"></div>
            </div>
            <span class="t5-pct">${pct}%</span>
          </div>`;
      }).join('');
    }
  }

  // ── Helpers ──────────────────────────────────────────────
  function setLiveState(on) {
    if (liveDot) liveDot.classList.toggle('live', on);
    if (liveLbl) liveLbl.textContent = on ? 'Live' : 'Camera off';
  }
  function showWcErr(msg) {
    if (!errAlert) return;
    errAlert.textContent = '⚠ ' + msg;
    errAlert.style.display = 'block';
  }
  function hideWcErr() {
    if (errAlert) errAlert.style.display = 'none';
  }
  function logStream(status, msg) {
    if (!streamLog) return;
    const icon = status === 'err' ? '✗' : '✓';
    const cls  = status === 'err' ? 'log-line err' : 'log-line ok';
    const time = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.className = cls;
    line.innerHTML = `<span class="log-ico">${icon}</span>[${time}] ${escapeHtmlWc(msg)}`;
    streamLog.prepend(line);
    while (streamLog.children.length > 30) streamLog.removeChild(streamLog.lastChild);
  }
  function getVal(id, fallback) {
    const el = document.getElementById(id);
    return el ? el.value : fallback;
  }
  function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }
  function escapeHtmlWc(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // Stop camera if user navigates away
  window.addEventListener('beforeunload', stopCamera);
})();