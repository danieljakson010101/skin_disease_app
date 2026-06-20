// ═══════════════════════════════════════════════════════
//  Training
// ═══════════════════════════════════════════════════════
let trainJobId = null;
let trainPoll  = null;

document.getElementById('trainBtn').addEventListener('click', async () => {
  const payload = {
    model:         document.getElementById('model').value,
    enhancement:   document.getElementById('enhancement').value,
    normalization: document.getElementById('normalization').value,
  };

  const btn = document.getElementById('trainBtn');
  btn.disabled    = true;
  btn.textContent = 'Training in progress…';

  document.getElementById('trainAlert').style.display = 'none';
  document.getElementById('trainLog').style.display   = 'block';
  document.getElementById('trainLog').innerHTML       = '';
  document.getElementById('trainBar').style.width     = '4%';

  try {
    const r = await fetch(API + '/api/train', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const d  = await r.json();
    trainJobId = d.job_id;
    pollTrain();
  } catch (e) {
    btn.disabled    = false;
    btn.textContent = 'Train Model on Dataset';
    showTrainMsg('Network error: ' + e.message);
  }
});

function pollTrain() {
  trainPoll = setInterval(async () => {
    try {
      const d = await (await fetch(API + '/api/train/status/' + trainJobId)).json();

      if (d.log && d.log.length) {
        const tl = document.getElementById('trainLog');
        tl.innerHTML   = d.log.map(([s, m]) => `<div class="tl ${s}">${m}</div>`).join('');
        tl.scrollTop   = tl.scrollHeight;
        document.getElementById('trainBar').style.width =
          Math.min(90, 4 + d.log.length * 4) + '%';
      }

      if (d.status === 'done') {
        clearInterval(trainPoll);
        document.getElementById('trainBar').style.width = '100%';
        const btn       = document.getElementById('trainBtn');
        btn.disabled    = false;
        btn.textContent = 'Re-train Model';
        showTrainMsg(
          `Model trained — Acc: ${d.metrics?.acc}%  F1: ${d.metrics?.f1}%`,
          'info'
        );
        checkStatus();

      } else if (d.status === 'error') {
        clearInterval(trainPoll);
        document.getElementById('trainBar').style.width = '0%';
        const btn       = document.getElementById('trainBtn');
        btn.disabled    = false;
        btn.textContent = 'Train Model on Dataset';
        showTrainMsg(d.message || 'Training failed. Check that dataset/train/ has images.');
      }
    } catch (_) {}
  }, 1200);
}

function showTrainMsg(msg, type = 'warn') {
  const el      = document.getElementById('trainAlert');
  el.textContent = msg;
  el.className   = 'alert ' + type;
  el.style.display = 'block';
}