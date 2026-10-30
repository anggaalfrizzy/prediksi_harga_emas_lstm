(() => {
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const css = getComputedStyle(document.documentElement);
const col = n => css.getPropertyValue(n).trim();
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const fmt = n => Number(n).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
const get = async u => { const r = await fetch(u); const j = await r.json(); if (!r.ok) throw new Error(j.message || r.status); return j; };
const banner = t => { const b = $('#banner'); b.textContent = t; b.hidden = false; };
const pick = (group, attr, val) => $$(group + ' button').forEach(b => b.classList.toggle('on', b.dataset[attr] === String(val)));
const hasChart = typeof Chart !== 'undefined';
if (hasChart) { Chart.defaults.font.family = "'Plus Jakarta Sans', system-ui, sans-serif"; Chart.defaults.color = col('--muted'); }

const opts = (extra = {}) => ({
  responsive: true, maintainAspectRatio: false, interaction: {mode: 'index', intersect: false},
  plugins: {legend: {display: false}, tooltip: {callbacks: {label: i => i.parsed.y == null ? null : `${i.dataset.label}: ${fmt(i.parsed.y)}`}}},
  scales: {
    x: {grid: {display: false}, ticks: {maxTicksLimit: 8, maxRotation: 0, callback(v) { const l = this.getLabelForValue(v); return this.chart.data.labels.length > 900 ? l.slice(0, 4) : l.slice(0, 7); }}},
    y: {grid: {color: col('--line')}, ticks: {callback: v => v.toLocaleString('en-US')}}
  }, ...extra});
const line = (label, data, color, more = {}) => ({label, data, borderColor: color, backgroundColor: color, borderWidth: 1.8, pointRadius: 0, pointHoverRadius: 4, tension: 0, ...more});
const fade = c => x => { const a = x.chart.chartArea; if (!a) return null; const g = x.chart.ctx.createLinearGradient(0, a.top, 0, a.bottom); g.addColorStop(0, c + '55'); g.addColorStop(1, c + '00'); return g; };

let H, hist, evalChart, evalData, fc;

function heroChart() {
  const step = Math.ceil(H.prices.length / 420);
  const L = H.dates.filter((_, i) => i % step === 0), D = H.prices.filter((_, i) => i % step === 0);
  const per = 2400 / D.length;
  const prevY = x => x.index === 0 ? x.chart.scales.y.getPixelForValue(D[0]) : x.chart.getDatasetMeta(x.datasetIndex).data[x.index - 1].getProjected('y');
  const anim = reduce ? false : {
    x: {type: 'number', easing: 'linear', duration: per, from: NaN, delay(x) { if (x.type !== 'data' || x.xStarted) return 0; x.xStarted = true; return x.index * per; }},
    y: {type: 'number', easing: 'linear', duration: per, from: prevY, delay(x) { if (x.type !== 'data' || x.yStarted) return 0; x.yStarted = true; return x.index * per; }}
  };
  new Chart($('#heroChart'), {type: 'line', data: {labels: L, datasets: [line('Harga emas', D, col('--amber'), {borderWidth: 2.6, fill: true, backgroundColor: fade(col('--amber'))})]}, options: opts({animation: anim})});
}

function setRange(key) {
  let i = 0;
  if (key !== 'all') {
    const d = new Date(H.dates.at(-1)); d.setFullYear(d.getFullYear() - Number(key));
    i = Math.max(0, H.dates.findIndex(x => x >= d.toISOString().slice(0, 10)));
  }
  const L = H.dates.slice(i), D = H.prices.slice(i);
  if (hist) { hist.data.labels = L; hist.data.datasets[0].data = D; hist.update(); }
  $('#sHigh').textContent = fmt(Math.max(...D)); $('#sLow').textContent = fmt(Math.min(...D));
  $('#sAvg').textContent = fmt(D.reduce((a, b) => a + b, 0) / D.length);
  const ch = (D.at(-1) / D[0] - 1) * 100;
  $('#sChg').textContent = (ch >= 0 ? '+' : '') + ch.toFixed(1) + '%';
  pick('#ranges', 'r', key);
}

function setEval(key) {
  if (!evalData || !evalChart) return;
  const n = key === 'all' ? evalData.dates.length : Number(key), s = evalData.dates.length - n;
  evalChart.data.labels = evalData.dates.slice(s);
  evalChart.data.datasets[0].data = evalData.actual.slice(s);
  evalChart.data.datasets[1].data = evalData.predicted.slice(s);
  evalChart.update(); pick('#evalRanges', 'e', key);
}

async function prediction() {
  try {
    const f = await get('/api/forecast?days=1');
    $('#pSetup').hidden = true;
    $('#pDate').textContent = f.dates[0]; $('#pValue').textContent = fmt(f.values[0]);
    const p = (f.values[0] / f.last_price - 1) * 100, chip = $('#pChip');
    chip.hidden = false; chip.className = 'chip ' + (p >= 0 ? 'up' : 'down');
    chip.textContent = (p >= 0 ? '+' : '') + p.toFixed(2) + '%';
    $('#pBase').textContent = `dari harga terakhir ${fmt(f.last_price)} (${f.last_date})`;
  } catch {
    $('#pSetup').hidden = false; $('#pValue').textContent = 'Belum ada';
    $('#pDate').textContent = 'hari kerja berikutnya'; $('#pBase').textContent = '';
  }
}

async function evaluation() {
  try {
    const {metrics: m, eval: e, history: h} = await get('/api/evaluation');
    $('#evalSetup').hidden = true; $('#evalBody').hidden = false;
    $('#mRmse').textContent = fmt(m.rmse); $('#mMae').textContent = fmt(m.mae);
    $('#mMape').textContent = m.mape.toFixed(2) + '%'; $('#mNaive').textContent = fmt(m.naive_rmse);
    $('#verdict').textContent = m.rmse < m.naive_rmse
      ? `Rata-rata meleset sekitar ${fmt(m.rmse)}, lebih kecil daripada tebakan sederhana "harga hari ini = harga kemarin" (${fmt(m.naive_rmse)}).`
      : `Rata-rata meleset sekitar ${fmt(m.rmse)}. Angka ini belum mengalahkan tebakan sederhana "harga hari ini = harga kemarin" (${fmt(m.naive_rmse)}). Hal ini umum pada harga pasar dan layak dibahas di laporan.`;
    evalData = e;
    if (hasChart) {
      evalChart = new Chart($('#evalChart'), {type: 'line', data: {labels: [], datasets: [line('Aktual', [], col('--amber'), {borderWidth: 2}), line('Prediksi', [], col('--indigo'), {borderWidth: 2})]}, options: opts()});
      setEval('all');
      if (h) new Chart($('#lossChart'), {type: 'line', data: {labels: h.loss.map((_, i) => i + 1), datasets: [line('loss', h.loss, col('--amber')), line('val_loss', h.val_loss, col('--indigo'))]},
        options: opts({plugins: {legend: {display: true}}, scales: {x: {title: {display: true, text: 'Epoch'}, grid: {display: false}}, y: {grid: {color: col('--line')}}}})});
    }
  } catch { $('#evalSetup').hidden = false; $('#evalBody').hidden = true; }
}

async function forecast(n = 10) {
  pick('#horizons', 'd', n);
  try {
    const f = await get('/api/forecast?days=' + n);
    $('#fcSetup').hidden = true; $('#fcBody').hidden = false;
    const hd = H.dates.slice(-90), hp = H.prices.slice(-90);
    const labels = [...hd, ...f.dates];
    const actual = [...hp, ...f.dates.map(() => null)];
    const pred = [...hd.slice(0, -1).map(() => null), hp.at(-1), ...f.values];
    if (hasChart) {
      if (!fc) fc = new Chart($('#fcChart'), {type: 'line', data: {labels, datasets: [line('Aktual', actual, col('--amber'), {borderWidth: 2}), line('Ramalan', pred, col('--mint'), {borderWidth: 2.4, borderDash: [5, 4]})]}, options: opts({animation: {duration: reduce ? 0 : 350}})});
      else { fc.data.labels = labels; fc.data.datasets[0].data = actual; fc.data.datasets[1].data = pred; fc.update(); }
    }
    $('#fcRows').innerHTML = f.dates.map((d, i) => {
      const p = (f.values[i] / f.last_price - 1) * 100;
      return `<tr><td>${d}</td><td class="num">${fmt(f.values[i])}</td><td class="num ${p >= 0 ? 'up' : 'down'}">${p >= 0 ? '+' : ''}${p.toFixed(2)}%</td></tr>`;
    }).join('');
    $('#fcMode').textContent = f.mode === 'live' ? 'Dihitung langsung oleh model LSTM.' : 'Ditampilkan dari hasil hitung yang tersimpan saat pelatihan (maksimal 30 hari).';
  } catch { $('#fcSetup').hidden = false; $('#fcBody').hidden = true; }
}

// Tombol dipasang lebih dulu, sebelum data dan grafik dimuat
$$('#ranges button').forEach(b => b.addEventListener('click', () => H && setRange(b.dataset.r)));
$$('#evalRanges button').forEach(b => b.addEventListener('click', () => setEval(b.dataset.e)));
$$('#horizons button').forEach(b => b.addEventListener('click', () => H && forecast(Number(b.dataset.d))));
$('#copyCmd').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText('python train.py'); $('#copyTxt').textContent = 'Tersalin'; } catch { $('#copyTxt').textContent = 'Salin manual'; }
  setTimeout(() => $('#copyTxt').textContent = 'Salin', 1800);
});

(async () => {
  if (location.protocol === 'file:') return banner('Halaman ini dibuka langsung dari file. Jalankan "python app.py", lalu buka http://127.0.0.1:5000');
  if (!hasChart) banner('Grafik gagal dimuat karena Chart.js tidak bisa diunduh. Periksa koneksi internet, lalu muat ulang.');
  try {
    const [s, h] = await Promise.all([get('/api/summary'), get('/api/history')]);
    H = h;
    $$('.unit').forEach(el => el.textContent = s.unit);
    $('#pBase').textContent = `Harga terakhir ${fmt(s.last_price)} (${s.last_date})`;
    $('#endDate').textContent = s.last_date;
    if (hasChart) {
      try { hist = new Chart($('#histChart'), {type: 'line', data: {labels: [], datasets: [line('Harga', [], col('--amber'), {borderWidth: 2, fill: true, backgroundColor: fade(col('--amber'))})]}, options: opts()}); } catch (e) { console.error(e); }
      try { heroChart(); } catch (e) { console.error(e); }
    }
    setRange('all');
    prediction(); evaluation(); forecast(10);
  } catch (e) { console.error(e); banner('Data gagal dimuat. Pastikan aplikasi berjalan lewat "python app.py".'); }
})();
})();
