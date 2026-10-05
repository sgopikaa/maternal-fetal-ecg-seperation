const API_BASE = ""; // same origin as Flask server

// ---------------------------------------------------------
// State
// ---------------------------------------------------------
let modelsReady = false;
let pollTimer = null;
let distressOn = false;
let fhrChart = null;
let idleAnimId = null;

// ---------------------------------------------------------
// ECG "chart paper" canvas renderer
// ---------------------------------------------------------
function drawGridPaper(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#FBF3EE";
  ctx.fillRect(0, 0, w, h);

  const minor = 8;
  ctx.strokeStyle = "#F6DDD5";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += minor) {
    ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, h); ctx.stroke();
  }
  for (let y = 0; y < h; y += minor) {
    ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(w, y + 0.5); ctx.stroke();
  }

  const major = minor * 5;
  ctx.strokeStyle = "#EFB6A8";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += major) {
    ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, h); ctx.stroke();
  }
  for (let y = 0; y < h; y += major) {
    ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(w, y + 0.5); ctx.stroke();
  }
}

function drawTrace(ctx, w, h, series, color, lineWidth = 1.6) {
  if (!series || series.length === 0) return;
  let min = Math.min(...series);
  let max = Math.max(...series);
  if (max - min < 1e-6) { max = min + 1; }
  const pad = 0.12 * (max - min);
  min -= pad; max += pad;

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  series.forEach((v, i) => {
    const x = (i / (series.length - 1)) * w;
    const y = h - ((v - min) / (max - min)) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderStrip(canvasId, seriesList) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width, h = rect.height;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  drawGridPaper(ctx, w, h);
  seriesList.forEach(({ data, color, lineWidth }) => drawTrace(ctx, w, h, data, color, lineWidth));
}

// idle looping placeholder trace for the hero before first run
function startIdleHero() {
  const canvas = document.getElementById("hero-canvas");
  const dpr = window.devicePixelRatio || 1;
  let phase = 0;

  function frame() {
    const rect = canvas.getBoundingClientRect();
    const w = rect.width, h = rect.height;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    drawGridPaper(ctx, w, h);

    const n = 400;
    const series = [];
    for (let i = 0; i < n; i++) {
      const t = i / n * 8 + phase;
      const beat = t % 1.0;
      let v = 0.05 * Math.sin(t * 6);
      if (beat < 0.04) v += Math.exp(-Math.pow((beat - 0.02) / 0.006, 2)) * 1.0;
      series.push(v);
    }
    drawTrace(ctx, w, h, series, "#B8CFC9", 1.4);
    phase += 0.02;
    idleAnimId = requestAnimationFrame(frame);
  }
  frame();
}

function stopIdleHero() {
  if (idleAnimId) cancelAnimationFrame(idleAnimId);
}

// ---------------------------------------------------------
// Status polling
// ---------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    updateStatus(data);
    return data;
  } catch (e) {
    setStatusPill("error", "Server unreachable");
    return null;
  }
}

function setStatusPill(kind, text) {
  const pill = document.getElementById("status-pill");
  pill.className = `status-pill status-${kind}`;
  document.getElementById("status-text").textContent = text;
}

function updateStatus(data) {
  modelsReady = data.models_ready;
  document.getElementById("run-btn").disabled = !modelsReady;
  if (data.training_in_progress) {
    setStatusPill("pending", "Training models on synthetic data…");
  } else if (data.models_ready) {
    setStatusPill("ready", "Models ready");
  } else {
    setStatusPill("pending", "Idle — press Retrain models");
  }
}

function pollUntilReady() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const data = await checkHealth();
    if (data && data.models_ready && !data.training_in_progress) {
      clearInterval(pollTimer);
    }
  }, 1500);
}

async function triggerTraining() {
  setStatusPill("pending", "Starting training…");
  document.getElementById("run-btn").disabled = true;
  await fetch(`${API_BASE}/api/train`, { method: "POST" });
  pollUntilReady();
}

// ---------------------------------------------------------
// Controls
// ---------------------------------------------------------
function bindSlider(id, outId, suffix) {
  const el = document.getElementById(id);
  const out = document.getElementById(outId);
  el.addEventListener("input", () => { out.textContent = `${el.value} ${suffix}`; });
}

function initControls() {
  bindSlider("maternal-hr", "maternal-hr-out", "bpm");
  bindSlider("fetal-hr", "fetal-hr-out", "bpm");
  bindSlider("snr", "snr-out", "dB");

  const toggle = document.getElementById("distress-toggle");
  toggle.addEventListener("click", () => {
    distressOn = !distressOn;
    toggle.setAttribute("aria-checked", String(distressOn));
  });

  document.getElementById("train-btn").addEventListener("click", triggerTraining);
  document.getElementById("run-btn").addEventListener("click", runAnalysis);

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => { t.classList.remove("active"); t.setAttribute("aria-selected", "false"); });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");
      const target = tab.dataset.tab;
      document.getElementById("tab-separation").hidden = target !== "separation";
      document.getElementById("tab-monitoring").hidden = target !== "monitoring";
    });
  });
}

// ---------------------------------------------------------
// Run pipeline + render
// ---------------------------------------------------------
async function runAnalysis() {
  const btn = document.getElementById("run-btn");
  btn.disabled = true;
  btn.textContent = "Running…";

  const payload = {
    maternal_hr: Number(document.getElementById("maternal-hr").value),
    fetal_hr: Number(document.getElementById("fetal-hr").value),
    fetal_distress: distressOn,
    snr_db: Number(document.getElementById("snr").value),
    seed: Math.floor(Math.random() * 100000),
  };

  try {
    const res = await fetch(`${API_BASE}/api/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error((await res.json()).message || "Pipeline failed");
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    alert(`Analysis failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run analysis";
  }
}

function renderResults(data) {
  document.getElementById("empty-state").hidden = true;

  // hero: switch from idle animation to real raw channel trace
  stopIdleHero();
  renderStrip("hero-canvas", [{ data: data.raw_channel, color: "#16262B", lineWidth: 1.3 }]);
  const meanFhr = data.monitoring.features.mean_fhr;
  document.getElementById("hero-hr-readout").textContent =
    meanFhr > 0 ? `${meanFhr.toFixed(0)} bpm (fetal, est.)` : "— bpm";

  // separation tab
  document.getElementById("tab-separation").hidden = false;
  renderStrip("canvas-truth-fetal", [{ data: data.ground_truth.fetal_ecg, color: "#0E7C66" }]);
  renderStrip("canvas-truth-maternal", [{ data: data.ground_truth.maternal_ecg, color: "#16262B" }]);
  renderStrip("canvas-ica", [{ data: data.separation.ica.fetal_component, color: "#2C6E8E" }]);
  renderStrip("canvas-cnn", [{ data: data.separation.cnn_autoencoder.fetal_component, color: "#0E7C66" }]);

  document.getElementById("ica-snr").textContent = `SNR ${data.separation.ica.snr_db} dB`;
  document.getElementById("cnn-snr").textContent = `SNR ${data.separation.cnn_autoencoder.snr_db} dB`;
  const bestLabel = data.separation.best_model === "cnn_autoencoder" ? "CNN Autoencoder leads" : "FastICA leads";
  document.getElementById("best-model-badge").textContent = bestLabel;

  // monitoring tab
  const feats = data.monitoring.features;
  document.getElementById("peaks-detected").textContent = data.monitoring.fetal_peaks_detected;
  document.getElementById("feat-mean").textContent = feats.mean_fhr > 0 ? `${feats.mean_fhr.toFixed(1)}` : "—";
  document.getElementById("feat-std").textContent = feats.mean_fhr > 0 ? `${feats.std_fhr.toFixed(1)}` : "—";
  document.getElementById("feat-rmssd").textContent = feats.mean_fhr > 0 ? `${feats.rmssd.toFixed(1)}` : "—";
  document.getElementById("feat-range").textContent = feats.mean_fhr > 0 ? `${feats.range_fhr.toFixed(1)}` : "—";

  renderFhrChart(data.monitoring.fhr_series_time, data.monitoring.fhr_series_bpm);

  const rf = data.monitoring.random_forest;
  document.getElementById("rf-label").textContent = (rf.label || "—").replace(/_/g, " ");
  const barsContainer = document.getElementById("rf-bars");
  barsContainer.innerHTML = "";
  Object.entries(rf.probabilities || {}).sort((a, b) => b[1] - a[1]).forEach(([label, p]) => {
    const row = document.createElement("div");
    row.className = "proba-row";
    row.innerHTML = `
      <span class="proba-label">${label.replace(/_/g, " ")}</span>
      <div class="proba-track"><div class="proba-fill" style="width:${(p * 100).toFixed(0)}%"></div></div>
      <span class="proba-pct">${(p * 100).toFixed(0)}%</span>`;
    barsContainer.appendChild(row);
  });

  const lstm = data.monitoring.cnn_lstm;
  document.getElementById("lstm-label").textContent = (lstm.label || "—").replace(/_/g, " ");
  document.getElementById("lstm-gauge").style.width = `${(lstm.probability * 100).toFixed(0)}%`;
  document.getElementById("lstm-prob").textContent = `${(lstm.probability * 100).toFixed(0)}%`;

  const concern = rf.label === "low_variability_distress" || rf.label === "bradycardia" || lstm.label === "distress_risk";
  document.getElementById("alert-banner").hidden = !concern;
}

function renderFhrChart(times, values) {
  const ctx = document.getElementById("fhr-chart").getContext("2d");
  if (fhrChart) fhrChart.destroy();
  fhrChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: times.map((t) => t.toFixed(1)),
      datasets: [{
        label: "Fetal HR (bpm)",
        data: values,
        borderColor: "#0E7C66",
        backgroundColor: "rgba(14,124,102,0.08)",
        borderWidth: 2,
        pointRadius: 2,
        tension: 0.25,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "time (s)", color: "#7C8A8D" }, grid: { color: "#F6DDD5" }, ticks: { color: "#7C8A8D" } },
        y: { title: { display: true, text: "bpm", color: "#7C8A8D" }, suggestedMin: 90, suggestedMax: 190, grid: { color: "#F6DDD5" }, ticks: { color: "#7C8A8D" } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

// ---------------------------------------------------------
// Init
// ---------------------------------------------------------
window.addEventListener("load", async () => {
  initControls();
  startIdleHero();
  const data = await checkHealth();
  if (data && !data.models_ready) {
    triggerTraining();
  } else if (data && data.training_in_progress) {
    pollUntilReady();
  }
});
