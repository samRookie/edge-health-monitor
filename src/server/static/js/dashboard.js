// Edge Health Monitor Dashboard Logic
// Adheres strictly to Edge Impulse Studio information architecture and tokens.css

// Chart.js global defaults
if (typeof Chart !== 'undefined') {
  Chart.defaults.font.family = '"IBM Plex Sans", -apple-system, sans-serif';
  Chart.defaults.animation = false;
  Chart.defaults.color = '#58626B';
  Chart.defaults.borderColor = '#CBC8BF';
  
  if (typeof ChartZoom !== 'undefined') {
    try {
      Chart.register(ChartZoom);
    } catch (e) {
      // already registered or auto-registered
    }
  }
}

// Page Navigation State
const navLinks = document.querySelectorAll('.nav-link');
const pageSections = document.querySelectorAll('.page-section');
const pageHeaderTitle = document.getElementById('page-header-title');

function showPage(pageId) {
  pageSections.forEach(sec => {
    sec.style.display = 'none';
  });
  
  const targetSec = document.getElementById(`page-${pageId}`);
  if (targetSec) {
    targetSec.style.display = 'block';
  }

  navLinks.forEach(link => {
    const parent = link.closest('.nav-item');
    if (link.getAttribute('data-page') === pageId) {
      parent.classList.add('active');
      pageHeaderTitle.textContent = link.querySelector('.nav-label').textContent;
      document.title = `${link.querySelector('.nav-label').textContent}, Edge Health Monitor`;
    } else {
      parent.classList.remove('active');
    }
  });

  // Render or resize charts when switching to page
  if (pageId === 'data') {
    renderSignalInspector();
  } else if (pageId === 'features') {
    renderPCAScatter();
  } else if (pageId === 'classifier-activity') {
    renderHARCurves();
  } else if (pageId === 'classifier-keyword') {
    renderKWSCurves();
  } else if (pageId === 'federated') {
    renderFederatedChart();
  } else if (pageId === 'robustness') {
    renderRobustnessChart();
  } else if (pageId === 'live') {
    renderLiveOscilloscope();
  }
}

navLinks.forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const pageId = link.getAttribute('data-page');
    window.location.hash = pageId;
    showPage(pageId);
  });
});

window.addEventListener('hashchange', () => {
  const hash = window.location.hash.replace('#', '') || 'dashboard';
  showPage(hash);
});

// Tab Navigation
const tabButtons = document.querySelectorAll('.tab-btn');
tabButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    const parentHeader = btn.closest('.tabs-header');
    parentHeader.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const targetTabId = btn.getAttribute('data-tab');
    document.querySelectorAll('.tab-content').forEach(tc => {
      tc.style.display = tc.id === targetTabId ? 'block' : 'none';
    });
  });
});

// Mode Controls
const liveModeSelect = document.getElementById('live-mode-select');
const alertModeSelect = document.getElementById('alert-mode-select');

if (liveModeSelect) {
  liveModeSelect.addEventListener('change', async () => {
    await fetch('/api/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ live_mode: liveModeSelect.value })
    });
  });
}

if (alertModeSelect) {
  alertModeSelect.addEventListener('change', async () => {
    await fetch('/api/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ alert_mode: alertModeSelect.value })
    });
  });
}

// Clear Events
const clearEventsBtn = document.getElementById('clear-events-btn');
const eventsTableBody = document.getElementById('events-table-body');

if (clearEventsBtn) {
  clearEventsBtn.addEventListener('click', async () => {
    await fetch('/api/events/clear', { method: 'POST' });
    eventsTableBody.innerHTML = '<tr><td colspan="5" class="empty-state">No live samples yet. Open the phone page or switch to Simulated replay.</td></tr>';
  });
}

// Zoom / Pan Chart Toolbars
let inspectorChart = null;
let pcaChart = null;
let harCurvesChart = null;
let kwsCurvesChart = null;
let fedChart = null;
let robustnessChart = null;
let liveChart = null;

function getChartByName(name) {
  if (name === 'inspector') return inspectorChart;
  if (name === 'pca') return pcaChart;
  if (name === 'har-curves') return harCurvesChart;
  if (name === 'kws-curves') return kwsCurvesChart;
  if (name === 'fed') return fedChart;
  if (name === 'robustness') return robustnessChart;
  if (name === 'live') return liveChart;
  return null;
}

document.addEventListener('click', (e) => {
  const btn = e.target.closest('.chart-tool-btn[data-chart]');
  if (!btn) return;
  const chartName = btn.getAttribute('data-chart');
  const action = btn.getAttribute('data-action');
  const chart = getChartByName(chartName);
  if (!chart) return;
  
  if (action === 'zoom-in') {
    if (chart.zoom) {
      chart.zoom(1.25);
    }
  } else if (action === 'zoom-out') {
    if (chart.zoom) {
      chart.zoom(0.8);
    }
  } else if (action === 'reset') {
    if (chart.resetZoom) {
      chart.resetZoom();
    }
  }
});

// 1. SIGNAL INSPECTOR (Page 2 - Data)
const inspectorSamples = {
  walking: {
    generate: (points) => ({
      x: Array.from({ length: points }, (_, i) => +(0.25 * Math.sin((i / 8) * Math.PI)).toFixed(3)),
      y: Array.from({ length: points }, (_, i) => +(0.15 * Math.cos((i / 8) * Math.PI)).toFixed(3)),
      z: Array.from({ length: points }, (_, i) => +(0.98 + 0.35 * Math.sin((i / 8) * Math.PI)).toFixed(3))
    })
  },
  upstairs: {
    generate: (points) => ({
      x: Array.from({ length: points }, (_, i) => +(0.38 * Math.sin((i / 6) * Math.PI) + 0.08 * (Math.random() - 0.5)).toFixed(3)),
      y: Array.from({ length: points }, (_, i) => +(0.22 * Math.cos((i / 6) * Math.PI)).toFixed(3)),
      z: Array.from({ length: points }, (_, i) => +(1.05 + 0.58 * Math.sin((i / 6) * Math.PI)).toFixed(3))
    })
  },
  sitting: {
    generate: (points) => ({
      x: Array.from({ length: points }, () => +(0.03 + 0.02 * (Math.random() - 0.5)).toFixed(3)),
      y: Array.from({ length: points }, () => +(0.05 + 0.02 * (Math.random() - 0.5)).toFixed(3)),
      z: Array.from({ length: points }, () => +(0.98 + 0.03 * (Math.random() - 0.5)).toFixed(3))
    })
  },
  laying: {
    generate: (points) => ({
      x: Array.from({ length: points }, () => +(0.95 + 0.02 * (Math.random() - 0.5)).toFixed(3)),
      y: Array.from({ length: points }, () => +(0.12 + 0.02 * (Math.random() - 0.5)).toFixed(3)),
      z: Array.from({ length: points }, () => +(0.08 + 0.02 * (Math.random() - 0.5)).toFixed(3))
    })
  },
  fall: {
    generate: (points) => {
      const x = [];
      const y = [];
      const z = [];
      for (let i = 0; i < points; i++) {
        if (i < 45) {
          x.push(+(0.2 * Math.sin(i / 6)).toFixed(3));
          y.push(+(0.1 * Math.cos(i / 6)).toFixed(3));
          z.push(+(1.0 + 0.3 * Math.sin(i / 6)).toFixed(3));
        } else if (i < 55) {
          // Free fall drop < 0.6g
          x.push(0.12);
          y.push(0.08);
          z.push(0.35);
        } else if (i < 68) {
          // Sharp impact spike > 3.0g
          x.push(0.65);
          y.push(0.95);
          z.push(3.32);
        } else {
          // Post fall immobility lying flat
          x.push(0.96);
          y.push(0.08);
          z.push(0.06);
        }
      }
      return { x, y, z };
    }
  },
  stop: {
    generate: (points) => {
      // Audio waveform envelope for word 'stop'
      const z = Array.from({ length: points }, (_, i) => {
        if (i > 35 && i < 90) {
          const env = Math.sin(((i - 35) / 55) * Math.PI);
          return +(env * (Math.sin(i * 1.8) * 0.85 + 0.15 * (Math.random() - 0.5))).toFixed(3);
        }
        return +(0.04 * (Math.random() - 0.5)).toFixed(3);
      });
      return {
        x: Array.from({ length: points }, () => 0.0),
        y: Array.from({ length: points }, () => 0.0),
        z
      };
    }
  }
};

function renderSignalInspector() {
  const ctx = document.getElementById('inspector-canvas');
  if (!ctx) return;
  if (inspectorChart) return;

  const points = 128;
  const labels = Array.from({ length: points }, (_, i) => i.toString());
  const curData = inspectorSamples.walking.generate(points);

  inspectorChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        { label: 'acc_x (g)', data: curData.x, borderColor: '#1F6F78', borderWidth: 1.5, pointRadius: 0, tension: 0 },
        { label: 'acc_y (g)', data: curData.y, borderColor: '#A06F18', borderWidth: 1.5, pointRadius: 0, tension: 0 },
        { label: 'acc_z (g)', data: curData.z, borderColor: '#4B5D8A', borderWidth: 1.5, pointRadius: 0, tension: 0 }
      ]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      scales: {
        x: {
          title: { display: true, text: 'Sample Index (50 Hz, 20 ms / sample)' },
          grid: { color: '#CBC8BF' },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 } }
        },
        y: {
          title: { display: true, text: 'Acceleration (g)' },
          grid: { color: '#CBC8BF' },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 } }
        }
      },
      plugins: {
        legend: { labels: { font: { family: 'IBM Plex Sans', size: 11 }, boxWidth: 12 } },
        zoom: {
          pan: { enabled: true, mode: 'x' },
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' }
        }
      }
    }
  });

  const selectEl = document.getElementById('inspector-select');
  if (selectEl) {
    selectEl.addEventListener('change', () => {
      const val = selectEl.value;
      const gen = inspectorSamples[val] ? inspectorSamples[val].generate(128) : inspectorSamples.walking.generate(128);
      inspectorChart.data.datasets[0].data = gen.x;
      inspectorChart.data.datasets[1].data = gen.y;
      inspectorChart.data.datasets[2].data = gen.z;
      
      if (val === 'stop') {
        inspectorChart.options.scales.x.title.text = 'Audio Frame (16 kHz, downsampled)';
        inspectorChart.options.scales.y.title.text = 'Normalized Audio Amplitude';
        inspectorChart.data.datasets[0].hidden = true;
        inspectorChart.data.datasets[1].hidden = true;
        inspectorChart.data.datasets[2].label = 'Audio waveform (stop)';
      } else {
        inspectorChart.options.scales.x.title.text = 'Sample Index (50 Hz, 20 ms / sample)';
        inspectorChart.options.scales.y.title.text = 'Acceleration (g)';
        inspectorChart.data.datasets[0].hidden = false;
        inspectorChart.data.datasets[1].hidden = false;
        inspectorChart.data.datasets[2].label = 'acc_z (g)';
      }
      if (inspectorChart.resetZoom) inspectorChart.resetZoom();
      inspectorChart.update();
    });
  }
}

// 2. FEATURE EXPLORER PCA SCATTER (Page 4 - Features)
function renderPCAScatter() {
  const ctx = document.getElementById('pca-scatter-canvas');
  if (!ctx) return;
  if (pcaChart) return;

  const classColors = ['#1F6F78', '#A06F18', '#A63A33', '#4B5D8A', '#6F7A36', '#85563A'];
  const classShapes = ['circle', 'triangle', 'rect', 'cross', 'star', 'rectRot'];
  const classNames = ['WALKING', 'UPSTAIRS', 'DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'];
  const centers = [
    [-2.5, 1.2], [-1.8, 2.5], [-2.1, 0.2],
    [2.8, -1.5], [2.2, -0.4], [1.5, -3.2]
  ];

  let sampleCounter = 1000;
  const datasets = classNames.map((name, idx) => {
    const [cx, cy] = centers[idx];
    const pts = Array.from({ length: 30 }, () => {
      sampleCounter++;
      return {
        x: +(cx + (Math.random() - 0.5) * 1.3).toFixed(2),
        y: +(cy + (Math.random() - 0.5) * 1.3).toFixed(2),
        sampleId: `#${sampleCounter}`,
        className: name
      };
    });
    return {
      label: name,
      data: pts,
      backgroundColor: classColors[idx],
      borderColor: classColors[idx],
      pointStyle: classShapes[idx],
      pointRadius: 4,
      pointHoverRadius: 6
    };
  });

  pcaChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      onClick: (e, elements) => {
        if (elements.length > 0) {
          const el = elements[0];
          const pt = datasets[el.datasetIndex].data[el.index];
          const idEl = document.getElementById('pca-card-id');
          const clsEl = document.getElementById('pca-card-class');
          const pc1El = document.getElementById('pca-card-pc1');
          const pc2El = document.getElementById('pca-card-pc2');
          if (idEl) idEl.textContent = pt.sampleId;
          if (clsEl) clsEl.textContent = pt.className;
          if (pc1El) pc1El.textContent = pt.x;
          if (pc2El) pc2El.textContent = pt.y;
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Principal component 1 (54.2% variance)' },
          grid: { color: '#CBC8BF' },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 } }
        },
        y: {
          title: { display: true, text: 'Principal component 2 (21.8% variance)' },
          grid: { color: '#CBC8BF' },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.raw.className} (${ctx.raw.sampleId}): [PC1: ${ctx.raw.x}, PC2: ${ctx.raw.y}]`
          }
        },
        zoom: {
          pan: { enabled: true, mode: 'xy' },
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' }
        }
      }
    }
  });

  // Class toggle buttons listener
  document.querySelectorAll('.class-toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = parseInt(btn.getAttribute('data-class-idx'), 10);
      const isVisible = pcaChart.isDatasetVisible(idx);
      pcaChart.setDatasetVisibility(idx, !isVisible);
      btn.classList.toggle('inactive', isVisible);
      pcaChart.update();
    });
  });
}

// 3. CLASSIFIER ACTIVITY CURVES (Page 5)
function renderHARCurves() {
  const ctx = document.getElementById('har-curves-canvas');
  if (!ctx) return;
  if (harCurvesChart) return;

  const epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const trainLoss = [0.467, 0.169, 0.129, 0.109, 0.100, 0.103, 0.097, 0.088, 0.097, 0.087];
  const valLoss = [1.300, 0.989, 0.663, 0.309, 0.374, 0.417, 0.634, 0.863, 0.515, 0.750];
  const trainAcc = [84.4, 94.2, 95.2, 95.2, 96.0, 95.6, 95.8, 95.9, 95.7, 95.9];
  const valAcc = [55.6, 65.7, 71.8, 90.3, 91.0, 91.2, 89.0, 91.2, 90.6, 91.6];

  harCurvesChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochs.map(e => `Epoch ${e}`),
      datasets: [
        { label: 'Train loss', data: trainLoss, borderColor: '#1F6F78', yAxisID: 'yLoss', borderWidth: 1.5, pointRadius: 3, tension: 0 },
        { label: 'Val loss', data: valLoss, borderColor: '#A06F18', yAxisID: 'yLoss', borderWidth: 1.5, pointRadius: 3, tension: 0 },
        { label: 'Train acc (%)', data: trainAcc, borderColor: '#3E7A4D', yAxisID: 'yAcc', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 3, tension: 0 },
        { label: 'Val acc (%)', data: valAcc, borderColor: '#4B5D8A', yAxisID: 'yAcc', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 3, tension: 0 }
      ]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#CBC8BF' }, ticks: { font: { family: 'IBM Plex Mono', size: 10 } } },
        yLoss: {
          type: 'linear',
          position: 'left',
          title: { display: true, text: 'Loss' },
          grid: { color: '#CBC8BF' },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 } }
        },
        yAcc: {
          type: 'linear',
          position: 'right',
          title: { display: true, text: 'Accuracy (%)' },
          grid: { drawOnChartArea: false },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 } },
          min: 40,
          max: 100
        }
      },
      plugins: {
        legend: { labels: { font: { family: 'IBM Plex Sans', size: 11 }, boxWidth: 12 } },
        zoom: {
          pan: { enabled: true, mode: 'x' },
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' }
        }
      }
    }
  });
}

// 4. CLASSIFIER KEYWORD CURVES (Page 6)
function renderKWSCurves() {
  const ctx = document.getElementById('kws-curves-canvas');
  if (!ctx) return;
  if (kwsCurvesChart) return;

  const epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const trainLoss = [1.727, 1.434, 1.216, 1.042, 0.897, 0.798, 0.711, 0.612, 0.559, 0.514];
  const valLoss = [2.097, 2.139, 2.154, 2.161, 2.120, 1.978, 1.730, 1.571, 0.888, 0.858];
  const trainAcc = [31.7, 44.9, 53.9, 60.6, 66.4, 70.8, 75.3, 78.9, 80.7, 82.0];
  const valAcc = [12.5, 12.5, 12.5, 12.8, 20.0, 31.7, 36.7, 43.3, 70.2, 69.5];

  kwsCurvesChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochs.map(e => `Epoch ${e}`),
      datasets: [
        { label: 'Train loss', data: trainLoss, borderColor: '#1F6F78', yAxisID: 'yLoss', borderWidth: 1.5, pointRadius: 3, tension: 0 },
        { label: 'Val loss', data: valLoss, borderColor: '#A06F18', yAxisID: 'yLoss', borderWidth: 1.5, pointRadius: 3, tension: 0 },
        { label: 'Train acc (%)', data: trainAcc, borderColor: '#3E7A4D', yAxisID: 'yAcc', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 3, tension: 0 },
        { label: 'Val acc (%)', data: valAcc, borderColor: '#4B5D8A', yAxisID: 'yAcc', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 3, tension: 0 }
      ]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#CBC8BF' }, ticks: { font: { family: 'IBM Plex Mono', size: 10 } } },
        yLoss: {
          type: 'linear',
          position: 'left',
          title: { display: true, text: 'Loss' },
          grid: { color: '#CBC8BF' },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 } }
        },
        yAcc: {
          type: 'linear',
          position: 'right',
          title: { display: true, text: 'Accuracy (%)' },
          grid: { drawOnChartArea: false },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 } },
          min: 0,
          max: 100
        }
      },
      plugins: {
        legend: { labels: { font: { family: 'IBM Plex Sans', size: 11 }, boxWidth: 12 } },
        zoom: {
          pan: { enabled: true, mode: 'x' },
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' }
        }
      }
    }
  });
}

// 5. FEDERATED LEARNING CHART (Page 8)
function renderFederatedChart() {
  const ctx = document.getElementById('federated-canvas');
  if (!ctx) return;
  if (fedChart) return;

  const rounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const centralized = [90.02, 90.02, 90.02, 90.02, 90.02, 90.02, 90.02, 90.02, 90.02, 90.02];
  const fedIid = [31.63, 30.10, 33.29, 47.13, 52.19, 55.68, 65.59, 69.97, 76.08, 81.03];
  const fedNonIid = [36.72, 44.89, 64.91, 72.45, 75.57, 78.96, 78.93, 79.91, 83.71, 85.27];

  fedChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: rounds.map(r => `Round ${r}`),
      datasets: [
        { label: 'Centralized baseline (90.02%)', data: centralized, borderColor: '#7B858D', borderDash: [5, 5], borderWidth: 1.5, pointRadius: 0, tension: 0 },
        { label: 'FedAvg IID', data: fedIid, borderColor: '#1F6F78', borderWidth: 2, pointRadius: 4, tension: 0 },
        { label: 'FedAvg Non-IID (Subject partition)', data: fedNonIid, borderColor: '#A06F18', borderWidth: 2, pointRadius: 4, tension: 0 }
      ]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#CBC8BF' }, ticks: { font: { family: 'IBM Plex Mono', size: 10 } } },
        y: {
          title: { display: true, text: 'Test Accuracy (%)' },
          grid: { color: '#CBC8BF' },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 } },
          min: 20,
          max: 100
        }
      },
      plugins: {
        legend: { labels: { font: { family: 'IBM Plex Sans', size: 11 }, boxWidth: 12 } },
        zoom: {
          pan: { enabled: true, mode: 'xy' },
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' }
        }
      }
    }
  });
}

// 6. ROBUSTNESS ATTACK & NOISE CHART (Page 9)
function renderRobustnessChart() {
  const ctx = document.getElementById('robustness-canvas');
  if (!ctx) return;
  if (robustnessChart) return;

  const epsilons = ['0.00', '0.02', '0.05', '0.10', '0.20'];
  const undefended = [69.50, 37.50, 17.67, 11.50, 6.17];
  const defended = [61.20, 58.40, 52.10, 44.60, 31.70];

  robustnessChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: epsilons.map(e => `eps = ${e}`),
      datasets: [
        { label: 'Undefended baseline (clean: 69.5%)', data: undefended, borderColor: '#A63A33', borderWidth: 2, pointRadius: 4, tension: 0 },
        { label: 'Defended model (noise augmentation)', data: defended, borderColor: '#1F6F78', borderWidth: 2, pointRadius: 4, tension: 0 }
      ]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'FGSM Perturbation Epsilon' },
          grid: { color: '#CBC8BF' },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 } }
        },
        y: {
          title: { display: true, text: 'Test Accuracy (%)' },
          grid: { color: '#CBC8BF' },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 } },
          min: 0,
          max: 80
        }
      },
      plugins: {
        legend: { labels: { font: { family: 'IBM Plex Sans', size: 11 }, boxWidth: 12 } },
        zoom: {
          pan: { enabled: true, mode: 'xy' },
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' }
        }
      }
    }
  });
}

// 7. REAL-TIME LIVE OSCILLOSCOPE (Page 10 - Live classification)
let isLivePaused = false;
const liveWindowSize = 60; // 3 seconds at 20 Hz
const liveLabels = Array.from({ length: liveWindowSize }, (_, i) => `${-(liveWindowSize - i) * 50}ms`);
const liveAccX = Array.from({ length: liveWindowSize }, () => 0.0);
const liveAccY = Array.from({ length: liveWindowSize }, () => 0.0);
const liveAccZ = Array.from({ length: liveWindowSize }, () => 1.0);
const liveMag = Array.from({ length: liveWindowSize }, () => 1.0);
const impactThresh = Array.from({ length: liveWindowSize }, () => 2.5);
const dropThresh = Array.from({ length: liveWindowSize }, () => 0.6);

function renderLiveOscilloscope() {
  const ctx = document.getElementById('live-sensor-canvas');
  if (!ctx) return;
  if (liveChart) return;

  liveChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: liveLabels,
      datasets: [
        { label: 'acc_x (g)', data: liveAccX, borderColor: '#1F6F78', borderWidth: 1.2, pointRadius: 0, tension: 0 },
        { label: 'acc_y (g)', data: liveAccY, borderColor: '#A06F18', borderWidth: 1.2, pointRadius: 0, tension: 0 },
        { label: 'acc_z (g)', data: liveAccZ, borderColor: '#4B5D8A', borderWidth: 1.2, pointRadius: 0, tension: 0 },
        { label: '|a| mag (g)', data: liveMag, borderColor: '#A63A33', borderWidth: 2.0, pointRadius: 0, tension: 0 },
        { label: 'Impact threshold (2.5g)', data: impactThresh, borderColor: '#A63A33', borderDash: [4, 4], borderWidth: 1.0, pointRadius: 0, tension: 0 },
        { label: 'Drop threshold (0.6g)', data: dropThresh, borderColor: '#85563A', borderDash: [4, 4], borderWidth: 1.0, pointRadius: 0, tension: 0 }
      ]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          display: true,
          grid: { color: '#CBC8BF' },
          ticks: { font: { family: 'IBM Plex Mono', size: 9 }, maxTicksLimit: 8 }
        },
        y: {
          title: { display: true, text: 'Acceleration (g)' },
          grid: { color: '#CBC8BF' },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 } },
          min: -0.5,
          max: 4.0
        }
      },
      plugins: {
        legend: { labels: { font: { family: 'IBM Plex Sans', size: 10 }, boxWidth: 10 } },
        zoom: {
          pan: { enabled: true, mode: 'y' },
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'y' }
        }
      }
    }
  });

  const pauseBtn = document.getElementById('btn-pause-stream');
  if (pauseBtn) {
    pauseBtn.addEventListener('click', () => {
      isLivePaused = !isLivePaused;
      pauseBtn.textContent = isLivePaused ? 'Resume' : 'Pause';
      const statusEl = document.getElementById('live-stream-status');
      if (statusEl) statusEl.textContent = isLivePaused ? 'Trace paused' : 'Streaming at 20 Hz';
    });
  }

  const clearBtn = document.getElementById('btn-clear-stream');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      for (let i = 0; i < liveWindowSize; i++) {
        liveAccX[i] = 0.0;
        liveAccY[i] = 0.0;
        liveAccZ[i] = 1.0;
        liveMag[i] = 1.0;
      }
      liveChart.update('none');
    });
  }
}

function pushLiveSensorSample(ax, ay, az, mag) {
  if (isLivePaused || !liveChart) return;
  liveAccX.shift(); liveAccX.push(ax);
  liveAccY.shift(); liveAccY.push(ay);
  liveAccZ.shift(); liveAccZ.push(az);
  liveMag.shift(); liveMag.push(mag);

  const magDisp = document.getElementById('live-mag-display');
  if (magDisp) magDisp.textContent = `${mag.toFixed(2)}g`;

  liveChart.update('none');
}

// Live Dashboard WebSocket Connection
let dashWs = null;
const alertBanner = document.getElementById('live-alert-banner');
const alertTimeEl = document.getElementById('alert-banner-time');
const alertRuleEl = document.getElementById('alert-banner-rule');

const curActLabel = document.getElementById('current-activity-label');
const curKwdLabel = document.getElementById('current-keyword-label');
const liveActProbs = document.getElementById('live-activity-probs');
const liveKwdProbs = document.getElementById('live-keyword-probs');

function connectDashboardWS() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/dashboard`;
  dashWs = new WebSocket(wsUrl);

  dashWs.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'live_update') {
      updateLiveClassification(msg);
    } else if (msg.type === 'init') {
      if (msg.live_mode && liveModeSelect) liveModeSelect.value = msg.live_mode;
      if (msg.alert_mode && alertModeSelect) alertModeSelect.value = msg.alert_mode;
      if (msg.events && msg.events.length > 0) {
        renderEventsList(msg.events);
      }
    }
  };

  dashWs.onclose = () => {
    setTimeout(connectDashboardWS, 2000);
  };
}

function updateLiveClassification(data) {
  // Push live IMU data into oscilloscope
  if (data.imu && Array.isArray(data.imu) && data.imu.length >= 3) {
    const mag = data.latest_imu_magnitude || Math.sqrt(data.imu[0]**2 + data.imu[1]**2 + data.imu[2]**2);
    pushLiveSensorSample(data.imu[0], data.imu[1], data.imu[2], mag);
  } else if (data.latest_imu_magnitude !== undefined) {
    pushLiveSensorSample(0.1, 0.1, data.latest_imu_magnitude, data.latest_imu_magnitude);
  }

  // Update Activity Recognition
  if (data.activity && curActLabel) {
    curActLabel.textContent = data.activity.label;
    if (data.activity.probabilities) {
      let html = '';
      for (const [cls, prob] of Object.entries(data.activity.probabilities)) {
        const pct = (prob * 100).toFixed(1);
        html += `
          <tr>
            <td>${cls}</td>
            <td>
              <div class="prob-meter-container">
                <div class="prob-meter-track">
                  <div class="prob-meter-fill" style="width: ${pct}%;"></div>
                </div>
                <span class="prob-meter-val">${prob.toFixed(3)}</span>
              </div>
            </td>
          </tr>`;
      }
      liveActProbs.innerHTML = html;
    }
  }

  // Update Keyword Spotting
  if (data.keyword && curKwdLabel) {
    curKwdLabel.textContent = data.keyword.label;
    if (data.keyword.probabilities) {
      let html = '';
      for (const [cls, prob] of Object.entries(data.keyword.probabilities)) {
        const pct = (prob * 100).toFixed(1);
        html += `
          <tr>
            <td>${cls}</td>
            <td>
              <div class="prob-meter-container">
                <div class="prob-meter-track">
                  <div class="prob-meter-fill" style="width: ${pct}%;"></div>
                </div>
                <span class="prob-meter-val">${prob.toFixed(3)}</span>
              </div>
            </td>
          </tr>`;
      }
      liveKwdProbs.innerHTML = html;
    }
  }

  // Process Alert Event
  if (data.alert) {
    if (alertBanner) {
      alertBanner.style.display = 'block';
      alertTimeEl.textContent = data.alert.time_str || new Date().toLocaleTimeString();
      alertRuleEl.textContent = data.alert.rule || 'Emergency condition met';
    }
    addEventRow(data.alert);
  }
}

function addEventRow(alert) {
  if (!eventsTableBody) return;
  if (eventsTableBody.querySelector('.empty-state')) {
    eventsTableBody.innerHTML = '';
  }

  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td class="mono">${alert.time_str || new Date().toLocaleTimeString()}</td>
    <td>${alert.source || 'Phone sensor'}</td>
    <td class="mono"><strong>${alert.activity || alert.keyword || 'ALERT'}</strong></td>
    <td class="num">${((alert.activity_confidence || alert.keyword_confidence || 0.95) * 100).toFixed(1)}%</td>
    <td><span class="status-pill"><span class="status-dot alert"></span>${alert.rule}</span></td>
  `;
  eventsTableBody.insertBefore(tr, eventsTableBody.firstChild);
}

function renderEventsList(events) {
  if (!eventsTableBody || events.length === 0) return;
  eventsTableBody.innerHTML = '';
  events.forEach(addEventRow);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  const initialPage = window.location.hash.replace('#', '') || 'dashboard';
  showPage(initialPage);
  connectDashboardWS();
});