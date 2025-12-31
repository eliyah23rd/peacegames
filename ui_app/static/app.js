const state = {
  files: [],
  data: null,
  currentMetric: null,
  currentTurnIndex: 0,
};

const palette = [
  "#34a39a",
  "#e98467",
  "#5f7a75",
  "#f4b073",
  "#93bd8a",
  "#c995a7",
  "#6f879e",
  "#f7c372",
  "#5aa39a",
  "#f58a4b",
];

const dataFileSelect = document.getElementById("dataFileSelect");
const metricSelect = document.getElementById("metricSelect");
const chartCanvas = document.getElementById("chartCanvas");
const chartTitle = document.getElementById("chartTitle");
const legend = document.getElementById("legend");
const turnViewBtn = document.getElementById("turnViewBtn");
const mapViewBtn = document.getElementById("mapViewBtn");
const timelineView = document.getElementById("timelineView");
const turnView = document.getElementById("turnView");
const mapView = document.getElementById("mapView");
const turnTableWrapper = document.getElementById("turnTableWrapper");
const turnLabel = document.getElementById("turnLabel");
const prevTurnBtn = document.getElementById("prevTurnBtn");
const nextTurnBtn = document.getElementById("nextTurnBtn");
const backBtn = document.getElementById("backBtn");
const mapImage = document.getElementById("mapImage");
const mapTurnLabel = document.getElementById("mapTurnLabel");
const mapPrevTurnBtn = document.getElementById("mapPrevTurnBtn");
const mapNextTurnBtn = document.getElementById("mapNextTurnBtn");
const mapBackBtn = document.getElementById("mapBackBtn");
const mapLegend = document.getElementById("mapLegend");

async function loadFiles() {
  const res = await fetch("/api/files");
  const payload = await res.json();
  state.files = payload.files || [];
  dataFileSelect.innerHTML = "";
  state.files.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    dataFileSelect.appendChild(opt);
  });
  if (state.files.length > 0) {
    dataFileSelect.value = state.files[0];
    await loadData(state.files[0]);
  }
}

async function loadData(filename) {
  const res = await fetch(`/api/data/${encodeURIComponent(filename)}`);
  state.data = await res.json();
  state.currentTurnIndex = 0;
  populateMetrics();
  renderChart();
  renderTurnTable();
}

function populateMetrics() {
  if (!state.data) return;
  metricSelect.innerHTML = "";
  const vars = state.data.ledger_vars || [];
  vars.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name.replace(/_/g, " ");
    metricSelect.appendChild(opt);
  });
  const defaultMetric = vars.includes("total_welfare") ? "total_welfare" : vars[0];
  state.currentMetric = defaultMetric || null;
  if (state.currentMetric) {
    metricSelect.value = state.currentMetric;
  }
}

function getMetricIndex(metric) {
  const vars = state.data?.ledger_vars || [];
  return vars.indexOf(metric);
}

function renderChart() {
  if (!state.data || !state.currentMetric) return;
  const ctx = chartCanvas.getContext("2d");
  ctx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);

  const { agents, turns, data } = state.data;
  const metricIndex = getMetricIndex(state.currentMetric);
  const values = data.map((agentRows) => agentRows.map((row) => row[metricIndex]));
  const allVals = values.flat();
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const padding = 50;
  const width = chartCanvas.width - padding * 2;
  const height = chartCanvas.height - padding * 2;

  const yScale = (val) => {
    if (maxVal === minVal) return padding + height / 2;
    return padding + height - ((val - minVal) / (maxVal - minVal)) * height;
  };
  const xScale = (idx) => {
    if (turns.length <= 1) return padding + width / 2;
    return padding + (idx / (turns.length - 1)) * width;
  };

  ctx.strokeStyle = "#d8c9b8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, padding + height);
  ctx.lineTo(padding + width, padding + height);
  ctx.stroke();

  ctx.fillStyle = "#6c655d";
  ctx.font = "12px Georgia";
  ctx.fillText(String(maxVal), 8, padding + 4);
  ctx.fillText(String(minVal), 8, padding + height);

  turns.forEach((turn, idx) => {
    ctx.fillText(String(turn), xScale(idx) - 6, padding + height + 18);
  });

  legend.innerHTML = "";
  agents.forEach((agent, idx) => {
    const color = palette[idx % palette.length];
    const series = values[idx];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    series.forEach((val, tIdx) => {
      const x = xScale(tIdx);
      const y = yScale(val);
      if (tIdx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    series.forEach((val, tIdx) => {
      const x = xScale(tIdx);
      const y = yScale(val);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    const item = document.createElement("span");
    const dot = document.createElement("i");
    dot.style.background = color;
    item.appendChild(dot);
    item.appendChild(document.createTextNode(agent));
    legend.appendChild(item);
  });

  chartTitle.textContent = state.currentMetric.replace(/_/g, " ");
}

function renderTurnTable() {
  if (!state.data) return;
  const { agents, turns, ledger_vars, data } = state.data;
  const turnIdx = state.currentTurnIndex;
  const turnValue = turns[turnIdx];
  turnLabel.textContent = `Turn ${turnValue}`;

  const maxByVar = ledger_vars.map((_, varIdx) => {
    let max = 0;
    agents.forEach((_, agentIdx) => {
      max = Math.max(max, data[agentIdx][turnIdx][varIdx]);
    });
    return max;
  });

  const container = document.createElement("div");
  container.className = "metric-grid";

  ledger_vars.forEach((metric, metricIdx) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    const title = document.createElement("h3");
    title.textContent = metric.replace(/_/g, " ");
    card.appendChild(title);

    const maxVal = maxByVar[metricIdx];
    agents.forEach((agent, agentIdx) => {
      const row = document.createElement("div");
      row.className = "bar-row";

      const label = document.createElement("div");
      label.className = "bar-label";
      label.textContent = agent;

      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      const val = data[agentIdx][turnIdx][metricIdx];
      const pct = maxVal > 0 ? Math.round((val / maxVal) * 100) : 0;
      fill.style.width = `${pct}%`;
      fill.style.background = palette[agentIdx % palette.length];
      track.appendChild(fill);

      const value = document.createElement("div");
      value.className = "bar-value";
      value.textContent = val;

      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(value);
      card.appendChild(row);
    });

    container.appendChild(card);
  });

  turnTableWrapper.innerHTML = "";
  turnTableWrapper.appendChild(container);
}

function renderMap() {
  if (!state.data) return;
  const { territory_names, territory_positions, territory_owners, turns } = state.data;
  if (!territory_names || !territory_positions || !territory_owners) return;

  const turnIdx = state.currentTurnIndex;
  mapTurnLabel.textContent = `Turn ${turns[turnIdx]}`;
  mapLegend.innerHTML = "";
  state.data.agents.forEach((agent, idx) => {
    const item = document.createElement("span");
    const dot = document.createElement("i");
    dot.style.background = palette[idx % palette.length];
    item.appendChild(dot);
    item.appendChild(document.createTextNode(agent));
    mapLegend.appendChild(item);
  });

  const fileName = dataFileSelect.value;
  const turnValue = turns[turnIdx];
  mapImage.src = `/api/map?file=${encodeURIComponent(fileName)}&turn=${turnValue}&t=${Date.now()}`;
}

dataFileSelect.addEventListener("change", async (e) => {
  await loadData(e.target.value);
});

metricSelect.addEventListener("change", (e) => {
  state.currentMetric = e.target.value;
  renderChart();
});

turnViewBtn.addEventListener("click", () => {
  timelineView.classList.add("hidden");
  turnView.classList.remove("hidden");
  mapView.classList.add("hidden");
  renderTurnTable();
});

backBtn.addEventListener("click", () => {
  turnView.classList.add("hidden");
  mapView.classList.add("hidden");
  timelineView.classList.remove("hidden");
});

prevTurnBtn.addEventListener("click", () => {
  if (!state.data) return;
  state.currentTurnIndex = Math.max(0, state.currentTurnIndex - 1);
  renderTurnTable();
});

nextTurnBtn.addEventListener("click", () => {
  if (!state.data) return;
  state.currentTurnIndex = Math.min(state.data.turns.length - 1, state.currentTurnIndex + 1);
  renderTurnTable();
});

mapViewBtn.addEventListener("click", () => {
  timelineView.classList.add("hidden");
  turnView.classList.add("hidden");
  mapView.classList.remove("hidden");
  renderMap();
});

mapBackBtn.addEventListener("click", () => {
  mapView.classList.add("hidden");
  timelineView.classList.remove("hidden");
});

mapPrevTurnBtn.addEventListener("click", () => {
  if (!state.data) return;
  state.currentTurnIndex = Math.max(0, state.currentTurnIndex - 1);
  renderMap();
});

mapNextTurnBtn.addEventListener("click", () => {
  if (!state.data) return;
  state.currentTurnIndex = Math.min(state.data.turns.length - 1, state.currentTurnIndex + 1);
  renderMap();
});

loadFiles().catch((err) => {
  console.error(err);
});
