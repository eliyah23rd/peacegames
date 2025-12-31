const state = {
  files: [],
  data: null,
  currentMetric: null,
  currentTurnIndex: 0,
};

const palette = [
  "#2a9d8f",
  "#e76f51",
  "#264653",
  "#f4a261",
  "#8ab17d",
  "#c08497",
  "#577590",
  "#f6bd60",
  "#4d908e",
  "#f3722c",
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
const mapCanvas = document.getElementById("mapCanvas");
const mapTurnLabel = document.getElementById("mapTurnLabel");
const mapPrevTurnBtn = document.getElementById("mapPrevTurnBtn");
const mapNextTurnBtn = document.getElementById("mapNextTurnBtn");
const mapBackBtn = document.getElementById("mapBackBtn");

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
  const ctx = mapCanvas.getContext("2d");
  ctx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);

  const coords = territory_names.map((name) => territory_positions[name] || [0, 0]);
  const xs = coords.map((c) => c[0]);
  const ys = coords.map((c) => c[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const padding = 40;
  const gridWidth = maxX - minX + 1;
  const gridHeight = maxY - minY + 1;
  const cellSize = Math.min(
    (mapCanvas.width - padding * 2) / Math.max(gridWidth, 1),
    (mapCanvas.height - padding * 2) / Math.max(gridHeight, 1),
  );
  const offsetX = padding;
  const offsetY = padding;

  const ownerRow = territory_owners[turnIdx] || [];
  const label = territory_names.length <= 40;
  const jitter = 0.22;
  const vertexCache = new Map();
  const edgeCache = new Map();

  const hashToUnit = (str) => {
    let h = 2166136261;
    for (let i = 0; i < str.length; i += 1) {
      h ^= str.charCodeAt(i);
      h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24);
    }
    return ((h >>> 0) % 100000) / 100000;
  };

  const jitterPoint = (key, coord) => {
    if (vertexCache.has(key)) {
      return vertexCache.get(key);
    }
    const rx = (hashToUnit(`${key}:x`) * 2 - 1) * jitter;
    const ry = (hashToUnit(`${key}:y`) * 2 - 1) * jitter;
    const pt = [coord[0] + rx, coord[1] + ry];
    vertexCache.set(key, pt);
    return pt;
  };

  const edgeKey = (a, b) => {
    const k1 = `${a[0]},${a[1]}`;
    const k2 = `${b[0]},${b[1]}`;
    return k1 < k2 ? `${k1}|${k2}` : `${k2}|${k1}`;
  };

  const sameCoord = (a, b) => a[0] === b[0] && a[1] === b[1];

  const edgePoints = (start, end) => {
    const key = edgeKey(start, end);
    if (!edgeCache.has(key)) {
      const sKey = `${start[0]},${start[1]}`;
      const eKey = `${end[0]},${end[1]}`;
      const s = jitterPoint(sKey, start);
      const e = jitterPoint(eKey, end);
      const isVertical = start[0] === end[0];
      let c1;
      let c2;
      if (isVertical) {
        const j1 = (hashToUnit(`${key}:c1`) * 2 - 1) * jitter;
        const j2 = (hashToUnit(`${key}:c2`) * 2 - 1) * jitter;
        c1 = [s[0] + j1, s[1] + (e[1] - s[1]) * 0.33];
        c2 = [s[0] + j2, s[1] + (e[1] - s[1]) * 0.66];
      } else {
        const j1 = (hashToUnit(`${key}:c1`) * 2 - 1) * jitter;
        const j2 = (hashToUnit(`${key}:c2`) * 2 - 1) * jitter;
        c1 = [s[0] + (e[0] - s[0]) * 0.33, s[1] + j1];
        c2 = [s[0] + (e[0] - s[0]) * 0.66, s[1] + j2];
      }
      edgeCache.set(key, {
        start: key.split("|")[0].split(",").map(Number),
        end: key.split("|")[1].split(",").map(Number),
        points: [s, c1, c2, e],
      });
    }
    const cached = edgeCache.get(key);
    if (sameCoord(start, cached.start)) {
      return cached.points;
    }
    return [cached.points[3], cached.points[2], cached.points[1], cached.points[0]];
  };

  const toCanvas = (pt) => [
    offsetX + (pt[0] - minX) * cellSize,
    offsetY + (pt[1] - minY) * cellSize,
  ];

  territory_names.forEach((name, idx) => {
    const [x, y] = territory_positions[name] || [0, 0];
    const owner = ownerRow[idx];
    const color = owner ? palette[state.data.agents.indexOf(owner) % palette.length] : "#e5e1dc";

    const edges = [
      edgePoints([x, y], [x + 1, y]),
      edgePoints([x + 1, y], [x + 1, y + 1]),
      edgePoints([x + 1, y + 1], [x, y + 1]),
      edgePoints([x, y + 1], [x, y]),
    ];

    ctx.beginPath();
    const start = toCanvas(edges[0][0]);
    ctx.moveTo(start[0], start[1]);
    edges.forEach((edge) => {
      const c1 = toCanvas(edge[1]);
      const c2 = toCanvas(edge[2]);
      const end = toCanvas(edge[3]);
      ctx.bezierCurveTo(c1[0], c1[1], c2[0], c2[1], end[0], end[1]);
    });
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.strokeStyle = "#5a4f4b";
    ctx.lineWidth = 1.4;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.fill();
    ctx.stroke();

    if (label) {
      const center = toCanvas([x + 0.5, y + 0.5]);
      ctx.fillStyle = "#2c2b2a";
      ctx.font = "10px Georgia";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(name, center[0], center[1]);
    }
  });
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
