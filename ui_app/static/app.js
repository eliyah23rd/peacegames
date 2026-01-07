const state = {
  files: [],
  data: null,
  currentMetric: null,
  currentTurnIndex: 0,
  currentFile: null,
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

const dataFileSelects = document.querySelectorAll(".dataFileSelect");
const metricSelect = document.getElementById("metricSelect");
const chartCanvas = document.getElementById("chartCanvas");
const chartTitle = document.getElementById("chartTitle");
const legend = document.getElementById("legend");
const allMetricsGrid = document.getElementById("allMetricsGrid");
const timelineView = document.getElementById("timelineView");
const turnView = document.getElementById("turnView");
const mapView = document.getElementById("mapView");
const turnTableWrapper = document.getElementById("turnTableWrapper");
const turnLabel = document.getElementById("turnLabel");
const prevTurnBtn = document.getElementById("prevTurnBtn");
const nextTurnBtn = document.getElementById("nextTurnBtn");
const mapImage = document.getElementById("mapImage");
const mapTurnLabel = document.getElementById("mapTurnLabel");
const mapPrevTurnBtn = document.getElementById("mapPrevTurnBtn");
const mapNextTurnBtn = document.getElementById("mapNextTurnBtn");
const mapLegend = document.getElementById("mapLegend");
const mapCapitals = document.getElementById("mapCapitals");
const experimentView = document.getElementById("experimentView");
const experimentFileSelect = document.getElementById("experimentFileSelect");
const experimentChartSelect = document.getElementById("experimentChartSelect");
const experimentSummary = document.getElementById("experimentSummary");
const experimentChart = document.getElementById("experimentChart");
const messagesView = document.getElementById("messagesView");
const messagesTurnLabel = document.getElementById("messagesTurnLabel");
const messagesPrevTurnBtn = document.getElementById("messagesPrevTurnBtn");
const messagesNextTurnBtn = document.getElementById("messagesNextTurnBtn");
const messageSenderSelect = document.getElementById("messageSenderSelect");
const messageRecipientSelect = document.getElementById("messageRecipientSelect");
const messagesList = document.getElementById("messagesList");
const newsView = document.getElementById("newsView");
const newsTurnLabel = document.getElementById("newsTurnLabel");
const newsPrevTurnBtn = document.getElementById("newsPrevTurnBtn");
const newsNextTurnBtn = document.getElementById("newsNextTurnBtn");
const newsBody = document.getElementById("newsBody");
const navTimeline = document.getElementById("navTimeline");
const navTurn = document.getElementById("navTurn");
const navMap = document.getElementById("navMap");
const navExperiment = document.getElementById("navExperiment");
const navMessages = document.getElementById("navMessages");
const navNews = document.getElementById("navNews");
const navReports = document.getElementById("navReports");
const reportsView = document.getElementById("reportsView");
const reportsTurnLabel = document.getElementById("reportsTurnLabel");
const reportsPrevTurnBtn = document.getElementById("reportsPrevTurnBtn");
const reportsNextTurnBtn = document.getElementById("reportsNextTurnBtn");
const reportAgentSelect = document.getElementById("reportAgentSelect");
const reportBody = document.getElementById("reportBody");
const constantsView = document.getElementById("constantsView");
const constantsBody = document.getElementById("constantsBody");
const navConstants = document.getElementById("navConstants");

function formatNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return String(value);
  return value.toFixed(2);
}

async function loadFiles() {
  const res = await fetch("/api/files");
  const payload = await res.json();
  state.files = payload.files || [];
  dataFileSelects.forEach((select) => {
    select.innerHTML = "";
    state.files.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      select.appendChild(opt);
    });
  });
  if (state.files.length > 0) {
    state.currentFile = state.files[0];
    dataFileSelects.forEach((select) => {
      select.value = state.currentFile;
    });
    await loadData(state.currentFile);
  }

  await loadExperimentFiles();
}

async function loadData(filename) {
  const res = await fetch(`/api/data/${encodeURIComponent(filename)}`);
  state.data = await res.json();
  state.currentFile = filename;
  state.currentTurnIndex = 0;
  populateMetrics();
  renderChart();
  renderTurnTable();
  populateMessageFilters();
  renderMessages();
  populateReportAgents();
  renderReport();
  renderNews();
  renderConstants();
}

async function loadExperimentFiles() {
  const res = await fetch("/api/experiments");
  const payload = await res.json();
  const files = payload.files || [];
  experimentFileSelect.innerHTML = "";
  files.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    experimentFileSelect.appendChild(opt);
  });
  if (files.length > 0) {
    experimentFileSelect.value = files[0];
    await loadExperiment(files[0]);
  }
}

async function loadExperiment(filename) {
  const res = await fetch(`/api/experiment?file=${encodeURIComponent(filename)}`);
  const payload = await res.json();
  experimentSummary.textContent = payload.summary || "";
  renderExperimentChart();
}

function populateMetrics() {
  if (!state.data) return;
  metricSelect.innerHTML = "";
  const vars = state.data.ledger_vars || [];
  const allOpt = document.createElement("option");
  allOpt.value = "__all__";
  allOpt.textContent = "all metrics";
  metricSelect.appendChild(allOpt);
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

function populateMessageFilters() {
  if (!state.data) return;
  const agents = state.data.agents || [];
  messageSenderSelect.innerHTML = "";
  agents.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    messageSenderSelect.appendChild(opt);
  });
  if (agents.length > 0) {
    messageSenderSelect.value = agents[0];
  }
  populateMessageRecipients();
}

function populateMessageRecipients() {
  if (!state.data) return;
  const { messages } = state.data;
  const turnIdx = state.currentTurnIndex;
  const sender = messageSenderSelect.value;
  const turnMessages = Array.isArray(messages) ? messages[turnIdx] || {} : {};
  const senderMsgs = turnMessages[sender] || {};
  messageRecipientSelect.innerHTML = "";
  const allMsgsOpt = document.createElement("option");
  allMsgsOpt.value = "__all__";
  allMsgsOpt.textContent = "<all msgs>";
  messageRecipientSelect.appendChild(allMsgsOpt);

  const recipients = Object.keys(senderMsgs).sort();
  if (recipients.includes("all")) {
    const opt = document.createElement("option");
    opt.value = "all";
    opt.textContent = "all";
    messageRecipientSelect.appendChild(opt);
  }
  recipients
    .filter((name) => name !== "all")
    .forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      messageRecipientSelect.appendChild(opt);
    });
  messageRecipientSelect.value = "__all__";
}

function populateReportAgents() {
  if (!state.data) return;
  const agents = state.data.agents || [];
  reportAgentSelect.innerHTML = "";
  agents.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    reportAgentSelect.appendChild(opt);
  });
  if (agents.length > 0) {
    reportAgentSelect.value = agents[0];
  }
}

function getMetricIndex(metric) {
  const vars = state.data?.ledger_vars || [];
  return vars.indexOf(metric);
}

function renderChart() {
  if (!state.data || !state.currentMetric) return;
  if (state.currentMetric === "__all__") {
    chartTitle.textContent = "All metrics";
    chartCanvas.classList.add("hidden");
    allMetricsGrid.classList.remove("hidden");
    renderAllMetrics();
    return;
  }
  const ctx = chartCanvas.getContext("2d");
  chartCanvas.classList.remove("hidden");
  allMetricsGrid.classList.add("hidden");
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
  ctx.fillText(formatNumber(maxVal), 8, padding + 4);
  ctx.fillText(formatNumber(minVal), 8, padding + height);

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

function renderAllMetrics() {
  if (!state.data) return;
  allMetricsGrid.innerHTML = "";
  const { agents, turns, ledger_vars, data } = state.data;
  ledger_vars.forEach((metric, metricIdx) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    const title = document.createElement("h3");
    title.textContent = metric.replace(/_/g, " ");
    const canvas = document.createElement("canvas");
    canvas.width = 240;
    canvas.height = 140;
    card.appendChild(title);
    card.appendChild(canvas);
    allMetricsGrid.appendChild(card);
    drawMiniChart(canvas, turns, agents, data, metricIdx);
  });
}

function drawMiniChart(canvas, turns, agents, data, metricIdx) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const values = data.map((agentRows) => agentRows.map((row) => row[metricIdx]));
  const allVals = values.flat();
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const padding = 18;
  const width = canvas.width - padding * 2;
  const height = canvas.height - padding * 2;

  const yScale = (val) => {
    if (maxVal === minVal) return padding + height / 2;
    return padding + height - ((val - minVal) / (maxVal - minVal)) * height;
  };
  const xScale = (idx) => {
    if (turns.length <= 1) return padding + width / 2;
    return padding + (idx / (turns.length - 1)) * width;
  };

  ctx.strokeStyle = "#eadfce";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, padding + height);
  ctx.lineTo(padding + width, padding + height);
  ctx.stroke();

  agents.forEach((agent, agentIdx) => {
    const series = values[agentIdx] || [];
    ctx.strokeStyle = palette[agentIdx % palette.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    series.forEach((val, idx) => {
      const x = xScale(idx);
      const y = yScale(val);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
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
      value.textContent = formatNumber(val);

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

function renderMessages() {
  if (!state.data) return;
  const { turns, messages } = state.data;
  const turnIdx = state.currentTurnIndex;
  const turnValue = turns[turnIdx];
  messagesTurnLabel.textContent = `Turn ${turnValue}`;

  const sender = messageSenderSelect.value;
  const recipient = messageRecipientSelect.value;
  const turnMessages = Array.isArray(messages) ? messages[turnIdx] || {} : {};

  messagesList.innerHTML = "";
  const senderMsgs = turnMessages[sender] || {};
  const filtered = Object.entries(senderMsgs).filter(([to]) => {
    if (recipient === "__all__") return true;
    return to === recipient;
  });

  if (filtered.length === 0) {
    messagesList.textContent = "No messages for this selection.";
    return;
  }
  filtered.forEach(([to, text]) => {
    const item = document.createElement("div");
    item.className = "message-item";
    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = `${sender} -> ${to}`;
    const body = document.createElement("div");
    body.textContent = text;
    item.appendChild(meta);
    item.appendChild(body);
    messagesList.appendChild(item);
  });
}

function renderNews() {
  if (!state.data) return;
  const { turns, news } = state.data;
  const turnIdx = state.currentTurnIndex;
  const turnValue = turns[turnIdx];
  newsTurnLabel.textContent = `Turn ${turnValue}`;
  const raw = Array.isArray(news) ? news[turnIdx] || "No news for this turn." : "No news for this turn.";
  const lines = raw.split("\n");
  const filtered = [];
  let skipping = false;
  for (const line of lines) {
    if (line.startsWith("Messages:")) {
      skipping = true;
      continue;
    }
    if (skipping) {
      if (line.startsWith(" - ")) {
        continue;
      }
      skipping = false;
    }
    filtered.push(line);
  }
  newsBody.textContent = filtered.join("\n");
}

function renderMap() {
  if (!state.data) return;
  const { territory_names, territory_positions, territory_owners, turns, capitals } = state.data;
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
  const capLines = [];
  if (capitals) {
    Object.keys(capitals)
      .sort()
      .forEach((agent) => {
        capLines.push(`${agent}: ${capitals[agent] || "none"}`);
      });
  }
  mapCapitals.textContent = capLines.length ? `Capitals:\n${capLines.join("\n")}` : "Capitals: none";

  const fileName = state.currentFile;
  const turnValue = turns[turnIdx];
  mapImage.src = `/api/map?file=${encodeURIComponent(fileName)}&turn=${turnValue}&t=${Date.now()}`;
}

dataFileSelects.forEach((select) => {
  select.addEventListener("change", async (e) => {
    const value = e.target.value;
    dataFileSelects.forEach((other) => {
      other.value = value;
    });
    await loadData(value);
    if (!mapView.classList.contains("hidden")) {
      renderMap();
    }
    if (!messagesView.classList.contains("hidden")) {
      renderMessages();
    }
    if (!newsView.classList.contains("hidden")) {
      renderNews();
    }
    if (!reportsView.classList.contains("hidden")) {
      renderReport();
    }
    if (!constantsView.classList.contains("hidden")) {
      renderConstants();
    }
  });
});

metricSelect.addEventListener("change", (e) => {
  state.currentMetric = e.target.value;
  renderChart();
});

function setActiveView(view) {
  timelineView.classList.add("hidden");
  turnView.classList.add("hidden");
  mapView.classList.add("hidden");
  experimentView.classList.add("hidden");
  messagesView.classList.add("hidden");
  newsView.classList.add("hidden");
  reportsView.classList.add("hidden");
  constantsView.classList.add("hidden");
  navTimeline.classList.remove("active");
  navTurn.classList.remove("active");
  navMap.classList.remove("active");
  navExperiment.classList.remove("active");
  navMessages.classList.remove("active");
  navNews.classList.remove("active");
  navReports.classList.remove("active");
  navConstants.classList.remove("active");

  if (view === "timeline") {
    timelineView.classList.remove("hidden");
    navTimeline.classList.add("active");
    renderChart();
  } else if (view === "turn") {
    turnView.classList.remove("hidden");
    navTurn.classList.add("active");
    renderTurnTable();
  } else if (view === "map") {
    mapView.classList.remove("hidden");
    navMap.classList.add("active");
    renderMap();
  } else if (view === "experiment") {
    experimentView.classList.remove("hidden");
    navExperiment.classList.add("active");
  } else if (view === "messages") {
    messagesView.classList.remove("hidden");
    navMessages.classList.add("active");
    populateMessageRecipients();
    renderMessages();
  } else if (view === "news") {
    newsView.classList.remove("hidden");
    navNews.classList.add("active");
    renderNews();
  } else if (view === "reports") {
    reportsView.classList.remove("hidden");
    navReports.classList.add("active");
    renderReport();
  } else if (view === "constants") {
    constantsView.classList.remove("hidden");
    navConstants.classList.add("active");
    renderConstants();
  }
}


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

messagesPrevTurnBtn.addEventListener("click", () => {
  if (!state.data) return;
  state.currentTurnIndex = Math.max(0, state.currentTurnIndex - 1);
  populateMessageRecipients();
  renderMessages();
});

messagesNextTurnBtn.addEventListener("click", () => {
  if (!state.data) return;
  state.currentTurnIndex = Math.min(state.data.turns.length - 1, state.currentTurnIndex + 1);
  populateMessageRecipients();
  renderMessages();
});

messageSenderSelect.addEventListener("change", () => {
  populateMessageRecipients();
  renderMessages();
});
messageRecipientSelect.addEventListener("change", () => renderMessages());

navTimeline.addEventListener("click", () => setActiveView("timeline"));
navTurn.addEventListener("click", () => setActiveView("turn"));
navMap.addEventListener("click", () => setActiveView("map"));
navExperiment.addEventListener("click", () => setActiveView("experiment"));
navMessages.addEventListener("click", () => setActiveView("messages"));
navNews.addEventListener("click", () => setActiveView("news"));
navReports.addEventListener("click", () => setActiveView("reports"));
navConstants.addEventListener("click", () => setActiveView("constants"));

function renderExperimentChart() {
  const file = experimentFileSelect.value;
  const chartType = experimentChartSelect.value;
  if (!file) {
    experimentChart.removeAttribute("src");
    return;
  }
  experimentChart.src = `/api/experiment_chart?file=${encodeURIComponent(file)}&type=${chartType}&t=${Date.now()}`;
}


experimentFileSelect.addEventListener("change", async (e) => {
  await loadExperiment(e.target.value);
});

experimentChartSelect.addEventListener("change", () => {
  renderExperimentChart();
});

function renderReport() {
  if (!state.data) return;
  const { turns, reports } = state.data;
  const turnIdx = state.currentTurnIndex;
  const turnValue = turns[turnIdx];
  reportsTurnLabel.textContent = `Turn ${turnValue}`;
  const agent = reportAgentSelect.value;
  const turnReports = Array.isArray(reports) ? reports[turnIdx] || {} : {};
  const raw = turnReports[agent] || "No report for this agent.";
  const lines = raw
    .split("\n")
    .filter((line) => !line.startsWith("Legal inbound cessions:") && !line.startsWith("Legal outbound cessions:"));
  reportBody.textContent = lines.join("\n");
}

function renderConstants() {
  if (!state.data) return;
  const constants = state.data.constants || {};
  const lines = Object.keys(constants)
    .sort()
    .map((key) => `${key}: ${constants[key]}`);
  constantsBody.textContent = lines.join("\n") || "No constants available.";
}

reportsPrevTurnBtn.addEventListener("click", () => {
  if (!state.data) return;
  state.currentTurnIndex = Math.max(0, state.currentTurnIndex - 1);
  renderReport();
});

reportsNextTurnBtn.addEventListener("click", () => {
  if (!state.data) return;
  state.currentTurnIndex = Math.min(state.data.turns.length - 1, state.currentTurnIndex + 1);
  renderReport();
});

reportAgentSelect.addEventListener("change", () => renderReport());

newsPrevTurnBtn.addEventListener("click", () => {
  if (!state.data) return;
  state.currentTurnIndex = Math.max(0, state.currentTurnIndex - 1);
  renderNews();
});

newsNextTurnBtn.addEventListener("click", () => {
  if (!state.data) return;
  state.currentTurnIndex = Math.min(state.data.turns.length - 1, state.currentTurnIndex + 1);
  renderNews();
});

loadFiles().catch((err) => {
  console.error(err);
});
