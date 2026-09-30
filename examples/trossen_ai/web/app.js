const camerasEl = document.getElementById("cameras");
const modeBadge = document.getElementById("mode-badge");
const runningBadge = document.getElementById("running-badge");
const promptInput = document.getElementById("prompt-input");
const maxStepsInput = document.getElementById("max-steps-input");
const startBtn = document.getElementById("start-btn");
const pauseBtn = document.getElementById("pause-btn");
const resumeBtn = document.getElementById("resume-btn");
const abortBtn = document.getElementById("abort-btn");
const telStep = document.getElementById("tel-step");
const telHz = document.getElementById("tel-hz");
const telLoop = document.getElementById("tel-loop");
const telPrompt = document.getElementById("tel-prompt");
const historyBody = document.getElementById("history-body");

const cameraImgs = {};

function ensureCameraTile(name) {
  if (cameraImgs[name]) return cameraImgs[name];
  const tile = document.createElement("div");
  tile.className = "camera-tile";
  const img = document.createElement("img");
  img.alt = name;
  const label = document.createElement("div");
  label.className = "camera-label";
  label.textContent = name;
  tile.appendChild(img);
  tile.appendChild(label);
  camerasEl.appendChild(tile);
  cameraImgs[name] = img;
  return img;
}

function renderStatus(status) {
  modeBadge.textContent = `modo: ${status.mode ?? "--"}`;

  const phaseLabels = {
    idle: ["parado", "badge-idle"],
    running: ["em execução", "badge-running"],
    paused: ["pausado", "badge-paused"],
    returning_to_ready: ["voltando para posição ready", "badge-running"],
  };
  const [label, cls] = phaseLabels[status.phase] ?? (status.is_running ? phaseLabels.running : phaseLabels.idle);
  runningBadge.textContent = label;
  runningBadge.className = `badge ${cls}`;

  telStep.textContent = status.max_steps ? `${status.episode_step} / ${status.max_steps}` : "--";
  telHz.textContent = status.hz ? `${status.hz.toFixed(1)} Hz` : "--";
  telLoop.textContent = status.loop_ms ? `${status.loop_ms.toFixed(1)} ms` : "--";
  telPrompt.textContent = status.prompt || "--";

  startBtn.disabled = !!status.is_running;

  const isPaused = !!status.is_paused;
  pauseBtn.hidden = !status.is_running || isPaused;
  resumeBtn.hidden = !isPaused;
  abortBtn.hidden = !status.is_running;
}

function renderCameras(cameras) {
  for (const [name, b64] of Object.entries(cameras)) {
    const img = ensureCameraTile(name);
    img.src = `data:image/jpeg;base64,${b64}`;
  }
}

function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/state`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    renderStatus(data.status);
    renderCameras(data.cameras);
  };

  ws.onclose = () => {
    setTimeout(connectWebSocket, 1000);
  };

  ws.onerror = () => {
    ws.close();
  };
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  return res.json();
}

startBtn.addEventListener("click", async () => {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    promptInput.focus();
    return;
  }
  const maxStepsValue = maxStepsInput.value.trim();
  const body = { prompt };
  if (maxStepsValue) {
    body.max_steps = parseInt(maxStepsValue, 10);
  }
  const result = await postJSON("/api/start", body);
  if (!result.ok) {
    alert(result.error || "Não foi possível iniciar a execução.");
  }
});

pauseBtn.addEventListener("click", () => {
  postJSON("/api/pause");
});

resumeBtn.addEventListener("click", () => {
  postJSON("/api/resume");
});

abortBtn.addEventListener("click", () => {
  postJSON("/api/abort");
});

function formatTime(ts) {
  if (!ts) return "--";
  return new Date(ts * 1000).toLocaleTimeString("pt-BR");
}

async function refreshHistory() {
  const res = await fetch("/api/history");
  const entries = await res.json();
  historyBody.innerHTML = "";
  for (const entry of entries) {
    const row = document.createElement("tr");

    const promptCell = document.createElement("td");
    promptCell.textContent = entry.prompt;

    const startCell = document.createElement("td");
    startCell.textContent = formatTime(entry.start_time);

    const stepsCell = document.createElement("td");
    stepsCell.textContent = entry.steps_completed;

    const reasonCell = document.createElement("td");
    reasonCell.textContent = entry.stop_reason || "em execução";

    const outcomeCell = document.createElement("td");
    const select = document.createElement("select");
    for (const [value, label] of [
      ["", "--"],
      ["success", "sucesso"],
      ["fail", "falha"],
    ]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      if (entry.outcome === value || (!entry.outcome && value === "")) {
        option.selected = true;
      }
      select.appendChild(option);
    }
    select.addEventListener("change", () => {
      postJSON(`/api/history/${entry.id}/outcome`, { outcome: select.value });
    });
    outcomeCell.appendChild(select);

    row.append(promptCell, startCell, stepsCell, reasonCell, outcomeCell);
    historyBody.appendChild(row);
  }
}

connectWebSocket();
refreshHistory();
setInterval(refreshHistory, 3000);
