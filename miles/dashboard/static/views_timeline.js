import { api } from "./api.js";
import { el, fmtNum, setViewCleanup, statBox } from "./app.js";
import { createCarpet } from "./carpet.js";
import { createFleet } from "./fleet.js";
import { createDataPool } from "./data_pool.js";
import { extent, hideTooltip, showTooltip } from "./charts.js";

// idle states share a light neutral family (train_wait also gets a hatch
// texture below — the accessibility channel — so idle doesn't lean on hue
// alone); the log_probs family shares one hue at three lightness steps;
// rollout gets the brand orange as the headline phase. Validated against
// the design system's 8-hue categorical order (scripts/validate_palette.js).
const PHASE_COLORS = {
  initialize: "#ded6ca",
  rollout: "#d55816",
  eval_rollout: "#e8a683",
  actor_train: "#2a78d6",
  train_log_probs: "#c3b8ec",
  log_probs: "#8b7bd8",
  ref_log_probs: "#4a3aa7",
  data_preprocess: "#eda100",
  train_wait: "#efe8dc",
  update_weights: "#e87ba4",
  ref_model_update: "#008300",
  save_model: "#1baf7a",
  sleep: "#e5ddd0",
  wake_up: "#e34948",
};
const DEFAULT_PHASE_COLOR = "#9c9488";
const OVERLAY_METRICS = [
  "sglang_num_running_reqs",
  "sglang_gen_throughput",
  "sglang_token_usage",
  "sglang_cache_hit_rate",
];
const OVERLAY_COLOR = "#b84a12";
const MEM_COLOR = "#1baf7a";
const UTIL_STROKE = "rgba(42, 120, 214, 0.9)";

const UTIL_FILL = "rgba(42, 120, 214, 0.14)";
const TEXT = "#231f1c";
const MUTED = "#7a7168";
const GRID = "#e8e1d8";

const FOLLOW_REDRAW_MS = 5000;
const LANE_CAP = 32;
const DEFAULT_LANES = 8; // no-selection default: evenly spaced across the cluster
const QUICK_PICKS = [
  { label: "pick: lowest util", criterion: "lowest_util" },
  { label: "pick: slowest update_weights", criterion: "slowest_phase:update_weights" },
];
const QUICK_PICK_K = 8;

const LANE_H = 66;
const PHASE_H = 14;
const UTIL_H = 40;
const M_LEFT = 96;
const M_TOP = 24;
const M_RIGHT = 14;

// relative elapsed since T0, auto-scaled so a multi-hour/day run stays
// readable instead of "+186:23" (Shang's ask): mm:ss below an hour, h:mm:ss
// below a day, d hh:mm:ss beyond that
function formatElapsed(seconds) {
  const rel = Math.max(0, Math.round(seconds));
  const pad = (n) => String(n).padStart(2, "0");
  const days = Math.floor(rel / 86400);
  const hours = Math.floor((rel % 86400) / 3600);
  const mins = Math.floor((rel % 3600) / 60);
  const secs = rel % 60;
  if (days > 0) return `+${days}d ${pad(hours)}:${pad(mins)}:${pad(secs)}`;
  if (hours > 0) return `+${hours}:${pad(mins)}:${pad(secs)}`;
  return `+${mins}:${pad(secs)}`;
}

export async function renderTimeline(view, meta, route) {
  // mutable data state, reloaded on every follow-mode refresh
  let selection = route?.lanes || null; // lane-selection grammar string
  let autoDefault = !selection; // seed the spaced default once topology is known
  let resolvedKeys = null; // full "node:gpu" set the grammar resolved to (null = no pool)
  let capped = 0; // lanes hidden by LANE_CAP
  let lanes = [];
  let windows = [];
  let phasesByLane = new Map();
  let processesByLane = new Map();
  let advisories = [];
  let mfu = null;
  let bubbles = [];
  let gpu = {};
  let engineSeries = [];
  let T0 = 0;
  let T1 = 1;
  let v0 = 0;
  let v1 = 1;
  let haveData = false;
  let overlayMetric = meta.capabilities.has_engine_series ? OVERLAY_METRICS[0] : null;
  let showMem = false;
  let showAbsolute = false; // x-axis ticks: relative "+h:mm:ss" (default) vs wall-clock
  let multiNode = false;
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const laneKey = (l) => `${l.node}:${l.gpu}`;
  // hard viewport cap (design §17): the lane view never displays more than
  // MAXW at once, so every data fetch is O(window), never O(run)
  const MAXW = meta.capabilities.max_window_s ?? Infinity;
  let allLanes = []; // full topology lane list (seeds the spaced default selection)
  let fetched = null; // {t0, t1} bounds the loaded phases/gpu/engine data covers

  // T0/T1 come from meta.time_range (an O(1) edge-stamp read server-side),
  // not from scanning fetched data. Keeps the trailing window pinned to the
  // growing run unless the user has panned away.
  function applyRange(range) {
    if (!range) {
      haveData = false;
      return;
    }
    const trailing = !haveData || v1 >= T1 - 1e-6;
    const span = haveData ? v1 - v0 : Math.min(MAXW, range[1] - range[0] || 1);
    [T0, T1] = range;
    if (trailing) {
      v1 = T1;
      v0 = Math.max(T0, T1 - Math.min(span, MAXW));
    }
    if (v1 - v0 < 1) v0 = Math.max(T0, v1 - 1);
    haveData = true;
  }

  let laneRequest = 0;
  async function loadData() {
    const request = ++laneRequest;
    const [topology, bubblesRes] = await Promise.all([
      api("/api/timeline/topology"),
      api("/api/timeline/bubbles"),
    ]);
    allLanes = topology.lanes;
    windows = topology.windows;
    bubbles = bubblesRes.bubbles;
    if (!haveData) {
      lanes = [];
      gpu = {};
      phasesByLane = new Map();
      processesByLane = new Map();
      return;
    }

    // fetch a margin beyond the viewport so small pans redraw without a reload
    const span = v1 - v0;
    const margin = Math.max(0, Math.min(span / 4, (MAXW - span) / 2));
    const f0 = Math.max(T0, v0 - margin);
    const f1 = Math.min(T1, v1 + margin);
    // first sight of the topology: DEFAULT_LANES evenly spaced across the
    // global index range (8 gpus -> all, 64 -> g0,g8,...,g56) becomes a REAL
    // selection — visible chips the user removes/extends/overwrites, no
    // parallel "default budget" state to reason about
    if (autoDefault && allLanes.length > 0) {
      autoDefault = false;
      if (allLanes.length > DEFAULT_LANES) {
        selection = Array.from(
          { length: DEFAULT_LANES },
          (_, i) => `g:${allLanes[Math.floor((i * allLanes.length) / DEFAULT_LANES)].index}`,
        ).join(", ");
        try { history.replaceState(null, "", `#/timeline?lanes=${encodeURIComponent(selection)}`); } catch { /* sandbox */ }
      }
    }
    const [phasesRes, gpuRes, processesRes, overlay] = await Promise.all([
      api("/api/timeline/phases", { t0: f0, t1: f1, lanes: selection }),
      api("/api/timeline/gpu", { t0: f0, t1: f1, max_points: 4000, lanes: selection }),
      api("/api/timeline/gpu_processes", { t0: f0, t1: f1, lanes: selection }),
      overlayMetric
        ? api("/api/timeline/engine_series", { metric: overlayMetric, t0: f0, t1: f1, max_points: 4000 })
        : null,
    ]);
    if (request !== laneRequest) return;
    engineSeries = overlay?.series ?? [];
    fetched = { t0: f0, t1: f1 };
    lanes = allLanes;
    resolvedKeys = null;
    if (selection) {
      // the filtered responses reveal which lanes the grammar resolved to
      const keys = new Set(Object.keys(gpuRes.lanes));
      for (const p of phasesRes.phases) keys.add(`${p.node}:${p.gpu}`);
      lanes = lanes.filter((l) => keys.has(laneKey(l)));
      resolvedKeys = new Set(lanes.map(laneKey));
    }
    const total = lanes.length;
    if (lanes.length > LANE_CAP) lanes = lanes.slice(0, LANE_CAP);
    capped = Math.max(0, total - lanes.length);
    gpu = gpuRes.lanes;
    multiNode = new Set(lanes.map((l) => l.node)).size > 1;
    phasesByLane = new Map(lanes.map((l) => [laneKey(l), []]));
    for (const p of phasesRes.phases) phasesByLane.get(`${p.node}:${p.gpu}`)?.push(p);
    processesByLane = new Map(lanes.map((l) => [laneKey(l), []]));
    for (const p of processesRes.processes) processesByLane.get(`${p.node}:${p.gpu}`)?.push(p);
  }

  // whole-run heuristic suggestions (design §config-tuning-advisory): cheap
  // to recompute but not viewport-scoped, so it refreshes on follow-mode
  // ticks rather than on every pan/zoom like loadData()
  async function loadAdvisories() {
    try {
      const res = await api("/api/advisory");
      advisories = res.advisories;
      mfu = res.mfu;
    } catch {
      /* transient fetch failure: keep the last-known list */
    }
  }

  // pan/zoom past the fetched bounds: debounced windowed reload
  let windowTimer = null;
  let panFrame = 0;
  function afterViewChange() {
    if (!panFrame) panFrame = requestAnimationFrame(() => {
      panFrame = 0;
      draw();
      dataPool.pan();
      if (manyLanes()) fleet.pan();
    });
    scheduleWindowLoad();
  }

  // external engines (no miles actor) have unknown GPU placement (gpus=[]):
  // fall back to host identity — the engine's addr host IS its node
  const engineHost = (addr) => addr.split("//").pop().split(":")[0];
  const engineAt = (node, gpuId, t) => {
    for (const w of windows) {
      if (t >= w.t0 && (w.t1 === null || t < w.t1)) {
        for (const e of w.engines) {
          const match = e.gpus.length
            ? e.gpus.some(([n, g]) => n === node && g === gpuId)
            : engineHost(e.addr) === node;
          if (match) return e.addr;
        }
      }
    }
    return null;
  };

  // ------------------------------- toolbar ----------------------------------
  const toolbar = el("div", { class: "controls" });
  const overlayScale = el("span", { style: `color: ${OVERLAY_COLOR}; font-size: 12px` });
  const followBadge = el("span", { class: "muted" });
  const renderToolbar = () => {
    const chips = OVERLAY_METRICS.map((m) =>
      el(
        "button",
        {
          class: m === overlayMetric ? "active" : "",
          onclick: async () => {
            overlayMetric = m === overlayMetric ? null : m;
            renderToolbar();
            await loadData();
            renderAll();
          },
        },
        [m.replace("sglang_", "")],
      ),
    );
    const memBtn = el(
      "button",
      { class: showMem ? "active" : "", onclick: () => ((showMem = !showMem), renderToolbar(), draw()) },
      ["mem"],
    );
    const absBtn = el(
      "button",
      { class: showAbsolute ? "active" : "", onclick: () => ((showAbsolute = !showAbsolute), renderToolbar(), dataPool.redraw(), draw()) },
      ["abs time"],
    );
    toolbar.replaceChildren(
      el("span", { class: "muted" }, ["Overlay"]),
      ...(meta.capabilities.has_engine_series ? chips : [el("span", { class: "muted" }, ["(no engine series)"])]),
      memBtn,
      absBtn,
      el(
        "button",
        { onclick: () => ((v1 = T1), (v0 = Math.max(T0, T1 - MAXW)), afterViewChange()) },
        ["latest window"],
      ),
      overlayScale,
      followBadge,
      ...(meta.data_buffer_length != null
        ? [el("span", { class: "muted" }, [`databuffer: ${meta.data_buffer_length} groups`])]
        : []),
      el("span", { class: "muted" }, [`Wheel = zoom · drag = pan · window ≤ ${Math.round(MAXW / 3600)}h`]),
    );
  };

  // ----------------------------- lane selection -----------------------------
  const selRow = el("div", { class: "controls" });
  const selError = el("span", { class: "error", style: "padding: 0" });
  const selTerms = () => (selection ? selection.split(",").map((s) => s.trim()).filter(Boolean) : []);

  async function setSelection(grammar) {
    selection = grammar || null;
    try { history.replaceState(null, "", selection ? `#/timeline?lanes=${encodeURIComponent(selection)}` : "#/timeline"); } catch { /* sandbox */ }
    selError.textContent = "";
    renderSelection();
    try {
      await loadData();
    } catch (err) {
      selError.textContent = String(err);
      return;
    }
    renderAll();
    if (!manyLanes()) carpet.redraw(); // selection markers
  }

  function renderSelection() {
    const input = el("input", {
      type: "text",
      placeholder: "add: g:0-31 · rank:0-7 · node:<ip> · gpu:<node>:<i> · engine:<addr> · role:train · every:128",
      style: "flex: 1; min-width: 280px",
    });
    input.onkeydown = (ev) => {
      if (ev.key === "Enter" && input.value.trim()) {
        setSelection([...selTerms(), input.value.trim()].join(", "));
      }
    };
    selRow.replaceChildren(
      el("span", { class: "muted" }, ["Lanes"]),
      ...selTerms().map((term) =>
        el("span", { class: "chip" }, [
          term,
          el(
            "button",
            { onclick: () => setSelection(selTerms().filter((t) => t !== term).join(", ")) },
            ["×"],
          ),
        ]),
      ),
      ...(selection ? [] : [el("span", { class: "muted" }, ["all"])]),
      input,
      ...QUICK_PICKS.map(({ label, criterion }) =>
        el(
          "button",
          {
            onclick: async () => {
              const res = await api("/api/timeline/outliers", { criterion, top_k: QUICK_PICK_K });
              if (res.outliers.length) setSelection(res.outliers.map((o) => `gpu:${o.node}:${o.gpu}`).join(", "));
            },
          },
          [label],
        ),
      ),
      ...(selection ? [el("button", { onclick: () => setSelection(null) }, ["Clear"])] : []),
      ...(capped
        ? [el("span", { class: "warn" }, [`${capped + lanes.length} lanes; showing first ${lanes.length}`])]
        : []),
      selError,
    );
  }

  // -------------------------------- carpet ----------------------------------
  const carpet = createCarpet({
    phaseColors: PHASE_COLORS,
    runStart: () => T0,
    selectedKeys: () => resolvedKeys ?? new Set(),
    // toggle semantics: brushed lanes join the detailed pool; a brush that only
    // covers already-pooled lanes removes them instead
    onBrush: (keys) => {
      const pool = resolvedKeys ?? new Set();
      if (keys.every((k) => pool.has(k))) {
        const kept = [...pool].filter((k) => !keys.includes(k));
        setSelection(kept.map((k) => `gpu:${k}`).join(", "));
      } else {
        const added = keys.filter((k) => !pool.has(k)).map((k) => `gpu:${k}`);
        setSelection([...selTerms(), ...added].join(", "));
      }
    },
  });
  const viewport = () => (haveData ? { t0: v0, t1: v1 } : {});
  const fleet = createFleet({ phaseColors: PHASE_COLORS, runStart: () => T0, viewport: viewport });
  const dataPool = createDataPool(meta, () => T0, () => showAbsolute, viewport);

  // above this many lanes the per-rank carpet stops scaling (payload and
  // pixels are both O(lanes)); serve --dashboard-use-utilization-overview forces it
  const CARPET_MAX_LANES = 64;

  // the overview (carpet or fleet panels) tracks the lane view's zoom window;
  // before any data exists it asks for the (empty) server default
  const manyLanes = () => meta.capabilities.use_utilization_overview || allLanes.length > CARPET_MAX_LANES;
  async function refreshOverview() {
    carpet.root.style.display = manyLanes() ? "none" : "";
    fleet.root.style.display = manyLanes() ? "" : "none";
    await Promise.all([dataPool.refresh(viewport()), manyLanes() ? fleet.refresh(viewport()) : carpet.refresh(viewport())]);
  }
  let windowRevision = 0;
  const changingWindow = () => windowTimer !== null || editingWindow || movingWindow;
  function scheduleWindowLoad(delay = 200) {
    const revision = ++windowRevision;
    loadStatus.textContent = "Loading this time window…";
    retryWindow.hidden = true;
    clearTimeout(windowTimer);
    windowTimer = setTimeout(async () => {
      try {
        const needLanes = !fetched || v0 < fetched.t0 || v1 > fetched.t1;
        await Promise.all([refreshOverview(), needLanes ? loadData() : null]);
        if (revision !== windowRevision) return;
        renderAll();
        loadStatus.textContent = "";
      } catch (error) {
        if (revision !== windowRevision) return;
        loadStatus.textContent = `Could not load this time window. ${error.message}`;
        retryWindow.hidden = false;
      } finally {
        if (revision === windowRevision) windowTimer = null;
      }
    }, delay);
  }

  // ----------------------------- bubble strip -------------------------------
  const bubbleStrip = el("div", { class: "bubblestrip" });
  const renderBubbles = () => {
    if (!bubbles.length) {
      bubbleStrip.replaceChildren();
      return;
    }
    const worst = Math.max(...bubbles.map((b) => b.wait_ratio ?? 0), 0.01);
    bubbleStrip.replaceChildren(el("span", { class: "muted" }, ["Wait ratio / step: "]));
    for (const b of bubbles) {
      const cell = el("span", { class: "bubble" }, [String(b.step)]);
      const ratio = b.wait_ratio ?? 0;
      cell.style.background = `rgba(224, 96, 96, ${(0.85 * ratio) / worst})`;
      cell.onclick = () => {
        if (b.step_time !== null) {
          v1 = b.ts;
          v0 = Math.max(b.ts - b.step_time, b.ts - MAXW);
          afterViewChange();
        }
      };
      cell.onmousemove = (ev) =>
        showTooltip(
          ev.clientX,
          ev.clientY,
          `step ${b.step}\nwait_ratio = ${fmtNum(ratio)}\nstep_time = ${fmtNum(b.step_time)}s`,
        );
      cell.onmouseleave = hideTooltip;
      bubbleStrip.append(cell);
    }
  };

  // -------------------------------- canvas ----------------------------------
  const canvas = el("canvas", { class: "timeline" });
  const dateField = name => el("input", { type: "datetime-local", step: "1", required: "", "aria-label": name });
  const from = dateField("Window start");
  const to = dateField("Window end");
  const loadStatus = el("span", { role: "status", class: "muted" });
  const retryWindow = el("button", { type: "button", hidden: "", onclick: () => scheduleWindowLoad(0) }, ["Retry"]);
  const editor = el("form", { class: "controls" }, [
    el("label", {}, ["From ", from]), el("label", {}, ["To ", to]), el("button", { type: "submit" }, ["Apply"]),
    el("span", { class: "muted" }, [Intl.DateTimeFormat().resolvedOptions().timeZone]), loadStatus, retryWindow,
  ]);
  const rail = el("div", { "aria-label": "Time window", style:
    "position:relative;height:24px;background:var(--border);border-radius:4px;touch-action:none;user-select:none;cursor:pointer" });
  const grip = side => el("span", { "data-edge": side, title: "Resize " + side, style:
    `position:absolute;${side}:0;width:8px;top:0;bottom:0;border-${side}:3px solid var(--accent);cursor:ew-resize` });
  const selectionBar = el("div", { title: "Drag to move the time window", style:
    "position:absolute;height:100%;background:rgba(42,120,214,.35);border-radius:3px;cursor:grab" },
    [grip("left"), grip("right")]);
  rail.append(selectionBar);
  const timeBar = el("div", { class: "panel", style: "position:sticky;top:46px;z-index:5" }, [editor, rail]);
  let editingWindow = false;
  let movingWindow = null;
  function localDate(t) {
    const date = new Date(Math.round(t) * 1000);
    return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 19);
  }
  function updateTimeBar() {
    if (!haveData || T1 <= T0) return;
    selectionBar.style.left = (100 * (v0 - T0) / (T1 - T0)) + "%";
    selectionBar.style.width = (100 * (v1 - v0) / (T1 - T0)) + "%";
    if (!editingWindow) {
      from.value = localDate(v0);
      to.value = localDate(v1);
    }
  }
  function setWindow(a, b) {
    v0 = a;
    v1 = b;
    editingWindow = false;
    to.setCustomValidity("");
    afterViewChange();
  }
  editor.oninput = () => { editingWindow = true; to.setCustomValidity(""); };
  editor.onsubmit = ev => {
    ev.preventDefault();
    if (!haveData) return;
    // Displayed bounds have one-second precision; retain fractional run edges.
    const a = from.value === localDate(T0) ? T0 : Math.max(T0, new Date(from.value).getTime() / 1000);
    const b = to.value === localDate(T1) ? T1 : Math.min(T1, new Date(to.value).getTime() / 1000);
    if (!(b > a) || b - a > MAXW + 1) {
      to.setCustomValidity("Choose an interval within this run, up to " + Math.round(MAXW / 3600) + " hours.");
      to.reportValidity();
      return;
    }
    setWindow(a, Math.min(b, a + MAXW));
  };
  const railTime = x => {
    const rect = rail.getBoundingClientRect();
    return T0 + (x - rect.left) / rect.width * (T1 - T0);
  };
  function moveWindow(a, span) {
    const start = clamp(a, T0, T1 - span);
    setWindow(start, start + span);
  }
  rail.onpointerdown = ev => {
    if (!haveData || ev.button !== 0 || movingWindow) return;
    const t = railTime(ev.clientX);
    if (ev.target === rail) moveWindow(t - (v1 - v0) / 2, v1 - v0);
    movingWindow = { id: ev.pointerId, edge: ev.target.dataset.edge, anchor: t, start: v0, end: v1 };
    rail.setPointerCapture(ev.pointerId);
  };
  rail.onpointermove = ev => {
    if (!movingWindow || ev.pointerId !== movingWindow.id) return;
    const { edge, anchor, start, end } = movingWindow;
    const delta = railTime(ev.clientX) - anchor;
    const minimum = Math.min(1, T1 - T0);
    if (edge === "left") setWindow(clamp(start + delta, Math.max(T0, end - MAXW), end - minimum), end);
    else if (edge === "right") setWindow(start, clamp(end + delta, start + minimum, Math.min(T1, start + MAXW)));
    else moveWindow(start + delta, end - start);
  };
  rail.onlostpointercapture = () => { movingWindow = null; };
  const overlayMax = () => Math.max(extent(...engineSeries.map((s) => s.value))[1], 1e-9);
  const memMax = () => Math.max(extent(...Object.values(gpu).map((s) => s.mem_mb))[1], 1);

  function draw() {
    updateTimeBar();
    canvas.style.height = `${M_TOP + Math.max(lanes.length, 1) * LANE_H + 8}px`;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const width = rect.width;
    const plotW = width - M_LEFT - M_RIGHT;
    ctx.font = "11px ui-monospace, monospace";
    if (!haveData) {
      ctx.fillStyle = MUTED;
      ctx.fillText("waiting for timeline data…", M_LEFT, M_TOP + 20);
      overlayScale.textContent = "";
      return;
    }
    const X = (t) => M_LEFT + ((t - v0) / (v1 - v0)) * plotW;

    ctx.fillStyle = MUTED;
    ctx.strokeStyle = GRID;
    const span = v1 - v0;
    const tickStep = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600].find((s) => span / s <= 10) || 7200;
    for (let t = Math.ceil((v0 - T0) / tickStep) * tickStep + T0; t <= v1; t += tickStep) {
      const label = showAbsolute ? new Date(t * 1000).toLocaleTimeString() : formatElapsed(t - T0);
      ctx.fillText(label, X(t) - Math.min(30, label.length * 3.2), 12);
      ctx.beginPath();
      ctx.moveTo(X(t), M_TOP - 6);
      ctx.lineTo(X(t), M_TOP + lanes.length * LANE_H);
      ctx.stroke();
    }

    lanes.forEach((lane, i) => {
      const key = laneKey(lane);
      const yPhase = M_TOP + i * LANE_H + 4;
      const yUtilTop = yPhase + PHASE_H + 4;
      const yUtilBot = yUtilTop + UTIL_H;

      ctx.fillStyle = TEXT;
      ctx.fillText(`g${lane.index}`, 8, yUtilTop + 4);
      if (multiNode) {
        ctx.fillStyle = MUTED;
        ctx.fillText(`${lane.node}:${lane.gpu}`, 8, yUtilTop + 18);
      }
      ctx.strokeStyle = GRID;
      ctx.beginPath();
      ctx.moveTo(M_LEFT, yUtilBot);
      ctx.lineTo(width - M_RIGHT, yUtilBot);
      ctx.stroke();

      for (const p of phasesByLane.get(key) ?? []) {
        if (p.t1 < v0 || p.t0 > v1) continue;
        const x0 = Math.max(X(p.t0), M_LEFT);
        const x1 = Math.min(X(p.t1), width - M_RIGHT);
        ctx.fillStyle = PHASE_COLORS[p.name] ?? DEFAULT_PHASE_COLOR;
        ctx.fillRect(x0, yPhase, Math.max(x1 - x0, 1), PHASE_H);
        if (p.name === "train_wait") {
          // hatch idle explicitly: bubbles must be visible, not blank
          ctx.strokeStyle = MUTED;
          ctx.save();
          ctx.beginPath();
          ctx.rect(x0, yPhase, x1 - x0, PHASE_H);
          ctx.clip();
          for (let x = x0 - PHASE_H; x < x1; x += 5) {
            ctx.beginPath();
            ctx.moveTo(x, yPhase + PHASE_H);
            ctx.lineTo(x + PHASE_H, yPhase);
            ctx.stroke();
          }
          ctx.restore();
        }
      }

      const series = gpu[key];
      if (series && series.ts.length) {
        const yOf = (u) => yUtilBot - (u / 100) * UTIL_H;
        ctx.beginPath();
        let started = false;
        for (let j = 0; j < series.ts.length; j++) {
          const t = series.ts[j];
          if (t < v0 || t > v1) continue;
          const x = X(t);
          const y = yOf(series.util[j]);
          started ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
          started = true;
        }
        ctx.strokeStyle = UTIL_STROKE;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.lineTo(X(Math.min(series.ts[series.ts.length - 1], v1)), yUtilBot);
        ctx.lineTo(X(Math.max(series.ts[0], v0)), yUtilBot);
        ctx.closePath();
        ctx.fillStyle = UTIL_FILL;
        ctx.fill();

        if (showMem) {
          const scale = memMax();
          ctx.beginPath();
          started = false;
          for (let j = 0; j < series.ts.length; j++) {
            const t = series.ts[j];
            if (t < v0 || t > v1) continue;
            const y = yUtilBot - (series.mem_mb[j] / scale) * UTIL_H;
            started ? ctx.lineTo(X(t), y) : ctx.moveTo(X(t), y);
            started = true;
          }
          ctx.strokeStyle = MEM_COLOR;
          ctx.stroke();
        }
      }

      if (overlayMetric && engineSeries.length) {
        const scale = overlayMax();
        for (const w of windows) {
          const w0 = Math.max(w.t0, v0);
          const w1 = Math.min(w.t1 ?? v1, v1);
          if (w0 >= w1) continue;
          const addr = engineAt(lane.node, lane.gpu, (w0 + w1) / 2);
          const s = engineSeries.find((es) => es.addr === addr);
          if (!s) continue;
          ctx.beginPath();
          let on = false;
          for (let j = 0; j < s.ts.length; j++) {
            const t = s.ts[j];
            if (t < w0 || t > w1) continue;
            const y = yUtilBot - (s.value[j] / scale) * UTIL_H;
            on ? ctx.lineTo(X(t), y) : ctx.moveTo(X(t), y);
            on = true;
          }
          ctx.strokeStyle = OVERLAY_COLOR;
          ctx.lineWidth = 1.6;
          ctx.stroke();
          ctx.lineWidth = 1;
        }
      }
    });

    overlayScale.textContent =
      overlayMetric && engineSeries.length
        ? `${overlayMetric.replace("sglang_", "")} scale 0–${fmtNum(overlayMax())}`
        : "";
  }

  // freshness: the newest data timestamp (server-side T1), not the browser
  // fetch time — a stalled collector must READ as stale despite live redraws
  const renderFreshness = () => {
    const stamp = haveData ? new Date(T1 * 1000).toLocaleTimeString() : "…";
    followBadge.textContent = meta.mode === "follow" ? `live · data → ${stamp}` : `data → ${stamp}`;
  };

  function renderAll() {
    renderFreshness();
    renderSelection();
    renderBubbles();
    renderLegend();
    renderMfu();
    renderAdvisories();
    draw();
  }

  // ------------------------------ interactions ------------------------------
  const timeAt = (clientX) => {
    const rect = canvas.getBoundingClientRect();
    return v0 + ((clientX - rect.left - M_LEFT) / (rect.width - M_LEFT - M_RIGHT)) * (v1 - v0);
  };
  canvas.onwheel = (ev) => {
    if (!haveData) return;
    ev.preventDefault();
    const pivot = timeAt(ev.clientX);
    const factor = Math.exp(ev.deltaY * 0.002);
    v0 = Math.max(T0, pivot + (v0 - pivot) * factor);
    v1 = Math.min(T1, pivot + (v1 - pivot) * factor);
    if (v1 - v0 > MAXW) {
      // zoom-out stops at the viewport cap, anchored at the cursor
      const anchor = (pivot - v0) / (v1 - v0);
      v0 = Math.max(T0, pivot - anchor * MAXW);
      v1 = v0 + MAXW;
    }
    afterViewChange();
  };
  let dragFrom = null;
  canvas.onmousedown = (ev) => (dragFrom = { x: ev.clientX, v0, v1 });
  const onMouseUp = () => (dragFrom = null);
  window.addEventListener("mouseup", onMouseUp);
  canvas.ondblclick = () => ((v1 = T1), (v0 = Math.max(T0, T1 - MAXW)), afterViewChange());
  canvas.onmousemove = (ev) => {
    if (!haveData) return;
    if (dragFrom) {
      const rect = canvas.getBoundingClientRect();
      const dt = ((dragFrom.x - ev.clientX) / (rect.width - M_LEFT - M_RIGHT)) * (dragFrom.v1 - dragFrom.v0);
      const span = dragFrom.v1 - dragFrom.v0;
      moveWindow(dragFrom.v0 + dt, span);
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const laneIdx = Math.floor((ev.clientY - rect.top - M_TOP) / LANE_H);
    if (laneIdx < 0 || laneIdx >= lanes.length || ev.clientX - rect.left < M_LEFT) {
      hideTooltip();
      return;
    }
    const lane = lanes[laneIdx];
    const key = laneKey(lane);
    const t = timeAt(ev.clientX);
    const lines = [`g${lane.index} ${key}  ${showAbsolute ? new Date(t * 1000).toLocaleTimeString() : formatElapsed(t - T0)}`];
    const phase = (phasesByLane.get(key) ?? []).find((p) => p.t0 <= t && t < p.t1);
    if (phase) lines.push(`phase: ${phase.name}${phase.rank >= 0 ? ` (rank ${phase.rank})` : ""}`);
    const series = gpu[key];
    if (series && series.ts.length) {
      let best = 0;
      for (let j = 0; j < series.ts.length; j++) {
        if (Math.abs(series.ts[j] - t) < Math.abs(series.ts[best] - t)) best = j;
      }
      lines.push(`util: ${series.util[best]}%  mem: ${fmtNum(series.mem_mb[best] / 1024)}G`);
    }
    const procs = processesByLane.get(key) ?? [];
    if (procs.length) {
      // per-process snapshots are far coarser than the util/mem series —
      // snap to the nearest snapshot's own ts, then list every process in it
      let nearestTs = procs[0].ts;
      for (const p of procs) if (Math.abs(p.ts - t) < Math.abs(nearestTs - t)) nearestTs = p.ts;
      const top = procs
        .filter((p) => p.ts === nearestTs)
        .sort((a, b) => b.mem_mb - a.mem_mb)
        .slice(0, 3);
      lines.push("top processes:", ...top.map((p) => `  ${p.name} (pid ${p.pid}): ${fmtNum(p.mem_mb / 1024)}G`));
    }
    if (overlayMetric) {
      const addr = engineAt(lane.node, lane.gpu, t);
      const s = engineSeries.find((es) => es.addr === addr);
      if (addr && s && s.ts.length) {
        let best = 0;
        for (let j = 0; j < s.ts.length; j++) if (Math.abs(s.ts[j] - t) < Math.abs(s.ts[best] - t)) best = j;
        lines.push(`${overlayMetric.replace("sglang_", "")}: ${fmtNum(s.value[best])}`, `engine: ${addr}`);
      }
    }
    showTooltip(ev.clientX, ev.clientY, lines.join("\n"));
  };
  canvas.onmouseleave = hideTooltip;
  for (const panel of [dataPool.root, fleet.root]) {
    panel.addEventListener("wheel", e => { if (e.target.tagName === "CANVAS") canvas.onwheel(e); }, {passive: false});
    panel.addEventListener("mousedown", e => { if (e.target.tagName === "CANVAS") canvas.onmousedown(e); });
    panel.addEventListener("mousemove", e => { if (dragFrom) canvas.onmousemove(e); });
    panel.addEventListener("dblclick", () => canvas.ondblclick());
  }

  // -------------------------------- legend ----------------------------------
  const legendPanel = el("div", { class: "panel" });
  const renderLegend = () => {
    const present = new Set();
    for (const items of phasesByLane.values()) for (const p of items) present.add(p.name);
    legendPanel.replaceChildren(
      el("h3", {}, ["Legend"]),
      el("div", { class: "legend" }, [
        ...[...present].sort().map((name) => {
          const swatch = el("span", {
            class: "bar",
            style: `width: 12px; background: ${PHASE_COLORS[name] ?? DEFAULT_PHASE_COLOR}`,
          });
          return el("span", { style: "display: inline-flex; gap: 4px; align-items: center" }, [swatch, name]);
        }),
        ...(overlayMetric
          ? [el("span", { style: `color: ${OVERLAY_COLOR}` }, [`— ${overlayMetric.replace("sglang_", "")}`])]
          : []),
        el("span", { style: `color: ${UTIL_STROKE}` }, ["— gpu util"]),
      ]),
    );
  };

  // --------------------------------- MFU tile --------------------------------
  const mfuPanel = el("div", {});
  const renderMfu = () => {
    if (!mfu) {
      mfuPanel.replaceChildren();
      return;
    }
    const pct = (v) => `${(v * 100).toFixed(1)}%`;
    mfuPanel.replaceChildren(
      el("div", { class: "panel" }, [
        el("h3", {}, ["Model FLOPs utilization"]),
        el("div", { class: "statgrid" }, [
          statBox("latest", pct(mfu.latest)),
          statBox(`mean over ${mfu.steps} step${mfu.steps === 1 ? "" : "s"}`, pct(mfu.mean)),
          statBox("device peak (dense bf16)", `${fmtNum(mfu.peak)} TFLOP/s`),
        ]),
        el("p", { class: "muted" }, [
          "3x forward FLOPs of the model, per GPU, over the whole actor train step. Activation recompute, ",
          "optimizer work and a CPU-offloaded optimizer all add time without adding model FLOPs, so they ",
          "lower this legitimately. Extra heads the FLOPs model does not cover (e.g. MTP) understate it.",
        ]),
      ]),
    );
  };

  // ------------------------------ config advisory ----------------------------
  const advisoryPanel = el("div", {});
  const renderAdvisories = () => {
    if (!advisories.length) {
      advisoryPanel.replaceChildren();
      return;
    }
    advisoryPanel.replaceChildren(
      el("div", { class: "panel" }, [
        el("h3", {}, ["Run advisory"]),
        ...advisories.map((a) =>
          el(
            "p",
            {
              class: a.level === "info" ? "muted" : "error",
              ...(a.level === "critical" ? { style: "font-weight: 600" } : {}),
            },
            [a.message],
          ),
        ),
      ]),
    );
  };

  view.replaceChildren(
    toolbar,
    selRow,
    mfuPanel,
    advisoryPanel,
    timeBar,
    dataPool.root,
    carpet.root,
    fleet.root,
    bubbleStrip,
    el("div", { class: "panel" }, [canvas]),
    legendPanel,
  );
  renderToolbar();
  applyRange(meta.time_range);
  await Promise.all([loadData(), loadAdvisories()]);
  await refreshOverview(); // topology decides carpet vs fleet panels
  renderAll();
  if (!manyLanes()) carpet.redraw(); // selection markers need the loaded lane list
  const onResize = () => {
    draw();
    manyLanes() ? fleet.redraw() : carpet.redraw();
    dataPool.redraw();
  };
  window.addEventListener("resize", onResize);

  // ------------------------- follow-mode auto refresh -----------------------
  let refreshing = false;
  let intervalId = null;
  if (meta.mode === "follow") {
    intervalId = setInterval(async () => {
      if (refreshing || changingWindow()) return;
      refreshing = true;
      try {
        // fresh global range (cheap edge-stamp read); getMeta() is cached
        const revision = windowRevision;
        const range = (await api("/api/meta")).time_range;
        if (revision !== windowRevision || changingWindow()) return;
        // A completed historical window does not need repeated multi-MB fetches.
        if (haveData && range && v1 < T1 && range[0] === T0 && range[1] === T1) return;
        applyRange(range);
        await Promise.all([loadData(), loadAdvisories()]);
        if (revision !== windowRevision || changingWindow()) return;
        renderAll();
        await refreshOverview();
      } catch {
        // transient fetch failure (server restart, network blip): keep polling
      } finally {
        refreshing = false;
      }
    }, FOLLOW_REDRAW_MS);
  }
  setViewCleanup(() => {
    if (intervalId !== null) clearInterval(intervalId);
    cancelAnimationFrame(panFrame);
    ++windowRevision;
    clearTimeout(windowTimer);
    carpet.destroy();
    window.removeEventListener("resize", onResize);
    window.removeEventListener("mouseup", onMouseUp);
  });
}
