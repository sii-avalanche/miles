// Run with: node --experimental-vm-modules tests/fast/dashboard/runtime_frontend.mjs
// Exercise the real runtime view and canvas renderer without browser dependencies.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createContext, SourceTextModule, SyntheticModule } from "node:vm";

class Element {
  constructor(tag, attrs = {}, children = []) {
    this.tag = tag;
    this.attrs = attrs;
    this.children = children;
    this.textContent = "";
    this.style = {};
    this.context = {
      paths: [], path: [],
      scale() {}, clearRect() {}, fillText() {}, fillRect() {}, save() {}, restore() {}, rect() {}, clip() {}, arc() {}, fill() {},
      beginPath() { this.path = []; },
      moveTo(x, y) { this.path.push(["M", x, y]); },
      lineTo(x, y) { this.path.push(["L", x, y]); },
      stroke() { this.paths.push({ color: this.strokeStyle, points: [...this.path] }); },
    };
  }
  append(...children) { this.children.push(...children); }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  replaceChildren(...children) { this.children = children; }
  getBoundingClientRect() { return { width: 420, height: 220, left: 0, top: 0 }; }
  getContext() { return this.context; }
}

const now = Date.now() / 1000;
const workload = "runtime/rollout/workload/all/";
const inference = "runtime/driver/fleet/inference/gpu_util_mean_pct";
let refresh, cleanup, frame = 0;
const draws = [], requests = [];
const context = createContext({
  Date, URL, console, setTimeout, clearTimeout,
  window: { devicePixelRatio: 1, addEventListener() {}, removeEventListener() {} },
  document: { documentElement: {} },
  getComputedStyle: () => ({ getPropertyValue: () => "#888" }),
  setInterval: fn => { refresh = fn; return 1; },
  clearInterval() {},
});
const chartSource = readFileSync(new URL("../../../miles/dashboard/static/charts.js", import.meta.url), "utf8");
const charts = new SourceTextModule(chartSource, { context });
await charts.link(() => { throw new Error("unexpected chart import"); });
await charts.evaluate();

const api = async (path, params = {}) => {
  requests.push({ path, params });
  if (path === "/api/meta") return {
    mode: "follow", runtime_interval_s: 10, time_range: [now - 20000, now],
    metric_keys: ["runtime/rollout/batch/collected_groups", "runtime/rollout/batch/target_groups"],
  };
  if (path === "/api/runtime/events") return { events: [
    { role: "driver", rollout_id: 1, name: "wait_rollout", t0: now - 10, t1: now, status: "cancelled" },
  ] };
  if (path === "/api/runtime/group_history") return { rows: [], counts: {} };
  if (path === "/api/runtime/status") return {
    training_strategy: { kind: frame < 2 ? "one_step_off_policy" : "fully_async", settings: { num_steps_per_rollout: 1, use_tis: true } },
    reward_info: { kind: "lmaaj", model_name: "Kimi-K3", timeout: 600, max_retries: 3 },
    rollout: { observed_at: now - 1, age_seconds: 1, fresh: frame < 2, stages: [
      { id: "generate", active: 0, oldest_seconds: 0, finished_total: 1024 },
      { id: "reward", active: 1, oldest_seconds: 3500, finished_total: 896 },
      { id: "database", active: 0, pending: 0, oldest_seconds: 0, finished_total: 893 },
    ] }, driver: { fresh: frame < 2, active_phases: [{ name: "wait_rollout", oldest_seconds: 4300 }] },
  };
  assert.equal(path, "/api/metrics");
  return Object.fromEntries(params.keys.split(",").map(key => [key,
    key === inference || key.startsWith(workload)
      ? { x: [5000, 5010], y: [0.5, 0.6], ts: [now - 10, now] }
      : { x: [], y: [], ts: [] },
  ]));
};
const mocks = {
  "./api.js": { api, APP_BASE: new URL("https://host/runs/abc/") },
  "./app.js": {
    el: (tag, attrs, children) => new Element(tag, attrs, children),
    fmtNum: value => String(value),
    setViewCleanup: fn => { cleanup = fn; },
  },
  "./charts.js": {
    SERIES_COLORS: charts.namespace.SERIES_COLORS,
    drawChart: charts.namespace.drawChart,
    drawMultiLine: (canvas, series, options) => {
      draws.push({ canvas, series, options });
      charts.namespace.drawMultiLine(canvas, series, options);
    },
  },
};
const source = readFileSync(new URL("../../../miles/dashboard/static/views_runtime.js", import.meta.url), "utf8");
const viewModule = new SourceTextModule(source, { context });
const flowModule = new SourceTextModule(readFileSync(new URL("../../../miles/dashboard/static/async_flow.js", import.meta.url), "utf8"), { context });
const timeModule = new SourceTextModule(readFileSync(new URL("../../../miles/dashboard/static/time_window.js", import.meta.url), "utf8"), { context });
const statusModule = new SourceTextModule(readFileSync(new URL("../../../miles/dashboard/static/runtime_status.js", import.meta.url), "utf8"), { context });
mocks["./charts.js"].hideTooltip = () => {};
mocks["./charts.js"].showTooltip = () => {};
mocks["./fleet.js"] = { createFleet: () => { throw new Error("No GPU telemetry advertised"); } };
mocks["./phase_colors.js"] = { PHASE_COLORS: {} };
const linker = name => name === "./runtime_status.js" ? statusModule : name === "./async_flow.js" ? flowModule : name === "./time_window.js" ? timeModule : new SyntheticModule(Object.keys(mocks[name]), function () {
  for (const [key, value] of Object.entries(mocks[name])) this.setExport(key, value);
}, { context });
await statusModule.link(linker);
await statusModule.evaluate();
await flowModule.link(linker);
await flowModule.evaluate();
await timeModule.link(linker);
await timeModule.evaluate();
await viewModule.link(linker);
await viewModule.evaluate();

const root = new Element("main");
await viewModule.namespace.renderRuntime(root, { mode: "follow" });
const text = node => typeof node === "string" ? node : node.textContent + node.children.map(text).join(" ");
assert.equal(requests.find(request => request.path === "/api/metrics").params.t0, now - 14400);
assert.ok(text(root).includes("One-step off-policy") && text(root).includes("TIS=on"));
assert.ok(text(root).includes("Waiting for rollout") && text(root).includes("Kimi-K3"));
assert.ok(!/[\p{Script=Han}]/u.test(text(root)), "Dashboard interface must be English");
assert.ok(!text(root).includes("Database backlog") && !text(root).includes("GPU utilization (%)"));
assert.equal(draws.length, 0, "generic Grafana charts must not render in RL Compute");
assert.ok(requests.filter(r => r.path === "/api/metrics").every(r => !r.params.keys.includes("fleet/")), "do not fetch removed charts");
frame = 1;
await refresh();
assert.ok(text(root).includes("One-step off-policy"));
frame = 2;
await refresh();
assert.ok(text(root).includes("Driver sample is stale; current phase unknown"));
assert.ok(!text(root).includes("Waiting for rollout"), "stale driver samples must not claim an active phase");
assert.ok(text(root).includes("Fully async") && text(root).includes("Generate continuously"));

const gapCanvas = new Element("canvas");
charts.namespace.drawMultiLine(gapCanvas, [{ label: "gap", ts: [0, 10, 100], value: [1, 2, 3] }], {
  timeOrigin: 0, xDomain: [0, 100], maxGapSeconds: 30,
});
const curve = gapCanvas.context.paths.find(path => path.color === charts.namespace.SERIES_COLORS[0]);
assert.equal(curve.points.map(point => point[0]).join(""), "MLM", "stale intervals must not be drawn as observed activity");
cleanup();
await viewModule.namespace.renderRuntime(root, { mode: "follow" }, { view: "data" });
assert.ok(text(root).includes("Group history: rejections, retries and training"));
assert.ok(!text(root).includes("GPU utilization & phase composition"));
cleanup();
console.log("Runtime frontend smoke passed: English UI, focused metrics, viewport, strategy and stale phases.");
