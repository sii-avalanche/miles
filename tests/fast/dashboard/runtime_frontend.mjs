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
  Date, console,
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
    metric_keys: [workload + "in_flight_group_fraction", ...(frame ? [workload + "collection_progress_fraction"] : [])],
  };
  if (path === "/api/runtime/events") return { events: [
    { role: "driver", rollout_id: 1, name: "wait_rollout", t0: now - 10, t1: now, status: "cancelled" },
  ] };
  assert.equal(path, "/api/metrics");
  return Object.fromEntries(params.keys.split(",").map(key => [key,
    key === inference || key.startsWith(workload)
      ? { x: [5000, 5010], y: [0.5, 0.6], ts: [now - 10, now] }
      : { x: [], y: [], ts: [] },
  ]));
};
const mocks = {
  "./api.js": { api },
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
mocks["./charts.js"].hideTooltip = () => {};
mocks["./charts.js"].showTooltip = () => {};
const linker = name => name === "./async_flow.js" ? flowModule : new SyntheticModule(Object.keys(mocks[name]), function () {
  for (const [key, value] of Object.entries(mocks[name])) this.setExport(key, value);
}, { context });
await flowModule.link(linker);
await flowModule.evaluate();
await viewModule.link(linker);
await viewModule.evaluate();

const root = new Element("main");
await viewModule.namespace.renderRuntime(root, { mode: "follow" });
const gpu = draws.find(draw => draw.series.some(series => series.label === "Inference"));
assert.equal(gpu.options.colorIndex("Inference"), 1, "a missing trainer series must not change inference's color");
assert.equal(gpu.canvas.context.paths.find(path => path.color === charts.namespace.SERIES_COLORS[1]).points[0][1], 52);
assert.equal(requests.find(request => request.path === "/api/metrics").params.t0, now - 14400);

frame = 1;
await refresh();
assert.ok(requests.filter(request => request.path === "/api/metrics").at(-1).params.keys.includes(workload + "collection_progress_fraction"));
const text = node => typeof node === "string" ? node : node.textContent + node.children.map(text).join(" ");
assert.ok(text(root).includes("wait_rollout: cancelled"));
assert.equal(text(root).split("collection_progress_fraction").length - 1, 1);
await refresh();
assert.equal(text(root).split("collection_progress_fraction").length - 1, 1, "refresh must not duplicate workload panels");

const gapCanvas = new Element("canvas");
charts.namespace.drawMultiLine(gapCanvas, [{ label: "gap", ts: [0, 10, 100], value: [1, 2, 3] }], {
  timeOrigin: 0, xDomain: [0, 100], maxGapSeconds: 30,
});
const curve = gapCanvas.context.paths.find(path => path.color === charts.namespace.SERIES_COLORS[0]);
assert.equal(curve.points.map(point => point[0]).join(""), "MLM", "stale intervals must not be drawn as observed activity");
cleanup();
console.log("Runtime frontend smoke passed: dynamic metrics, stable colors, viewport, cancellation and missing-data gaps.");
