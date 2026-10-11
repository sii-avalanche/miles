// Run with: node --experimental-vm-modules tests/fast/dashboard/async_flow_frontend.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createContext, SourceTextModule, SyntheticModule } from "node:vm";

class Element {
  constructor(tag, attrs = {}, children = []) {
    this.tag = tag; this.attrs = attrs; this.children = children; this.style = {}; this.textContent = "";
    this.context = {
      boxes: [], labels: [], scale() {},
      fillRect(x, y, width, height) { this.boxes.push({ x, y, width, height, color: this.fillStyle }); },
      fillText(text, x, y) { this.labels.push({ text, x, y }); },
    };
  }
  replaceChildren(...children) { this.children = children; }
  getBoundingClientRect() { return { width: 1000, height: 450, left: 0, top: 0 }; }
  getContext() { return this.context; }
}
let tooltip;
const mocks = {
  "./api.js": { api: async () => { throw new Error("unexpected API call"); } },
  "./app.js": { el: (tag, attrs, children) => new Element(tag, attrs, children), fmtNum: String },
  "./charts.js": { hideTooltip: () => { tooltip = null; }, showTooltip: (x, y, text) => { tooltip = text; } },
};
const context = createContext({ Date, window: { devicePixelRatio: 1 } });
const mod = new SourceTextModule(readFileSync(new URL("../../../miles/dashboard/static/async_flow.js", import.meta.url), "utf8"), { context });
await mod.link(name => new SyntheticModule(Object.keys(mocks[name]), function () {
  for (const [key, value] of Object.entries(mocks[name])) this.setExport(key, value);
}, { context }));
await mod.evaluate();
const event = (name, t0, t1, extra = {}) => ({ role: "driver", rollout_id: 2, name, t0, t1, status: t1 === null ? "running" : "completed", ...extra });
const closed = event("train", 10, 50), open = event("train", 10, null);
assert.equal(mod.namespace.finalEvents([closed, open])[0].t1, 50, "closed twins win independent of ingestion order");
const prefix = "runtime/rollout/buffer/workload/math";
const metric = values => ({ ts: [10, 20, 80], x: [10, 20, 80], y: values });
const metrics = Object.fromEntries([
  ["queued_groups", [2, 3, 0]], ["capacity_groups", [10, 10, 10]], ["current_weight_version", [4, 5, 5]],
  ["versions/lag_0_groups", [1, 0, 0]], ["versions/lag_1_groups", [1, 3, 0]],
  ["versions/unknown_groups", [0, 0, 0]], ["versions/mixed_groups", [1, 0, 0]],
].map(([key, values]) => [prefix + "/" + key, metric(values)]));
const flow = mod.namespace.createAsyncFlow();
flow.draw({
  metrics, phases: [], truncated: true,
  engines: [{ addr: "http://engine", ts: [10, 20, 80], value: [4, 0, 3] }],
  events: [closed, open, event("collect_batch", 5, 20, { role: "rollout" }), event("update_weights", 50, 60),
    event("weight_published", 60, 60, { role: "rollout", rollout_id: null, details: { previous_version: 4, current_version: 5 } }),
    event("batch_ready", 20, 20, { role: "rollout", details: { current_version: 4, groups: 2, versions: { 3: 1, 4: 1 }, workload_versions: { math: { 3: 1, 4: 1 } }, mixed_groups: 1 } }),
    event("buffer_reject", 62, 62, { role: "rollout", rollout_id: null, details: { reason: "stale", action: "retry", workload: "math", groups: 1, oldest_version: 1, newest_version: 1, current_version: 5, queued_groups: 2 } }),
  ],
}, { start_ts: 0, time_range: [0, 100], engine_interval_s: 5, runtime_interval_s: 10 }, 0, 100);
const canvas = flow.root.children.find(e => e.tag === "canvas");
const X = t => 156 + t / 100 * 830;
const engineBoxes = canvas.context.boxes.filter(b => b.y === 28);
assert.equal(engineBoxes[0].x, X(10));
assert.equal(engineBoxes[0].width, X(20) - X(10));
assert.equal(engineBoxes[1].x + engineBoxes[1].width, X(35), "a telemetry gap must not appear continuously idle or busy");
assert.ok(!engineBoxes.some(b => b.x < X(50) && b.x + b.width > X(50)));
const labels = canvas.context.labels.map(l => l.text);
assert.ok(labels.includes("Pool/workload/math"));
assert.ok(labels.includes("Train → inference"));
const rowY = label => canvas.context.labels.find(l => l.text === label).y - 12;
canvas.onmousemove({ clientX: X(55), clientY: rowY("Train → inference") + 5 });
assert.ok(tooltip.includes("50") && tooltip.includes("10s"));
canvas.onmousemove({ clientX: X(60), clientY: rowY("Train → inference") + 5 });
assert.ok(tooltip.includes("4 → 5"));
canvas.onmousemove({ clientX: X(15), clientY: rowY("Pool/workload/math") + 12 });
assert.ok(tooltip.includes("V = 4") && tooltip.includes("V−1: 1") && tooltip.includes("subset): 1"));
canvas.onmousemove({ clientX: X(62), clientY: rowY("Rejected data") + 5 });
assert.ok(tooltip.includes("stale → retry") && tooltip.includes("workload math"));
const text = node => typeof node === "string" ? node : node.textContent + node.children.map(text).join(" ");
assert.ok(text(flow.root).includes("Event limit reached"));
assert.ok(text(flow.root).includes('"math":{"3":1,"4":1}'));
console.log("Async flow smoke passed: overlapping phases, weight publications, workload versions, rejections and telemetry gaps.");
