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
let apiCalls = null;
const mocks = {
  "./api.js": { api: async (path, params) => {
    if (!apiCalls) throw new Error("unexpected API call");
    apiCalls.push({ path, params }); return {};
  }, APP_BASE: new URL("https://host/proxy/runs/abc/") },
  "./app.js": { el: (tag, attrs, children) => new Element(tag, attrs, children), fmtNum: String },
  "./charts.js": { hideTooltip: () => { tooltip = null; }, showTooltip: (x, y, text) => { tooltip = text; } },
};
let opened;
const context = createContext({ Date, URL, window: { devicePixelRatio: 1, open: url => { opened = url; } } });
const mod = new SourceTextModule(readFileSync(new URL("../../../miles/dashboard/static/async_flow.js", import.meta.url), "utf8"), { context });
const stages = new SourceTextModule(readFileSync(new URL("../../../miles/dashboard/static/runtime_status.js", import.meta.url), "utf8"), { context });
const linker = name => name === "./runtime_status.js" ? stages : new SyntheticModule(Object.keys(mocks[name]), function () {
  for (const [key, value] of Object.entries(mocks[name])) this.setExport(key, value);
}, { context });
await stages.link(linker);
await stages.evaluate();
await mod.link(linker);
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
metrics["runtime/rollout/sample/reward/waiting"] = metric([4, 0, 3]);
metrics["runtime/rollout/sample/reward/oldest_wait_seconds"] = metric([60, 0, 100]);
flow.draw({
  history: { counts: { stale: 2, identified_stale: 2, dynamic_filter: 1, identified_dynamic: 0 },
    coverage_note: "Historical training partially recovered", rows: [{ ts: 62, timestamp_kind: "decision", event: "buffer_reject", reason: "stale", group_index: 7, query_id: "1677", generation_attempt: 1,
      prompt_preview: "Find the smallest", stale_count_at_event: 1, stale_count_total: 2, artifact: "prompts/group_7_attempt_1.html",
      training: { status: "training_completed", batch_id: 3, generation_attempt: 3, completed_at: 90 } },
    { ts: 63, timestamp_kind: "decision", event: "buffer_reject", reason: "dynamic_filter", training: { status: "unknown" } }] },
  metrics, phases: [], truncated: true,
  engines: [{ addr: "http://engine", ts: [10, 20, 80], value: [4, 0, 3] }],
  events: [closed, open, event("collect_batch", 5, 20, { role: "rollout" }), event("update_weights", 50, 60),
    event("weight_published", 60, 60, { role: "rollout", rollout_id: null, details: { previous_version: 4, current_version: 5 } }),
    event("batch_ready", 20, 20, { role: "rollout", details: { current_version: 4, groups: 2, versions: { 3: 1, 4: 1 }, workload_versions: { math: { 3: 1, 4: 1 } }, mixed_groups: 1 } }),
    event("buffer_reject", 62, 62, { role: "rollout", rollout_id: null, details: { reason: "stale", action: "retry", workload: "math", groups: 1, oldest_version: 1, newest_version: 1, current_version: 5, queued_groups: 2 } }),
  ],
}, { start_ts: 0, time_range: [0, 100], engine_interval_s: 5, runtime_interval_s: 10, reward_info: { kind: "lmaaj" } }, 0, 100);
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
assert.ok(labels.includes("LMaaJ scoring"));
const rewardBoxes = canvas.context.boxes.filter(b => b.y === rowY("LMaaJ scoring"));
assert.equal(rewardBoxes[0].color, "#8b5cc7");
assert.equal(rewardBoxes[1].color, "#ded6ca");
assert.equal(rewardBoxes[1].x + rewardBoxes[1].width, X(50), "reward telemetry gaps stay blank");
canvas.onmousemove({ clientX: X(15), clientY: rowY("LMaaJ scoring") + 5 });
assert.ok(tooltip.includes("4 samples") && tooltip.includes("60 s") && tooltip.includes("not HTTP concurrency"));
const segments = stages.namespace.stageSegments({ ts: [10, 20, 30, 100], y: [4, NaN, 0, 5] }, 0, 120, 30, 120);
assert.equal(JSON.stringify(segments.map(s => [s.start, s.end, s.value])), JSON.stringify([[10, 20, 4], [30, 60, 0]]));
assert.equal(stages.namespace.stageLabel(stages.namespace.ROLLOUT_STAGES[1]), "Reward scoring", "unknown judges must not be labelled LMaaJ");
canvas.onmousemove({ clientX: X(55), clientY: rowY("Train → inference") + 5 });
assert.ok(tooltip.includes("50") && tooltip.includes("10s"));
canvas.onmousemove({ clientX: X(60), clientY: rowY("Train → inference") + 5 });
assert.ok(tooltip.includes("4 → 5"));
canvas.onmousemove({ clientX: X(15), clientY: rowY("Pool/workload/math") + 12 });
assert.ok(tooltip.includes("V = 4") && tooltip.includes("V−1: 1") && tooltip.includes("subset): 1"));
canvas.onmousemove({ clientX: X(62), clientY: rowY("Rejected data") + 5 });
assert.ok(tooltip.includes("stale → retry") && tooltip.includes("workload math"));
assert.ok(tooltip.includes("prompt 1677") && tooltip.includes("2 stale events total"));
canvas.onclick({ clientX: X(62), clientY: rowY("Rejected data") + 5 });
assert.equal(opened, "https://host/proxy/runs/abc/analysis/prompts/group_7_attempt_1.html");
const text = node => typeof node === "string" ? node : node.textContent + node.children.map(text).join(" ");
assert.ok(text(flow.root).includes("Event limit reached"));
assert.ok(text(flow.root).includes('"math":{"3":1,"4":1}'));
assert.ok(text(flow.root).includes("2 total") && text(flow.root).includes("Training completed"));
const section = flow.root.children.find(e => e.attrs?.class === "group-history");
const controls = section.children.find(e => e.attrs?.class === "controls");
const select = controls.children[0], search = controls.children[1], identityFilter = controls.children[2];
identityFilter.value = "all";
select.value = "dynamic_filter"; select.attrs.onchange();
assert.ok(text(section).includes("Identity not recorded") && text(section).includes("Subsequent outcome not recorded"));
assert.ok(!text(section).includes("prompt 1677"));
select.value = "stale"; search.value = "no matching query"; search.attrs.oninput();
assert.ok(text(section).includes("No recorded events match the current filters"));

const batchEvent = (t, count, batch = 1, status = "collecting") => event("batch_pool", t, t, {
  role: "rollout", rollout_id: batch, details: { collected_groups: count, target_groups: 4, status },
});
const abortEvent = event("batch_abort", 19, 19, {
  role: "rollout", rollout_id: 1, details: { groups: 3, samples_upper_bound: 48, action: "discard" },
});
const batchEvents = [batchEvent(5, 0), batchEvent(10, 1), batchEvent(15, 4, 1, "ready"), abortEvent,
  batchEvent(20, 0, 1, "released"), batchEvent(80, 0, 2), batchEvent(85, 2, 2)];
const batchFlow = mod.namespace.createAsyncFlow();
batchFlow.draw({ metrics: {}, events: batchEvents, engines: [], history: { rows: [], counts: {} } },
  { start_ts: 0, time_range: [0, 100], runtime_interval_s: 10 }, 0, 100);
const batchCanvas = batchFlow.root.children.find(e => e.tag === "canvas");
const batchY = batchCanvas.context.labels.find(l => l.text === "Data pool (batch)").y - 12;
const batchBoxes = batchCanvas.context.boxes.filter(b => b.y >= batchY && b.y < batchY + 24);
assert.ok(batchCanvas.context.labels.some(l => l.text === "Abort 3"));
assert.ok(batchBoxes.some(b => b.color === "#2a78d6" && b.height === 5), "one collected group fills 1/4 of the target height");
assert.ok(batchBoxes.some(b => b.color === "#2a78d6" && b.height === 20), "a full batch reaches target height");
assert.ok(!batchBoxes.some(b => b.x < X(65) && b.x + b.width > X(65)), "pool telemetry gaps remain blank");
batchCanvas.onmousemove({ clientX: X(17), clientY: batchY + 12 });
assert.ok(tooltip.includes("Collected 4 / 4 groups") && tooltip.includes("passed filters"));
batchCanvas.onmousemove({ clientX: X(19), clientY: batchY + 12 });
assert.ok(tooltip.includes("3 unfinished groups") && tooltip.includes("48 sample slots") && tooltip.includes("do not equal actual cancelled"));
batchCanvas.onmousemove({ clientX: X(25), clientY: batchY + 21 });
assert.ok(tooltip.includes("Collected 0 / 4 groups") && tooltip.includes("released"));
assert.ok(text(batchFlow.root).includes("Batch collection"));
assert.ok(!text(batchFlow.root).includes("Version telemetry requires"));
const batchPrefix = "runtime/rollout/batch/";
const batchMetrics = Object.fromEntries([
  ["collected_groups", [0, 1, 2]], ["target_groups", [4, 4, 4]], ["rollout_id", [1, 1, 2]], ["active", [1, 1, 1]],
].map(([key, values]) => [batchPrefix + key, metric(values)]));
const snapshots = mod.namespace.batchPoolSnapshots(batchMetrics, [batchEvent(20, 4, 1, "ready")]);
assert.equal(snapshots.length, 3);
assert.equal(snapshots[1].count, 4, "an exact transition wins over a snapshot at the same timestamp");
assert.equal(snapshots[2].batch, 2);
assert.equal(mod.namespace.batchPoolSnapshots({}, [event("batch_pool", 10, 10, { details: { collected_groups: 1, target_groups: 4 } })]).length, 0, "other roles cannot create a rollout pool");
apiCalls = [];
await mod.namespace.loadAsyncFlow({ metric_keys: Object.keys(batchMetrics) }, 0, 100);
assert.ok(apiCalls.find(c => c.path === "/api/metrics").params.keys.includes(batchPrefix + "collected_groups"));
const legacyBatchFlow = mod.namespace.createAsyncFlow();
legacyBatchFlow.draw({ metrics: {}, events: [], engines: [], history: { rows: [], counts: {} } },
  { start_ts: 0, time_range: [0, 100], training_strategy: { kind: "one_step_off_policy" } }, 0, 100);
const legacyCanvas = legacyBatchFlow.root.children.find(e => e.tag === "canvas");
assert.ok(legacyCanvas.context.labels.some(l => l.text === "Collection sizes were not recorded"));
assert.ok(!legacyCanvas.context.boxes.length, "legacy runs must not receive invented pool sizes");
assert.ok(text(legacyBatchFlow.root).includes("updated training code"));

const selectedEvent = event("training_batch_selected", 21, 21, {
  role: "rollout", rollout_id: 1, details: { groups: [{ group_index: 1 }, { group_index: 2 }] },
});
const rejectEvent = event("batch_reject", 12, 12, {
  role: "rollout", rollout_id: 1, details: { groups: 1, reason: "zero_std_1.0", action: "drop" },
});
const decisionFlow = mod.namespace.createAsyncFlow();
const decisionMeta = { start_ts: 0, time_range: [0, 100], training_strategy: { kind: "one_step_off_policy" } };
decisionFlow.draw({ metrics: {}, events: [selectedEvent, rejectEvent, abortEvent], engines: [] }, decisionMeta, 0, 100);
const decisionCanvas = decisionFlow.root.children.find(e => e.tag === "canvas");
const decisionY = label => decisionCanvas.context.labels.find(l => l.text === label).y - 12;
assert.equal(decisionCanvas.context.boxes.filter(b => b.y === decisionY("Batch ready")).length, 1);
assert.equal(decisionCanvas.context.boxes.filter(b => b.y === decisionY("Rejected data")).length, 2);
decisionCanvas.onmousemove({ clientX: X(21), clientY: decisionY("Batch ready") + 5 });
assert.ok(tooltip.includes("training_batch_selected") && tooltip.includes("2 groups"));
decisionCanvas.onmousemove({ clientX: X(12), clientY: decisionY("Rejected data") + 5 });
assert.ok(tooltip.includes("zero_std_1.0") && tooltip.includes("drop"));
decisionCanvas.context.boxes.length = 0;
decisionFlow.draw({ metrics: {}, events: [selectedEvent, event("batch_ready", 20, 20, { rollout_id: 1, details: {} })], engines: [] }, decisionMeta, 0, 100);
assert.equal(decisionCanvas.context.boxes.filter(b => b.y === decisionY("Batch ready")).length, 1, "native batch_ready must not be duplicated by the selection fallback");
console.log("Async flow smoke passed: overlapping phases, weight publications, workload versions, rejections and telemetry gaps.");
