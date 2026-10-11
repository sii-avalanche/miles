import { api } from "./api.js";
import { el, fmtNum } from "./app.js";
import { hideTooltip, showTooltip } from "./charts.js";

const BUCKETS = [
  ["lag_0_groups", "V", "#2a78d6"], ["lag_1_groups", "V−1", "#1baf7a"],
  ["lag_2_groups", "V−2", "#eda100"], ["lag_3_groups", "V−3", "#e87ba4"],
  ["older_groups", "Older", "#e34948"], ["future_groups", "Future", "#4a3aa7"],
  ["unknown_groups", "Unknown", "#9c9488"],
];
const PHASES = {
  collect_batch: ["Collect batch", "#eda100"], drain_batch: ["Collect batch", "#eda100"],
  wait_rollout: ["Trainer wait", "#a9a9a9"], train: ["Train", "#2a78d6"],
  critic_train: ["Train", "#2a78d6"], update_weights: ["Train → inference", "#e87ba4"],
  initial_weight_sync: ["Train → inference", "#e87ba4"],
};
const LEFT = 156, RIGHT = 14, TOP = 28, ROW = 28;

export async function loadAsyncFlow(meta, t0, t1) {
  const keys = (meta.metric_keys ?? []).filter(k => /^runtime\/rollout\/buffer\//.test(k) || k === "runtime/rollout/telemetry/events_dropped_total");
  const [eventData, metrics, engineData, phaseData] = await Promise.all([
    api("/api/runtime/events", { t0, t1, limit: 5000 }),
    keys.length ? api("/api/metrics", { keys: keys.join(","), x: "runtime/time_s", t0, t1 }) : {},
    meta.capabilities?.has_engine_series
      ? api("/api/timeline/engine_series", { metric: "sglang_num_running_reqs", t0, t1, max_points: 4000 }) : { series: [] },
    meta.capabilities?.has_timeline ? api("/api/timeline/phases", { t0, t1 }) : { phases: [] },
  ]);
  return { ...eventData, metrics, engines: engineData.series, phases: phaseData.phases };
}

export function finalEvents(events) {
  const latest = new Map();
  for (const event of events) {
    const key = `${event.role}/${event.rollout_id}/${event.name}/${event.t0}`;
    // A closing record wins even if file ingestion reordered it before its open twin.
    if (!latest.has(key) || event.t1 !== null) latest.set(key, event);
  }
  return [...latest.values()];
}

export function createAsyncFlow() {
  const canvas = el("canvas", { class: "timeline" });
  const note = el("p", { class: "muted" });
  const table = el("tbody");
  const poolSummary = el("tbody");
  const poolTable = el("table", {}, [
    el("thead", {}, [el("tr", {}, ["Pool / workload", "Sample time", "Reference V", "Queued", ...BUCKETS.map(([, label]) => label), "Mixed (subset)"].map(label => el("th", {}, [label])))]),
    poolSummary,
  ]);
  const root = el("div", { class: "panel" }, [
    el("h3", {}, ["Rollout, data pool, training and weight sync"]),
    el("p", { class: "muted" }, [
      "Shared time axis: green = engine requests running; gray = observed zero requests; gaps = no samples. ",
      "Yellow = collect this batch; blue = train; pink = train → inference weight sync. ",
      "Fully async keeps a background producer; full buffers, evaluation and weight updates can interrupt generation. ",
      "Pool counts are groups, classified by their oldest token weight version. Mixed-version groups are also reported in hover. ",
      "V is the last published inference weight version. Hover for wall time, version counts and rejection details.",
    ]), el("div", { class: "legend" }, BUCKETS.map(([, label, color]) =>
      el("span", { style: `color:${color};margin-right:12px` }, [`■ ${label}`]))), note, canvas,
    poolTable,
    el("details", {}, [el("summary", {}, ["Batch versions, weight publications and rejected data (latest 100 events)"]),
      el("table", {}, [el("thead", {}, [el("tr", {}, ["Time", "Event / batch", "Details"].map(x => el("th", {}, [x])))]), table])]),
  ]);

  function draw(data, meta, t0, t1, origin = meta.start_ts ?? t0) {
    const events = finalEvents(data.events ?? []);
    const phaseEvents = events.filter(e => e.name in PHASES);
    // Old runs with Timer telemetry still expose train / sync intervals, without invented batch IDs.
    for (const [name, fallback] of [["train", "actor_train"], ["update_weights", "update_weights"], ["collect_batch", "rollout"]]) {
      const hasRuntime = phaseEvents.some(e => name === "update_weights"
        ? ["update_weights", "initial_weight_sync"].includes(e.name)
        : name === "train" ? ["train", "critic_train"].includes(e.name) : ["collect_batch", "drain_batch"].includes(e.name));
      if (!hasRuntime) for (const p of data.phases ?? []) if (p.name === fallback) {
        phaseEvents.push({ ...p, name, status: "observed", rollout_id: null });
      }
    }
    const engines = data.engines.slice(0, 16);
    const prefixes = [...new Set(Object.keys(data.metrics).filter(k => k.endsWith("/versions/unknown_groups"))
      .map(k => k.slice(0, -"/versions/unknown_groups".length)))].sort();
    const rows = [
      ...engines.map((s, i) => ({ label: `Engine ${i + 1}`, engine: s })),
      ...(!engines.length ? [{ label: "Engine requests", missing: true }] : []),
      ...["Collect batch", "Trainer wait", "Train", "Train → inference"].map(label => ({ label })),
      ...prefixes.map(prefix => ({ label: `Pool${prefix.slice("runtime/rollout/buffer".length)}`, prefix })),
      ...(!prefixes.length ? [{ label: "Data pool", missing: true }] : []),
      { label: "Batch ready" }, { label: "Rejected data" },
    ];
    const gap = Math.max(1, 3 * (meta.engine_interval_s || 5));
    const poolGap = Math.max(30, 3 * (meta.runtime_interval_s || 10));
    canvas.style.height = `${TOP + rows.length * ROW + 8}px`;
    const rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
    ctx.font = "11px ui-monospace, monospace";
    const X = t => LEFT + (t - t0) / Math.max(t1 - t0, 1) * (rect.width - LEFT - RIGHT);
    const hits = [];
    const summaries = [];
    const box = (a, b, row, color, text, height = 16, offset = 0) => {
      if (b < t0 || a > t1) return;
      const x = X(Math.max(a, t0)), end = X(Math.min(b, t1));
      ctx.fillStyle = color;
      const y = TOP + row * ROW + offset;
      ctx.fillRect(x, y, Math.max(2, end - x), height);
      hits.push({ x, end: Math.max(x + 3, end), y, height, text });
    };
    for (let i = 0; i <= 6; i++) {
      const t = t0 + (t1 - t0) * i / 6;
      ctx.fillStyle = "#7a7168";
      ctx.fillText(`+${fmtNum(t - origin)}s`, X(t) - 18, 14);
    }
    rows.forEach((row, index) => {
      ctx.fillStyle = "#231f1c"; ctx.fillText(row.label, 6, TOP + index * ROW + 12);
      if (row.missing) { ctx.fillStyle = "#7a7168"; ctx.fillText("No telemetry available", LEFT, TOP + index * ROW + 12); }
      if (row.engine) {
        const s = row.engine;
        for (let i = 0; i < s.ts.length; i++) {
          if (!Number.isFinite(s.value[i])) continue;
          // Bound sample support; never paint across telemetry outages or into the future.
          const end = Math.min(s.ts[i + 1] ?? s.ts[i], s.ts[i] + gap, t1);
          box(s.ts[i], end, index, s.value[i] > 0 ? "#1baf7a" : "#ded6ca",
            `${s.addr}\n${s.value[i]} running requests\nSample: ${new Date(s.ts[i] * 1000).toLocaleString()}\nRequest activity may include evaluation; this is sampled, not token-level continuity.`);
        }
      }
      if (row.prefix) drawPool(row.prefix, index);
      for (const e of phaseEvents) {
        if (PHASES[e.name][0] !== row.label) continue;
        if (e.name === "drain_batch" && phaseEvents.some(p => p.name === "collect_batch" && p.rollout_id === e.rollout_id)) continue;
        box(e.t0, e.t1 ?? Math.min(t1, meta.time_range?.[1] ?? t1), index,
          e.status === "failed" ? "#e34948" : PHASES[e.name][1],
          `${row.label} · ${e.rollout_id === null ? (e.name === "initial_weight_sync" ? "initial sync" : "no batch ID") : `batch ${e.rollout_id}`}\n${new Date(e.t0 * 1000).toLocaleString()} → ${e.t1 === null ? "end not observed" : new Date(e.t1 * 1000).toLocaleString()}\n${fmtNum((e.t1 ?? t1) - e.t0)}s · ${e.status}\n${row.label === "Train → inference" ? "End-to-end sync orchestration; publication markers show the resulting version." : ""}`);
      }
      for (const e of events) {
        if ((row.label === "Batch ready" && e.name === "batch_ready") ||
            (row.label === "Rejected data" && (e.name === "buffer_reject" ||
              (e.name === "batch_ready" && e.details?.groups_removed_by_batch_filter > 0)))) {
          box(e.t0, e.t0, index, row.label === "Batch ready" ? "#eda100" : "#e34948", eventText(e));
        }
      }
    });
    // A publication marks confirmed completion, on the sync row, with its actual version.
    const syncRow = rows.findIndex(r => r.label === "Train → inference");
    for (const e of events) if (e.name === "weight_published") box(e.t0, e.t0, syncRow, "#4a3aa7", eventText(e), 24);

    function drawPool(prefix, row) {
      const queued = data.metrics[prefix + "/queued_groups"];
      if (!queued?.ts?.length) return;
      const lookup = new Map(Object.entries(data.metrics).filter(([key]) => key.startsWith(prefix + "/"))
        .map(([key, s]) => [key, new Map((s.ts ?? []).map((t, i) => [t, s.y[i]]))]));
      const latest = queued.ts.findLastIndex(t => t >= t0 && t <= t1);
      if (latest >= 0) {
        const t = queued.ts[latest];
        const value = key => lookup.get(prefix + "/" + key)?.get(t);
        summaries.push(el("tr", {}, [prefix.slice("runtime/rollout/buffer".length) || "Total",
          new Date(t * 1000).toLocaleTimeString(), fmtNum(value("current_weight_version")), fmtNum(queued.y[latest]),
          ...BUCKETS.map(([key]) => fmtNum(value("versions/" + key))), fmtNum(value("versions/mixed_groups")),
        ].map(text => el("td", {}, [text]))));
      }
      const max = Math.max(1, ...queued.y);
      for (let i = 0; i < queued.ts.length; i++) {
        const t = queued.ts[i], end = Math.min(queued.ts[i + 1] ?? t, t + poolGap, t1);
        const value = key => {
          return lookup.get(prefix + "/" + key)?.get(t) ?? null;
        };
        const version = value("current_weight_version");
        const counts = BUCKETS.map(([key, label, color]) => [value("versions/" + key), label, color]);
        const text = `Pool: ${new Date(t * 1000).toLocaleString()}\nV = ${version ?? "unknown"}; queued = ${queued.y[i]} groups\n` +
          counts.map(([n, label]) => `${label}: ${n ?? "missing"}`).join("\n") +
          `\nMixed-version groups (subset): ${value("versions/mixed_groups") ?? "missing"}\nShared capacity: ${value("capacity_groups") ?? "missing"}`;
        let offset = 20;
        for (const [n, , color] of counts) {
          if (n === null || n <= 0) continue;
          const h = n / max * 20; offset -= h;
          box(t, end, row, color, text, h, offset);
        }
        // Empty pools remain hoverable, so zero is distinguishable from missing telemetry.
        if (queued.y[i] === 0) box(t, end, row, "#ded6ca", text, 2, 20);
      }
    }
    canvas.onmousemove = ev => {
      const r = canvas.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
      const hit = hits.findLast(h => x >= h.x && x <= h.end && y >= h.y && y <= h.y + h.height);
      hit ? showTooltip(ev.clientX, ev.clientY, hit.text) : hideTooltip();
    };
    canvas.onmouseleave = hideTooltip;
    poolSummary.replaceChildren(...summaries);
    poolTable.style.display = summaries.length ? "" : "none";
    note.textContent = `${BUCKETS.map(([, label]) => label).join(" · ")} pool buckets. ` +
      (data.truncated ? "Event limit reached; narrow the time window to inspect omitted events. " : "") +
      ((data.metrics["runtime/rollout/telemetry/events_dropped_total"]?.y?.at(-1) ?? 0) > 0
        ? "The telemetry writer reported dropped events; decision history may be incomplete. " : "") +
      (data.engines.length > engines.length ? `Showing ${engines.length} of ${data.engines.length} engine series. ` : "") +
      (prefixes.length ? "Pool height scales to the maximum queued groups in this window." : "Version telemetry requires updated training code.");
    table.replaceChildren(...events.filter(e => ["buffer_reject", "batch_ready", "weight_published"].includes(e.name))
      .sort((a, b) => b.t0 - a.t0).slice(0, 100).map(e => el("tr", {}, [
        el("td", {}, [new Date(e.t0 * 1000).toLocaleString()]),
        el("td", {}, [`${e.name}${e.rollout_id !== null ? ` / ${e.rollout_id}` : ""}`]),
        el("td", { style: "white-space:pre-wrap;overflow-wrap:anywhere" }, [eventText(e)]),
      ])));
  }
  return { root, draw };
}

function eventText(e) {
  const d = e.details ?? {};
  if (e.name === "weight_published") return `${new Date(e.t0 * 1000).toLocaleString()} · weights published: ${d.previous_version ?? "initial"} → ${d.current_version} · policy ${d.trainer_model_id ?? "default"}`;
  if (e.name === "batch_ready") return `Batch ${e.rollout_id} · ${d.groups} groups · V=${d.current_version ?? "unknown"}\nVersions (oldest per group): ${JSON.stringify(d.versions)}\nWorkload / versions: ${JSON.stringify(d.workload_versions)}\nBatch filter removed: ${d.groups_removed_by_batch_filter ?? "not recorded"} groups\nMixed groups: ${d.mixed_groups} · policy ${d.trainer_model_id ?? "default"}`;
  return `${new Date(e.t0 * 1000).toLocaleString()} · ${d.reason} → ${d.action}\n${d.groups} groups · versions ${d.oldest_version ?? "unknown"}…${d.newest_version ?? "unknown"} · V=${d.current_version ?? "unknown"}\nQueued after rejection: ${d.queued_groups} · workload ${d.workload ?? "unknown"} · policy ${d.trainer_model_id ?? "default"}`;
}
