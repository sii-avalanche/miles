import { api, APP_BASE } from "./api.js";
import { el, fmtNum } from "./app.js";
import { hideTooltip, showTooltip } from "./charts.js";
import { ROLLOUT_STAGES, STAGE_METRICS, stageLabel, stageSegments } from "./runtime_status.js";

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
const BATCH = "runtime/rollout/batch/";

export async function loadAsyncFlow(meta, t0, t1) {
  const keys = (meta.metric_keys ?? []).filter(k => /^runtime\/rollout\/(buffer|batch)\//.test(k) || STAGE_METRICS.includes(k) || k === "runtime/rollout/telemetry/events_dropped_total");
  const [eventData, metrics, engineData, phaseData, history] = await Promise.all([
    api("/api/runtime/events", { t0, t1, limit: 5000 }),
    keys.length ? api("/api/metrics", { keys: keys.join(","), x: "runtime/time_s", t0, t1 }) : {},
    meta.capabilities?.has_engine_series
      ? api("/api/timeline/engine_series", { metric: "sglang_num_running_reqs", t0, t1, max_points: 4000 }) : { series: [] },
    meta.capabilities?.has_timeline ? api("/api/timeline/phases", { t0, t1 }) : { phases: [] },
    api("/api/runtime/group_history", { t0, t1, limit: 5000 }),
  ]);
  return { ...eventData, metrics, engines: engineData.series, phases: phaseData.phases, history };
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

export function batchPoolSnapshots(metrics, events) {
  const snapshots = new Map();
  const lookup = key => new Map((metrics[BATCH + key]?.ts ?? []).map((t, i) => [t, metrics[BATCH + key].y[i]]));
  const targets = lookup("target_groups"), batches = lookup("rollout_id"), active = lookup("active");
  const collected = metrics[BATCH + "collected_groups"];
  for (let i = 0; i < (collected?.ts?.length ?? 0); i++) {
    const t = collected.ts[i], count = collected.y[i], target = targets.get(t);
    if (Number.isFinite(t) && Number.isFinite(count) && count >= 0 && Number.isFinite(target) && target > 0) {
      snapshots.set(t, { ts: t, count, target, batch: batches.get(t), status: active.get(t) === 0 ? "released" : "collecting" });
    }
  }
  for (const e of events) if (e.role === "rollout" && e.name === "batch_pool") {
    const d = e.details ?? {};
    if (Number.isFinite(e.t0) && Number.isFinite(d.collected_groups) && d.collected_groups >= 0 && Number.isFinite(d.target_groups) && d.target_groups > 0) {
      snapshots.set(e.t0, { ts: e.t0, count: d.collected_groups, target: d.target_groups, batch: e.rollout_id, status: d.status });
    }
  }
  return [...snapshots.values()].sort((a, b) => a.ts - b.ts);
}

export function createAsyncFlow() {
  const canvas = el("canvas", { class: "timeline" });
  const note = el("p", { class: "muted" });
  const table = el("tbody");
  const poolSummary = el("tbody");
  const groupRows = el("tbody");
  const coverage = el("p", { class: "muted" });
  const search = el("input", { type: "text", placeholder: "Search prompt / query ID / group", oninput: drawGroups });
  const reason = el("select", { onchange: drawGroups }, [
    ["stale", "Stale retries"], ["dynamic_filter", "Dynamic filter"], ["selected", "Selected for training"], ["", "All events"],
  ].map(([value, label]) => el("option", { value }, [label])));
  const identityFilter = el("select", { onchange: drawGroups }, [
    el("option", { value: "known" }, ["Identified prompts"]), el("option", { value: "all" }, ["All (including unknown identities)"]),
  ]);
  let history = { rows: [], counts: {} }, selectedTs = null;
  const groupSection = el("div", { class: "group-history" }, [
    el("h3", {}, ["Group history: rejections, retries and training"]), coverage,
    el("div", { class: "controls" }, [reason, search, identityFilter]),
    el("div", { class: "tablewrap" }, [el("table", { class: "data" }, [
      el("thead", {}, [el("tr", {}, ["Time / evidence", "Action", "Prompt / group / attempt", "Stale count", "Version", "Subsequent training", "Details"].map(label => el("th", {}, [label])))]), groupRows,
    ])]),
  ]);
  const artifactUrl = path => new URL(`analysis/${path}`, APP_BASE).href;
  function drawGroups() {
    const query = (search.value || "").toLowerCase();
    const filter = reason.value ?? "stale";
    const rows = history.rows.filter(row => (!filter || (filter === "selected" ? row.event === "training_batch_selected" : row.reason === filter)) &&
      (identityFilter.value === "all" || row.group_index != null) &&
      `${row.query_id ?? ""} ${row.group_index ?? ""} ${row.prompt_preview ?? ""}`.toLowerCase().includes(query));
    const counts = history.counts || {};
    coverage.textContent = `Full log: ${counts.stale ?? 0} stale events (${counts.identified_stale ?? 0} identified); ${counts.dynamic_filter ?? 0} dynamic filter events (${counts.identified_dynamic ?? 0} identified groups). ` +
      `${counts.selected_groups ?? 0} groups confirmed selected. ${rows.length} matching events in this window; showing the latest ${Math.min(200, rows.length)}. Stale counts are per group across the full log. ` +
      (history.truncated ? "Record limit reached; narrow the time window. " : "") +
      ` ${history.coverage_note || "Older logs may lack identities."} Missing subsequent records do not prove that training did not occur.`;
    groupRows.replaceChildren(...rows.slice(0, 200).map(row => {
      const training = row.training || { status: "unknown" };
      const status = { unknown: "Subsequent outcome not recorded", selected: "Selected into batch", training_completed: "Training completed", training_failed: "Training failed", training_running: "Training started; completion not recorded" }[training.status];
      const trainingText = status + (training.batch_id != null ? ` · batch ${training.batch_id} · attempt ${training.generation_attempt ?? "?"}` : "") +
        (training.completed_at ? ` · ${new Date(training.completed_at * 1000).toLocaleString("en-GB")}` : "");
      const identity = row.group_index == null ? "Identity not recorded" : `prompt ${row.query_id ?? "?"} · group ${row.group_index} · attempt ${row.generation_attempt ?? "?"}`;
      const versions = row.current_version == null ? "Not recorded" : `${row.oldest_version ?? "?"}…${row.newest_version ?? "?"} / V=${row.current_version}`;
      return el("tr", { style: row.ts === selectedTs ? "background:#fff3db" : "" }, [
        el("td", {}, [new Date(row.ts * 1000).toLocaleString("en-GB"), row.timestamp_kind === "group_completed" ? " (generation completed; filter time not recorded)" : ""]),
        el("td", {}, [row.reason === "stale" ? (row.action === "drop" ? "Stale → drop" : "Stale → retry") : row.reason === "dynamic_filter" ? "Dynamic filter → drop" : row.event === "training_batch_selected" ? "Selected for training" : `${row.reason} → ${row.action}`]),
        el("td", { title: row.prompt_preview || "", style: "white-space:normal;text-align:left;min-width:230px" }, [identity, el("div", { class: "muted" }, [row.prompt_preview || ""])]),
        el("td", {}, [row.stale_count_total == null ? "Unknown" : `${row.stale_count_at_event} → ${row.stale_count_total} total`]),
        el("td", {}, [versions]), el("td", { style: "white-space:normal;min-width:170px" }, [trainingText]),
        el("td", {}, [row.artifact ? el("a", { href: artifactUrl(row.artifact), target: "_blank", rel: "noopener" }, ["Open HTML / Trace"]) : "Details unavailable"]),
      ]);
    }));
    if (!rows.length) groupRows.replaceChildren(el("tr", {}, [el("td", { colspan: 7 }, ["No recorded events match the current filters."])]));
  }
  const poolTable = el("table", {}, [
    el("thead", {}, [el("tr", {}, ["Pool / workload", "Sample time", "Reference V", "Queued", ...BUCKETS.map(([, label]) => label), "Mixed (subset)"].map(label => el("th", {}, [label])))]),
    poolSummary,
  ]);
  const root = el("div", { class: "panel" }, [
    el("h3", {}, ["Rollout, data pool, training and weight sync"]),
    el("p", { class: "muted" }, [
      "Shared time axis: green = engine requests running; gray = observed zero requests; gaps = no samples. ",
      "Yellow = collect this batch; blue = train; pink = train → inference weight sync. ",
      "Stage rows show generating, reward/LMaaJ (including queue and retries), Trace persistence and database backlog. Colored = active; gray = observed zero; gaps = missing samples. ",
      "Fully async keeps a background producer; full buffers, evaluation and weight updates can interrupt generation. ",
      "Pool counts are groups, classified by their oldest token weight version. Mixed-version groups are also reported in hover. ",
      "One-step pools show valid groups collected for this batch, scaled to its target. Red Abort markers show when cancellation was requested and how many unfinished groups it targeted. ",
      "V is the last published inference weight version. Hover for wall time, version counts and rejection details.",
    ]), el("div", { class: "legend" }, BUCKETS.map(([, label, color]) =>
      el("span", { style: `color:${color};margin-right:12px` }, [`■ ${label}`]))), note, canvas,
    poolTable, groupSection,
    el("details", {}, [el("summary", {}, ["Batch versions, weight publications and rejected data (latest 100 events)"]),
      el("table", {}, [el("thead", {}, [el("tr", {}, ["Time", "Event / batch", "Details"].map(x => el("th", {}, [x])))]), table])]),
  ]);

  function draw(data, meta, t0, t1, origin = meta.start_ts ?? t0) {
    canvas.setAttribute?.("data-time-start", t0);
    canvas.setAttribute?.("data-time-end", t1);
    history = data.history ?? { rows: [], counts: {} };
    drawGroups();
    const decisions = new Map(history.rows.filter(row => row.timestamp_kind === "decision" && row.event === "buffer_reject").map(row => [row.ts, row]));
    const events = finalEvents(data.events ?? []);
    const readyBatches = new Set(events.filter(e => e.name === "batch_ready").map(e => e.rollout_id));
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
    const batchSnapshots = batchPoolSnapshots(data.metrics, events);
    const hasBatchRecords = BATCH + "collected_groups" in data.metrics || events.some(e => e.role === "rollout" && ["batch_pool", "batch_abort"].includes(e.name));
    const hasBatchPool = hasBatchRecords || ["one_step_off_policy", "synchronous"].includes(meta.training_strategy?.kind);
    const rows = [
      ...(engines.length ? engines.map((s, i) => ({ label: `Engine ${i + 1}`, engine: s }))
        : [{ label: "Engine requests", missing: true }]),
      ...ROLLOUT_STAGES.map(stage => ({ label: stageLabel(stage, meta.reward_info), stage })),
      ...["Collect batch", "Trainer wait", "Train", "Train → inference"].map(label => ({ label })),
      ...prefixes.map(prefix => ({ label: `Pool${prefix.slice("runtime/rollout/buffer".length)}`, prefix })),
      ...(hasBatchPool ? [{ label: "Data pool (batch)", batchPool: true }] : []),
      ...(!prefixes.length && !hasBatchPool ? [{ label: "Data pool", missing: true }] : []),
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
    const box = (a, b, row, color, text, height = 16, offset = 0, decision = null) => {
      if (b < t0 || a > t1) return;
      const x = X(Math.max(a, t0)), end = X(Math.min(b, t1));
      ctx.fillStyle = color;
      const y = TOP + row * ROW + offset;
      ctx.fillRect(x, y, Math.max(2, end - x), height);
      hits.push({ x, end: Math.max(x + 3, end), y, height, text, decision });
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
            `${s.addr}\n${s.value[i]} running requests\nSample: ${new Date(s.ts[i] * 1000).toLocaleString("en-GB")}\nRequest activity may include evaluation; this is sampled, not token-level continuity.`);
        }
      }
      if (row.prefix) drawPool(row.prefix, index);
      if (row.batchPool) {
        drawBatchPool(index);
        for (const e of events) if (e.role === "rollout" && e.name === "batch_abort") {
          box(e.t0, e.t0, index, "#e34948", eventText(e), 24);
          if (e.t0 >= t0 && e.t0 <= t1) {
            ctx.fillStyle = "#b83c35";
            ctx.fillText(`Abort ${e.details?.groups ?? "?"}`, Math.min(X(e.t0) + 4, rect.width - 72), TOP + index * ROW + 10);
          }
        }
      }
      if (row.stage) {
        const stage = row.stage;
        const key = stage.prefix + (stage.id === "database" ? "/pending_writes" : "/waiting");
        const ageKey = stage.prefix + (stage.id === "database" ? "/oldest_pending_seconds" : "/oldest_wait_seconds");
        const ages = data.metrics[ageKey];
        const ageLookup = new Map((ages?.ts ?? []).map((t, i) => [t, ages.y[i]]));
        const segments = stageSegments(data.metrics[key], t0, t1, poolGap);
        if (!segments.length) { ctx.fillStyle = "#7a7168"; ctx.fillText("No telemetry in window", LEFT, TOP + index * ROW + 12); }
        for (const s of segments) box(s.start, s.end, index, s.value > 0 ? stage.color : "#ded6ca",
          `${row.label}\n${s.value} ${stage.id === "database" ? "pending writes (including active)" : "samples in stage"}\nLongest wait: ${fmtNum(ageLookup.get(s.ts))} s\nSample: ${new Date(s.ts * 1000).toLocaleString("en-GB")}\n${stage.id === "reward" ? "Includes limiter queue, judge request and retries; not HTTP concurrency." : "Sampled stage activity."}`);
      }
      for (const e of phaseEvents) {
        if (PHASES[e.name][0] !== row.label) continue;
        if (e.name === "drain_batch" && phaseEvents.some(p => p.name === "collect_batch" && p.rollout_id === e.rollout_id)) continue;
        box(e.t0, e.t1 ?? Math.min(t1, meta.time_range?.[1] ?? t1), index,
          e.status === "failed" ? "#e34948" : PHASES[e.name][1],
          `${row.label} · ${e.rollout_id === null ? (e.name === "initial_weight_sync" ? "initial sync" : "no batch ID") : `batch ${e.rollout_id}`}\n${new Date(e.t0 * 1000).toLocaleString("en-GB")} → ${e.t1 === null ? "end not observed" : new Date(e.t1 * 1000).toLocaleString("en-GB")}\n${fmtNum((e.t1 ?? t1) - e.t0)}s · ${e.status}\n${row.label === "Train → inference" ? "End-to-end sync orchestration; publication markers show the resulting version." : ""}`);
      }
      for (const e of events) {
        if ((row.label === "Batch ready" && (e.name === "batch_ready" ||
              (e.name === "training_batch_selected" && !readyBatches.has(e.rollout_id)))) ||
            (row.label === "Rejected data" && (["buffer_reject", "batch_reject", "batch_abort"].includes(e.name) ||
              (e.name === "batch_ready" && e.details?.groups_removed_by_batch_filter > 0)))) {
          const decision = decisions.get(e.t0);
          const identityText = decision?.group_index != null ? `\nprompt ${decision.query_id} · group ${decision.group_index} · attempt ${decision.generation_attempt} · ${decision.stale_count_total} stale events total\nClick to open HTML details` : "";
          box(e.t0, e.t0, index, row.label === "Batch ready" ? "#eda100" : "#e34948", eventText(e) + identityText, 16, 0, decision);
        }
      }
    });
    // A publication marks confirmed completion, on the sync row, with its actual version.
    const syncRow = rows.findIndex(r => r.label === "Train → inference");
    for (const e of events) if (e.name === "weight_published") box(e.t0, e.t0, syncRow, "#4a3aa7", eventText(e), 24);

    function drawBatchPool(row) {
      let visible = false;
      for (let i = 0; i < batchSnapshots.length; i++) {
        const s = batchSnapshots[i];
        const end = Math.min(batchSnapshots[i + 1]?.ts ?? s.ts, s.ts + poolGap, t1);
        if (s.ts > t1 || end < t0) continue;
        visible = true;
        const text = `Batch collection · batch ${s.batch ?? "unknown"}\nCollected ${s.count} / ${s.target} groups (passed filters)\n${new Date(s.ts * 1000).toLocaleString("en-GB")} · ${s.status ?? "observed"}\nThe full batch goes to training; collection restarts at zero for the next batch.`;
        box(s.ts, end, row, "#ded6ca", text, 1);
        const height = Math.min(1, s.count / s.target) * 20;
        box(s.ts, end, row, s.count > 0 ? "#2a78d6" : "#ded6ca", text, Math.max(2, height), 20 - height);
      }
      if (!visible) {
        ctx.fillStyle = "#7a7168";
        ctx.fillText(hasBatchRecords ? "No collection telemetry in window" : "Collection sizes were not recorded", LEFT, TOP + row * ROW + 12);
      }
    }

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
        const text = `Pool: ${new Date(t * 1000).toLocaleString("en-GB")}\nV = ${version ?? "unknown"}; queued = ${queued.y[i]} groups\n` +
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
    canvas.onclick = ev => {
      const r = canvas.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
      const hit = hits.findLast(h => x >= h.x && x <= h.end && y >= h.y && y <= h.y + h.height);
      if (!hit?.decision) return;
      if (hit.decision.artifact) window.open(artifactUrl(hit.decision.artifact), "_blank", "noopener");
      else {
        selectedTs = hit.decision.ts; reason.value = hit.decision.reason; search.value = "";
        drawGroups(); groupSection.scrollIntoView?.({ behavior: "smooth", block: "start" });
      }
    };
    poolSummary.replaceChildren(...summaries);
    poolTable.style.display = summaries.length ? "" : "none";
    note.textContent = (hasBatchPool ? "Batch collection: blue height = collected / target. Red Abort markers show unfinished groups requested for cancellation; hover for time and count. " : `${BUCKETS.map(([, label]) => label).join(" · ")} pool buckets. `) +
      (hasBatchPool ? "Batch ready uses the recorded batch selection time when needed. Rejected data includes filter rejections and cancellation requests; missing historical group rejections cannot be reconstructed. " : "") +
      (hasBatchPool && !hasBatchRecords ? "Collection counts were not recorded. A run launched with updated training code is required for the collection curve. " : "") +
      (data.truncated ? "Event limit reached; narrow the time window to inspect omitted events. " : "") +
      ((data.metrics["runtime/rollout/telemetry/events_dropped_total"]?.y?.at(-1) ?? 0) > 0
        ? "The telemetry writer reported dropped events; decision history may be incomplete. " : "") +
      (data.engines.length > engines.length ? `Showing ${engines.length} of ${data.engines.length} engine series. ` : "") +
      (prefixes.length ? "Pool height scales to the maximum queued groups in this window." : hasBatchPool ? "" : "Pool sizes were not recorded by this run; new telemetry requires updated training code.");
    table.replaceChildren(...events.filter(e => ["buffer_reject", "batch_reject", "batch_ready", "weight_published", "batch_abort"].includes(e.name) ||
      (e.name === "training_batch_selected" && !readyBatches.has(e.rollout_id)))
      .sort((a, b) => b.t0 - a.t0).slice(0, 100).map(e => el("tr", {}, [
        el("td", {}, [new Date(e.t0 * 1000).toLocaleString("en-GB")]),
        el("td", {}, [`${e.name}${e.rollout_id !== null ? ` / ${e.rollout_id}` : ""}`]),
        el("td", { style: "white-space:pre-wrap;overflow-wrap:anywhere" }, [eventText(e)]),
      ])));
  }
  return { root, draw };
}

function eventText(e) {
  const d = e.details ?? {};
  if (e.name === "training_batch_selected") return `${new Date(e.t0 * 1000).toLocaleString("en-GB")} · Batch ${e.rollout_id}\nBatch selected for training at the recorded training_batch_selected time.\n${d.groups?.length ?? "unknown"} groups; earlier collection completion is unknown.`;
  if (e.name === "batch_reject") return `${new Date(e.t0 * 1000).toLocaleString("en-GB")} · Batch ${e.rollout_id}\n${d.reason ?? "unspecified filter"} → ${d.action}\n${d.groups} groups; rejected by the one-step batch filter.`;
  if (e.name === "batch_abort") return `${new Date(e.t0 * 1000).toLocaleString("en-GB")} · batch ${e.rollout_id}\nAbort requested for ${d.groups} unfinished groups, up to ${d.samples_upper_bound ?? "unknown"} sample slots\n${d.action === "retain_partial" ? "Retain partial results for continuation" : "Drop unfinished groups"}; requested counts do not equal actual cancelled HTTP requests or samples.`;
  if (e.name === "weight_published") return `${new Date(e.t0 * 1000).toLocaleString("en-GB")} · weights published: ${d.previous_version ?? "initial"} → ${d.current_version} · policy ${d.trainer_model_id ?? "default"}`;
  if (e.name === "batch_ready") return `Batch ${e.rollout_id} · ${d.groups} groups · V=${d.current_version ?? "unknown"}\nVersions (oldest per group): ${JSON.stringify(d.versions)}\nWorkload / versions: ${JSON.stringify(d.workload_versions)}\nBatch filter removed: ${d.groups_removed_by_batch_filter ?? "not recorded"} groups\nMixed groups: ${d.mixed_groups} · policy ${d.trainer_model_id ?? "default"}`;
  return `${new Date(e.t0 * 1000).toLocaleString("en-GB")} · ${d.reason} → ${d.action}\n${d.groups} groups · versions ${d.oldest_version ?? "unknown"}…${d.newest_version ?? "unknown"} · V=${d.current_version ?? "unknown"}\nQueued after rejection: ${d.queued_groups} · workload ${d.workload ?? "unknown"} · policy ${d.trainer_model_id ?? "default"}`;
}
