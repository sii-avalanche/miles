import { api } from "./api.js";
import { el, setViewCleanup, fmtNum } from "./app.js";
import { drawMultiLine, drawChart, SERIES_COLORS } from "./charts.js";

const R = "runtime/rollout/";
const D = "runtime/driver/";
const PANELS = [
  ["GPU utilization (%)", [[D + "fleet/train/gpu_util_mean_pct", "Trainer"], [D + "fleet/inference/gpu_util_mean_pct", "Inference"]]],
  ["Collection / buffer (fraction)", [[R + "drain/progress_fraction", "Collected / target"], [R + "buffer/fill_fraction", "Buffer / capacity"]]],
  ["Waiting seconds per wall second", [[D + "phase/wait_rollout/seconds_per_s", "Driver waiting for rollout"], [R + "buffer/full_wait/seconds_per_s", "Producer blocked on full buffer"], [R + "buffer/empty_wait/seconds_per_s", "Consumer waiting for a group"]]],
  ["Group rates (/s)", [[R + "producer/groups_completed_per_s", "Generated"], [R + "buffer/groups_consumed_per_s", "Selected"], [R + "buffer/dynamic_rejected_groups_per_s", "Dynamic rejects"], [R + "buffer/stale_rejected_groups_per_s", "Stale rejects"]]],
  ["Production and long tail", [[R + "producer/in_flight_groups", "Active groups"], [R + "producer/completed_pending_groups", "Completed tasks awaiting collection"]]],
  ["Oldest group / event loop delay (s)", [[R + "producer/oldest_in_flight_seconds", "Oldest in-flight group"], [R + "event_loop/lag_seconds", "Event loop lag"], [R + "event_loop/heartbeat_age_seconds", "Seconds since loop heartbeat"]]],
  ["Database backlog", [[R + "record/pending_writes", "Pending writes"], [R + "record/pool/requests_waiting", "Waiting for connection"], [R + "record/pool/pool_available", "Available connections"]]],
  ["Database results (/s)", [[R + "record/success_per_s", "Success"], [R + "record/pool_timeout_per_s", "PoolTimeout"], [R + "record/other_errors_per_s", "Other errors"], [R + "record/pool/connections_errors_per_s", "Connection establishment errors"]]],
  ["Connection pool", [[R + "record/pool/pool_size", "Pool size"], [R + "record/pool/pool_available", "Available"], [R + "record/pool/pool_max", "Pool limit"]]],
  ["Record latency (s, last 256 completions)", [[R + "record/end_to_end/p95_seconds", "End-to-end p95"], [R + "record/create_rollout/p95_seconds", "create_rollout p95"], [R + "record/oldest_pending_seconds", "Oldest pending write"]]],
  ["CPU memory (GiB)", [[R + "host/process_rss_gb", "Executor RSS"], [R + "host/container_memory_gb", "Container usage"], [R + "host/container_memory_limit_gb", "Container limit"]]],
  ["Executor CPU and event loop", [[R + "host/process_cpu_seconds_per_s", "CPU cores busy (1 = one core)"], [R + "event_loop/lag_seconds", "Event loop lag (s)"]]],
  ["Inference requests / KV", [[D + "fleet/inference/queued_requests", "Queued requests"], [D + "fleet/inference/running_requests", "Running requests"]]],
  ["KV usage (fraction)", [[D + "fleet/inference/kv_usage_mean", "Mean"], [D + "fleet/inference/kv_usage_max", "Max"]]],
  ["Mamba state usage (fraction)", [[D + "fleet/inference/mamba_usage_mean", "Mean"], [D + "fleet/inference/mamba_usage_max", "Max"]]],
  ["GPU telemetry coverage (fraction)", [[D + "fleet/train/coverage_fraction", "Trainer"], [D + "fleet/inference/coverage_fraction", "Inference"]]],
];

const PHASE_COLORS = { wait_rollout: "#a9a9a9", train: "#287fd0", critic_train: "#437aaa", update_weights: "#e69b35", checkpoint: "#8c6baf", eval_dispatch: "#249d87" };

function legend(specs) {
  return el("p", { class: "muted", style: "font-size:12px" }, specs.map(([, label], i) =>
    el("span", { style: `margin-right:12px;color:${SERIES_COLORS[i % SERIES_COLORS.length]}` }, [label])));
}

function stepRows(events) {
  const final = new Map();
  for (const event of events) {
    if (event.role === "driver" && event.rollout_id !== null) final.set(`${event.rollout_id}/${event.name}/${event.t0}`, event);
  }
  const byStep = new Map();
  for (const event of final.values()) {
    if (!byStep.has(event.rollout_id)) byStep.set(event.rollout_id, []);
    byStep.get(event.rollout_id).push(event);
  }
  return [...byStep.entries()].sort((a, b) => b[0] - a[0]).map(([step, phases]) => {
    const start = Math.min(...phases.map(p => p.t0));
    const end = Math.max(...phases.map(p => p.t1 ?? Date.now() / 1000));
    const failed = phases.find(p => p.status === "failed");
    const cancelledPhase = phases.find(p => p.status === "cancelled");
    const completed = phases.some(p => p.name === "step" && p.status === "completed");
    const state = failed ? `${failed.name}: ${failed.error_type}` : completed ? "completed" : cancelledPhase ? `${cancelledPhase.name}: cancelled` : "open (end not observed)";
    const bar = el("div", { style: "height:22px;position:relative;background:#f1f1f1;min-width:320px" });
    for (const phase of phases) {
      if (!(phase.name in PHASE_COLORS)) continue;
      const duration = (phase.t1 ?? end) - phase.t0;
      bar.append(el("span", {
        title: `${phase.name}: ${duration.toFixed(2)} s; ${phase.status}`,
        style: `position:absolute;left:${100 * (phase.t0 - start) / Math.max(end - start, 0.01)}%;width:${100 * duration / Math.max(end - start, 0.01)}%;height:100%;background:${phase.status === "failed" ? "#bb3030" : PHASE_COLORS[phase.name]}`,
      }));
    }
    return el("tr", {}, [el("td", {}, [String(step)]), el("td", {}, [state]), el("td", {}, [fmtNum(end - start)]), el("td", {}, [bar])]);
  });
}

export async function renderRuntime(view, meta) {
  const status = el("p", { class: "muted" });
  const grid = el("div", { style: "display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px" });
  const tableBody = el("tbody");
  const panels = PANELS.map(([title, specs]) => {
    const canvas = el("canvas", { class: "chart" });
    const note = el("p", { class: "muted" }, ["No samples yet"]);
    grid.append(el("div", { class: "panel" }, [el("h3", {}, [title]), legend(specs), note, canvas]));
    return { canvas, note, specs };
  });
  const moeGrid = el("div", { style: "display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px" });
  const workloadGrid = el("div", { style: "display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px" });
  const workloadPanels = [];
  let moePanels = [];
  view.replaceChildren(
    el("h2", {}, ["RL bottlenecks"]), status,
    el("p", { class: "muted" }, ["Runtime charts show the last four hours in elapsed seconds. GPU utilization measures device activity; it is not kernel occupancy or MFU. A failure identifies a stage, not its root cause. Missing telemetry is shown as missing."]),
    grid,
    el("h3", {}, ["Workload collection and in-flight share"]),
    el("p", { class: "muted" }, ["Collection progress = valid groups / (batch group target × workload target fraction). In-flight group share is a scheduling proxy; it does not measure GPU allocation. Targets describe the desired mix and do not change the scheduler."]),
    workloadGrid,
    el("h3", {}, ["Expert load of the served policy"]),
    el("p", { class: "muted" }, ["Optional routing replay statistics for the selected training batch, including prompt positions. These are inference-side assignments, not trainer measurements."]),
    moeGrid,
    el("h3", {}, ["Driver step timeline"]),
    el("p", { class: "muted" }, ["Gray: wait for rollout; blue: train; orange: weight sync; purple: checkpoint; green: eval dispatch; red: observed failure. Open intervals do not prove the worker is alive. Async phases in different processes may overlap."]),
    el("table", {}, [el("thead", {}, [el("tr", {}, ["Rollout ID", "Status / observed error", "Wall seconds", "Phases (hover)"].map(x => el("th", {}, [x])))]), tableBody]),
  );
  let cancelled = false;
  let refreshing = false;
  async function refresh() {
    if (cancelled || refreshing) return;
    refreshing = true;
    try {
      const liveMeta = await api("/api/meta");
      for (const suffix of ["collection_progress_fraction", "in_flight_group_fraction"]) {
        if (workloadPanels.some(p => p.suffix === suffix)) continue;
        const specs = liveMeta.metric_keys.filter(k => k.startsWith(R + "workload/") && k.endsWith("/" + suffix)).map(k => [k, k.split("/").at(-2)]);
        if (!specs.length) continue;
        const canvas = el("canvas", { class: "chart" });
        const note = el("p", { class: "muted" });
        workloadGrid.append(el("div", { class: "panel" }, [el("h3", {}, [suffix]), legend(specs), note, canvas]));
        workloadPanels.push({ canvas, note, specs, suffix });
      }
      if (!moePanels.length) {
        const keys = liveMeta.metric_keys.filter(k => /^moe\/rollout_layer_\d+\/(cv|max_over_mean|cold_experts_fraction)$/.test(k));
        for (const key of keys) {
          const canvas = el("canvas", { class: "chart" });
          moeGrid.append(el("div", { class: "panel" }, [el("h3", {}, [key]), canvas]));
          moePanels.push({ key, canvas });
        }
      }
      const allPanels = [...panels, ...workloadPanels];
      const requested = [...new Set(allPanels.flatMap(p => p.specs.map(([key]) => key)))];
      const t1 = liveMeta.time_range?.[1];
      const t0 = t1 === undefined ? undefined : t1 - 4 * 3600;
      const [series, { events }] = await Promise.all([
        api("/api/metrics", { keys: requested.join(","), x: "runtime/time_s", t0, t1 }),
        api("/api/runtime/events"),
      ]);
      if (cancelled) return;
      const withData = Object.values(series).filter(s => s.x?.length);
      const xDomain = withData.length ? [Math.min(...withData.map(s => s.x[0])), Math.max(...withData.map(s => s.x.at(-1)))] : undefined;
      const maxGapSeconds = Math.max(30, 3 * (liveMeta.runtime_interval_s || 10));
      for (const { canvas, specs, note } of allPanels) {
        const available = specs.filter(([key]) => series[key]?.x?.length);
        const stale = available.filter(([key]) => Date.now() / 1000 - (series[key].ts?.at(-1) ?? 0) > maxGapSeconds);
        note.textContent = !available.length ? "No samples available for this panel" : stale.length ? `Last samples older than ${maxGapSeconds} s: ${stale.map(([, label]) => label).join(", ")}` : "";
        drawMultiLine(canvas, available.map(([key, label]) => ({ label, ts: series[key].x, value: series[key].y })), {
          timeOrigin: 0, xDomain, maxGapSeconds,
          colorIndex: label => specs.findIndex(([, name]) => name === label),
        });
      }
      const stamps = Object.values(series).flatMap(s => s.ts ?? []);
      const latest = stamps.reduce((a, b) => Math.max(a, b), 0);
      status.textContent = latest ? `Latest runtime sample: ${Math.max(0, Date.now() / 1000 - latest).toFixed(1)} s ago · ${liveMeta.mode}` : "Runtime monitoring has no samples yet. Enable perf_monitor_interval for the next run.";
      tableBody.replaceChildren(...stepRows(events));
      if (moePanels.length) {
        const values = await api("/api/metrics", { keys: moePanels.map(p => p.key).join(","), x: "rollout/step" });
        if (!cancelled) for (const { key, canvas } of moePanels) {
          const s = values[key];
          drawChart(canvas, (s?.x ?? []).map((x, i) => ({ x, y: s.y[i] })), { line: true });
        }
      }
    } catch (error) {
      if (!cancelled) status.textContent = String(error);
    } finally {
      refreshing = false;
    }
  }
  await refresh();
  const interval = meta.mode === "follow" ? setInterval(refresh, 5000) : null;
  setViewCleanup(() => { cancelled = true; if (interval) clearInterval(interval); });
}
