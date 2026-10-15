import { api } from "./api.js";
import { el, setViewCleanup } from "./app.js";
import { createAsyncFlow, loadAsyncFlow } from "./async_flow.js";
import { getRunTimeWindow, createTimeBar } from "./time_window.js";
import { createRuntimeStatus } from "./runtime_status.js";
import { createFleet } from "./fleet.js";
import { PHASE_COLORS } from "./phase_colors.js";

export async function renderRuntime(view, meta, route = {}) {
  const dataView = route.view === "data";
  const timeWindow = getRunTimeWindow(meta);
  const timeBars = [];
  const timeBar = label => { const bar = createTimeBar(timeWindow, label); timeBars.push(bar); return bar.root; };
  const asyncFlow = createAsyncFlow({ timeBar, showHistory: dataView, showEngines: dataView });
  const flowHelp = asyncFlow.root.querySelector?.(":scope > p.muted");
  if (flowHelp) {
    const help = el("details", { class: "flow-help" }, [el("summary", {}, ["Timeline legend and counting rules"])]);
    flowHelp.before(help);
    help.append(flowHelp);
  }
  const runtimeStatus = createRuntimeStatus();
  const fleet = !dataView && meta.capabilities?.has_timeline ? createFleet({
    phaseColors: PHASE_COLORS, runStart: () => meta.start_ts, timeBar,
    viewRange: () => timeWindow.get(),
  }) : null;
  let flowData, flowMeta;
  const status = el("p", { class: "muted" });
  const gpuStatus = el("p", { class: "muted" });
  const shortcuts = dataView ? [
    ["Rollout samples & traces", "#/rollout/latest"], ["Overview", "#/"],
  ] : [
    ["GPU details →", "#/gpu"], ["Data pool & group history →", "#/data"],
    ["Training metrics →", "#/metrics"], ["Monitoring & alerts →", "#/monitoring"],
  ];
  view.replaceChildren(
    el("h2", {}, [dataView ? "Data & samples" : "Overview"]),
    el("p", { class: "muted" }, [dataView
      ? "Data pool versions, batch selection, rejected groups and retries. Open samples to inspect conversations, tokens and Traces."
      : "GPU utilization and the complete RL pipeline on one time window: generation, reward, data pool, trainer wait, training and weight sync."]),
    el("nav", { class: "compute-sections", "aria-label": "Related views" }, shortcuts.map(([label, href]) => el("a", { href }, [label]))),
    timeBar("Shared window"), status,
    ...(!dataView ? [runtimeStatus.root,
      el("h3", {}, ["GPU utilization & phase composition"]),
      el("p", { class: "muted" }, ["All recorded GPUs: utilization median, p10–p90 and minimum. Phase colors describe controller activity; gaps mean missing samples."]),
      ...(fleet ? [fleet.root, gpuStatus] : [el("p", { class: "muted" }, ["GPU telemetry was not recorded for this run."])]),
    ] : []),
    asyncFlow.root,
  );
  let cancelled = false;
  let refreshing = false;
  let pendingRefresh = false, refreshTimer = null;
  function drawRuntime() {
    if (cancelled || !flowData) return;
    const range = timeWindow.get();
    const events = flowData.events.filter(e => e.t0 <= range.t1 && (e.t1 ?? range.t1) >= range.t0);
    asyncFlow.draw({ ...flowData, events,
      history: { ...flowData.history, rows: (flowData.history?.rows ?? []).filter(r => r.ts >= range.t0 && r.ts <= range.t1) },
    }, flowMeta, range.t0, range.t1, range.origin);

  }
  function scheduleRefresh() {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refresh, 300);
  }
  const unsubscribe = timeWindow.subscribe(() => { drawRuntime(); fleet?.redraw(); scheduleRefresh(); });
  let interval = null;
  const onResize = () => { drawRuntime(); fleet?.redraw(); };
  window.addEventListener("resize", onResize);
  setViewCleanup(() => {
    cancelled = true;
    if (interval) clearInterval(interval);
    clearTimeout(refreshTimer);
    unsubscribe();
    for (const bar of timeBars) bar.destroy();
    fleet?.destroy();
    window.removeEventListener("resize", onResize);
  });
  async function refresh() {
    if (cancelled) return;
    if (refreshing) { pendingRefresh = true; return; }
    refreshing = true;
    let requestedVersion;
    try {
      const liveMeta = await api("/api/meta");
      if (cancelled) return;
      timeWindow.updateBounds(liveMeta.time_range);
      const range = timeWindow.get();
      requestedVersion = range.version;
      const { t0, t1 } = range;
      const [flow, currentStatus] = await Promise.all([
        loadAsyncFlow(liveMeta, t0, t1),
        api("/api/runtime/status").catch(() => null),
        fleet?.refresh({ t0, t1 }).then(() => { gpuStatus.textContent = ""; }).catch(error => {
          if (!cancelled) gpuStatus.textContent = `GPU overview unavailable: ${error.message}`;
        }),
      ]);
      if (cancelled || requestedVersion !== timeWindow.get().version) return;
      flowData = flow; flowMeta = liveMeta;
      runtimeStatus.update(currentStatus);
      drawRuntime();
      const observed = currentStatus?.rollout?.observed_at;
      status.textContent = observed ? `Latest rollout sample: ${Math.max(0, Date.now() / 1000 - observed).toFixed(1)} s ago · ${liveMeta.mode}` : "No rollout samples recorded yet.";
    } catch (error) {
      if (!cancelled) { status.textContent = String(error); runtimeStatus.update(null); }
    } finally {
      refreshing = false;
      if (!cancelled && (pendingRefresh || (requestedVersion !== undefined && requestedVersion !== timeWindow.get().version))) {
        pendingRefresh = false; scheduleRefresh();
      }
    }
  }
  await refresh();
  if (cancelled) return;
  if (!cancelled && meta.mode === "follow") interval = setInterval(refresh, 5000);
}
