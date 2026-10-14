import { api } from "./api.js";
import { el, setViewCleanup } from "./app.js";
import { createAsyncFlow, loadAsyncFlow } from "./async_flow.js";
import { createTimeWindow, createTimeBar } from "./time_window.js";
import { createRuntimeStatus } from "./runtime_status.js";
import { grafanaUrl } from "./observability.js";

export async function renderRuntime(view, meta) {
  const timeWindow = createTimeWindow({ bounds: meta.time_range, origin: meta.start_ts,
    maxSpan: Math.min(meta.capabilities?.max_window_s ?? 14400, 14400) });
  const timeBars = [];
  const timeBar = label => { const bar = createTimeBar(timeWindow, label); timeBars.push(bar); return bar.root; };
  const asyncFlow = createAsyncFlow({ timeBar });
  asyncFlow.root.id = "execution";
  const flowHelp = asyncFlow.root.querySelector?.(":scope > p.muted");
  if (flowHelp) {
    const help = el("details", { class: "flow-help" }, [el("summary", {}, ["Timeline legend and counting rules"])]);
    flowHelp.before(help);
    help.append(flowHelp);
  }
  const runtimeStatus = createRuntimeStatus();
  let flowData, flowMeta;
  const status = el("p", { class: "muted" });
  const utilizationView = el("div");
  utilizationView.id = "gpu";
  const grafana = grafanaUrl(meta) ? el("a", { class: "btn grafana-link", href: grafanaUrl(meta, timeWindow.get()), target: "_blank", rel: "noopener" }, ["Grafana · Metrics and alerts ↗"]) : null;
  view.replaceChildren(
    el("div", { class: "compute-heading" }, [el("div", {}, [el("h2", {}, ["RL Compute"]),
      el("p", { class: "muted" }, ["Batch collection, training, weight versions and group history."])]), ...(grafana ? [grafana] : [])]),
    el("nav", { class: "compute-sections", "aria-label": "RL Compute sections" }, [
      el("a", { href: "#execution", onclick: event => { event.preventDefault(); asyncFlow.root.scrollIntoView({ behavior: "smooth" }); } }, ["Execution · Batch / Wait / Weight"]),
      el("a", { href: "#gpu", onclick: event => { event.preventDefault(); utilizationView.scrollIntoView({ behavior: "smooth" }); } }, ["GPU · Utilization and phases"]),
      el("a", { href: "#groups", onclick: event => { event.preventDefault(); asyncFlow.root.querySelector(".group-history")?.scrollIntoView({ behavior: "smooth" }); } }, ["Data history · Prompt / Group / Attempt"]),
    ]), status,
    el("p", { class: "muted" }, ["Time charts share a window. Select rejection events to inspect prompts, attempts and subsequent training. GPU utilization shows observed activity; missing samples remain blank."]),
    runtimeStatus.root, asyncFlow.root, utilizationView,
  );
  let utilizationCleanup = null;
  let cancelled = false;
  let refreshing = false;
  let pendingRefresh = false, refreshTimer = null;
  function drawRuntime() {
    if (cancelled || !flowData) return;
    const range = timeWindow.get();
    if (grafana) grafana.href = grafanaUrl(flowMeta || meta, range);
    const events = flowData.events.filter(e => e.t0 <= range.t1 && (e.t1 ?? range.t1) >= range.t0);
    asyncFlow.draw({ ...flowData, events,
      history: { ...flowData.history, rows: (flowData.history?.rows ?? []).filter(r => r.ts >= range.t0 && r.ts <= range.t1) },
    }, flowMeta, range.t0, range.t1, range.origin);

  }
  function scheduleRefresh() {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refresh, 300);
  }
  const unsubscribe = timeWindow.subscribe(() => { drawRuntime(); scheduleRefresh(); });
  let interval = null;
  const onResize = drawRuntime;
  window.addEventListener("resize", onResize);
  setViewCleanup(() => {
    cancelled = true;
    if (interval) clearInterval(interval);
    clearTimeout(refreshTimer);
    unsubscribe();
    for (const bar of timeBars) bar.destroy();
    utilizationCleanup?.();
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
  if (meta.capabilities?.has_timeline) {
    try {
      const { renderTimeline } = await import("./views_timeline.js");
      await renderTimeline(utilizationView, meta, null, {
        embedded: true, timeWindow, registerCleanup: cleanup => { utilizationCleanup = cleanup; },
      });
    } catch (error) {
      utilizationView.replaceChildren(el("p", { class: "error" }, [String(error)]));
    }
  }
  if (!cancelled && meta.mode === "follow") interval = setInterval(refresh, 5000);
}
