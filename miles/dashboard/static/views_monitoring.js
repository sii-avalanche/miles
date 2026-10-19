import { el, setViewCleanup } from "./app.js";
import { grafanaUrl } from "./api.js";
import { getRunTimeWindow, createTimeBar } from "./time_window.js";

export async function renderMonitoring(view, meta) {
  const timeWindow = getRunTimeWindow(meta);
  const bar = createTimeBar(timeWindow, "Monitoring");
  const link = el("a", { class: "btn", target: "_blank", rel: "noopener" }, ["Open full Grafana ↗"]);
  const content = el("div");
  view.replaceChildren(el("h2", {}, ["Monitoring & alerts"]),
    el("p", { class: "muted" }, ["Grafana: infrastructure trends, throughput, memory, stage waits and alerts for this run. Overview correlates these symptoms with RL execution; W&B compares experiments."]),
    bar.root, content);
  if (!grafanaUrl(meta)) {
    content.append(el("p", { class: "muted" }, ["Grafana is not connected. Recorded GPU, execution, data pool and training views are available from the navigation above."]));
    setViewCleanup(() => bar.destroy());
    return;
  }
  const frame = grafanaUrl(meta, null, { embed: true }) ? el("iframe", {
    class: "monitoring-frame", title: "Grafana monitoring and alerts for this run",
  }) : null;
  content.append(el("div", { class: "controls" }, [link, el("button", { onclick: applyRange }, ["Apply shared time window"])]),
    el("p", { class: "muted" }, ["The initial range comes from Miles. Use Apply shared time window to reload Grafana after changing the range above. Changes inside Grafana stay in Grafana. If the embedded view is unavailable, use Open full Grafana."]),
    ...(frame ? [frame] : [el("p", { class: "muted" }, ["This Grafana is hosted on another origin; open it using the link above."])]));
  function updateLink() { link.href = grafanaUrl(meta, timeWindow.get()); }
  function applyRange() {
    updateLink();
    if (frame) frame.src = grafanaUrl(meta, timeWindow.get(), { embed: true });
  }
  const unsubscribe = timeWindow.subscribe(updateLink);
  applyRange();
  setViewCleanup(() => { unsubscribe(); bar.destroy(); if (frame) frame.src = "about:blank"; });
}
