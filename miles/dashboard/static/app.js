import { APP_BASE, getMeta } from "./api.js";
import { renderMetrics } from "./views_metrics.js";
import { renderRollout } from "./views_rollout.js";
import { renderTokens } from "./views_tokens.js";
import { renderOverview } from "./views_overview.js";
import { renderMonitoring } from "./views_monitoring.js";

// tiny DOM builder: el("div", {class: "x", onclick: fn}, [children|strings])
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) node[k] = v;
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const child of children) {
    node.append(child);
  }
  return node;
}

// views with background work (follow-mode auto refresh) register a cleanup
// here; render() runs it before switching views so intervals never leak
let activeViewCleanup = null;

export function setViewCleanup(cleanup) {
  activeViewCleanup = cleanup;
}

export const fmtNum = (v) => {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "✓" : "";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return String(v);
  const a = Math.abs(v);
  if (a >= 1e5 || (a < 1e-3 && a > 0)) return v.toExponential(2);
  return v.toFixed(3);
};

export function statBox(label, value) {
  return el("div", { class: "stat" }, [el("div", { class: "v" }, [fmtNum(value)]), el("div", { class: "k" }, [label])]);
}

function parseRoute() {
  const [path, query] = (location.hash.slice(1) || "/").split("?");
  const segments = path.split("/").filter(Boolean);
  const params = new URLSearchParams(query || "");
  if (!segments.length || ["overview", "runtime", "gpu", "timeline", "data"].includes(segments[0])) {
    return { view: "overview", lanes: params.get("lanes") };
  }
  if (segments[0] === "monitoring") return { view: "monitoring" };
  if (segments[0] === "rollout" && segments.length >= 2) {
    const evaluation = params.get("eval") === "1";
    // "latest" is resolved against the dump at render time rather than baked
    // into the link, because the newest step is often listed before it is
    // readable; a typed step number is always honoured exactly
    if (segments[1] === "latest") {
      return { view: "rollout", rolloutId: null, evaluation };
    }
    const rolloutId = Number(segments[1]);
    if (segments[2] === "sample" && segments.length === 4) {
      return {
        view: "tokens",
        rolloutId,
        sampleIndex: Number(segments[3]),
        sampleOccurrence: Number(params.get("occurrence") || 0),
        evaluation,
      };
    }
    return { view: "rollout", rolloutId, evaluation };
  }
  return { view: "metrics" };
}

function crumbs(route, meta) {
  const nav = (label, href, active, onclick = null) =>
    el("a", { class: `nav${active ? " active" : ""}`, href, onclick }, [label]);
  const parts = [];
  if (/\/runs\/[a-f0-9]+\/$/.test(APP_BASE.pathname)) {
    parts.push(nav("All runs", new URL("../../", APP_BASE).href, false));
  }
  parts.push(nav("Overview", "#/", route.view === "overview"));
  parts.push(nav("Training metrics", "#/metrics", route.view === "metrics"));
  parts.push(nav("Monitoring & alerts", "#/monitoring", route.view === "monitoring"));
  // Keep the sample table accessible from its detail views.
  if (["rollout", "tokens"].includes(route.view) && meta.rollout_ids.train.length) {
    const onRollout = route.view === "rollout" || route.view === "tokens";
    const replaceInPlace = (event) => {
      event.preventDefault();
      location.replace("#/rollout/latest");
    };
    parts.push(nav("Rollouts", "#/rollout/latest", onRollout, onRollout ? replaceInPlace : null));
  }
  if ((route.view === "rollout" || route.view === "tokens") && route.rolloutId !== null) {
    const evalSuffix = route.evaluation ? "?eval=1" : "";
    parts.push(
      el("span", { class: "crumb" }, [
        "› ",
        el("a", { href: `#/rollout/${route.rolloutId}${evalSuffix}` }, [
          `${route.evaluation ? "eval " : ""}step ${route.rolloutId}`,
        ]),
      ]),
    );
  }
  if (route.view === "tokens") {
    const occurrence = route.sampleOccurrence ? ` · leaf ${route.sampleOccurrence + 1}` : "";
    parts.push(el("span", { class: "crumb" }, [`› sample ${route.sampleIndex}${occurrence}`]));
  }
  document.getElementById("crumbs").replaceChildren(...parts);
}

let renderVersion = 0;
async function render() {
  const version = ++renderVersion;
  if (activeViewCleanup) {
    activeViewCleanup();
    activeViewCleanup = null;
  }
  const route = parseRoute();
  const view = document.getElementById("view");
  view.replaceChildren(el("p", { class: "muted" }, ["loading…"]));
  try {
    const meta = await getMeta();
    if (version !== renderVersion) return;
    crumbs(route, meta);
    const modeLabel =
      meta.mode === "follow" ? "Auto-refresh enabled; source freshness determines current activity" : "Static snapshot; auto-refresh disabled";
    const runinfo = [
      el("span", {}, [`run: ${meta.run_name ?? "unnamed run"}`]),
      " · ",
      el("span", { title: modeLabel }, [meta.mode === "follow" ? "● live" : "◼ static"]),
      ...(meta.capabilities.has_metrics ? [] : [" · dump-derived metrics"]),
    ];
    if (meta.wandb_url) runinfo.push(" · ", el("a", { href: meta.wandb_url, target: "_blank", rel: "noopener", title: "Compare experiments, configs and training curves" }, ["W&B · Experiments ↗"]));
    document.getElementById("runinfo").replaceChildren(...runinfo);
    if (route.view === "metrics") await renderMetrics(view, meta);
    else if (route.view === "overview") await renderOverview(view, meta, route);
    else if (route.view === "monitoring") await renderMonitoring(view, meta);
    else if (route.view === "rollout") await renderRollout(view, meta, route);
    else await renderTokens(view, meta, route);
  } catch (err) {
    if (version === renderVersion) view.replaceChildren(el("div", { class: "error" }, [String(err)]));
  }
}

window.addEventListener("hashchange", render);
render();
