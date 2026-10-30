import { api } from "./api.js";
import { el, fmtNum } from "./app.js";
import { hideTooltip, showTooltip } from "./charts.js";

const LEFT = 96, RIGHT = 14, TOP = 28, ROW = 26;
const COLORS = ["#287bdd", "#15b58c", "#efa000", "#ec83b4", "#e5484d", "#5840aa", "#999187"];
const LABELS = ["V", "V−1", "V−2", "V−3", "Older", "Future", "Unknown"];
const STAGES = {rollout: ["Collect", "#efa000"], train_wait: ["Trainer wait", "#aaa"],
  train: ["Train", "#287bdd"], update_weights: ["Train → infer", "#ec83b4"]};

export function createDataPool(meta) {
  const canvas = el("canvas", {style: "width:100%;display:block"});
  const table = el("tbody");
  const root = el("div", {class: "panel"}, [
    el("h3", {}, ["Rollout, data pool, training and weight sync"]),
    el("p", {class: "muted"}, ["Engine green = requests running; gray = observed idle; gaps = no sample. Pool counts are queued groups, bucketed by oldest weight version relative to the last publication. Phase bars reuse the recorded Miles timers."]),
    el("div", {class: "legend"}, LABELS.map((label, i) => el("span", {style: `color:${COLORS[i]};margin-right:12px`}, [`■ ${label}`]))),
    canvas, el("details", {}, [el("summary", {}, ["Latest pool events"]), el("table", {}, [table])]),
  ]);
  let state = null, hits = [];
  async function refresh(range = {}) {
    const [pool, engines, phases] = await Promise.all([
      api("/api/timeline/data_pool", range),
      api("/api/timeline/engine_series", {...range, metric: "sglang_num_running_reqs", max_points: 4000}),
      api("/api/timeline/phases", range),
    ]);
    state = {events: pool.events.some(e => e.event !== "snapshot") ? pool.events.filter(e => e.event !== "snapshot") : pool.events, engines: engines.series, phases: phases.phases, range};
    redraw();
  }
  function redraw() {
    if (!state) return;
    const {events, engines, phases, range} = state;
    const [a, b] = [range.t0 ?? meta.time_range?.[0], range.t1 ?? meta.time_range?.[1]];
    if (!(b > a)) return;
    const pools = [...new Set(events.map(e => e.trainer_model_id ?? "all"))];
    const rows = [...engines.map((s, i) => ({label: `Engine ${i + 1}`, engine: s})),
      ...Object.entries(STAGES).map(([name, [label, color]]) => ({name, label, color})),
      ...pools.map(pool => ({label: `Pool ${pool}`, pool})), {label: "Batch ready", event: "batch_ready"}, {label: "Rejected", event: "reject"}];
    canvas.style.height = `${TOP + rows.length * ROW + 8}px`;
    const rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr); ctx.font = "11px ui-monospace, monospace";
    const X = t => LEFT + (t - a) / (b - a) * (rect.width - LEFT - RIGHT);
    hits = [];
    function box(t0, t1, row, color, detail, height = 15, offset = 0) {
      if (t1 < a || t0 > b) return;
      const x = X(Math.max(a, t0)), end = X(Math.min(b, t1)), y = TOP + row * ROW + offset;
      ctx.fillStyle = color; ctx.fillRect(x, y, Math.max(2, end - x), height);
      hits.push({x, end: Math.max(x + 3, end), y, height, detail});
    }
    for (let i = 0; i <= 6; i++) {
      const t = a + (b - a) * i / 6; ctx.fillStyle = "#7a7168";
      ctx.fillText(`+${fmtNum(t - (meta.start_ts ?? a))}s`, X(t) - 18, 14);
    }
    rows.forEach((row, index) => {
      ctx.fillStyle = "#231f1c"; ctx.fillText(row.label, 4, TOP + index * ROW + 12, LEFT - 10);
      if (row.engine) {
        const s = row.engine, gap = Math.max(10, 3 * (meta.engine_interval_s ?? 2));
        s.ts.forEach((t, i) => box(t, Math.min(s.ts[i + 1] ?? b, t + gap), index,
          s.value[i] > 0 ? "#15b58c" : "#aaa", `${s.addr}\n${new Date(t * 1000).toLocaleTimeString()} · ${s.value[i]} requests`));
      } else if (row.name) {
        const seen = new Set();
        for (const phase of phases) if ((phase.name === row.name || phase.name.endsWith(`/${row.name}`))) {
          const key = `${phase.t0}/${phase.t1}`;
          if (seen.has(key)) continue; seen.add(key);
          box(phase.t0, phase.t1, index, row.color, `${row.label}\n${new Date(phase.t0 * 1000).toLocaleTimeString()} · ${fmtNum(phase.t1 - phase.t0)}s`);
        }
      } else if (row.pool) {
        const samples = events.filter(e => (e.trainer_model_id ?? "all") === row.pool);
        const peak = samples.reduce((max, s, i) => s.ts <= b && (samples[i + 1]?.ts ?? b) >= a ? Math.max(max, s.length) : max, 1);
        samples.forEach((s, i) => {
          const buckets = Array(7).fill(0);
          const versions = Object.keys(s.versions ?? {}).length ? s.versions : {unknown: s.length};
          for (const [version, count] of Object.entries(versions)) {
            const lag = s.current_version - Number(version);
            const bucket = version === "unknown" || s.current_version == null ? 6 : lag < 0 ? 5 : lag > 3 ? 4 : lag;
            buckets[bucket] += count;
          }
          const detail = `${row.label} · ${new Date(s.ts * 1000).toLocaleTimeString()}\nQueued ${s.length} / ${s.capacity ?? "?"} · reference V ${s.current_version ?? "unknown"}\n${LABELS.map((v, n) => `${v}: ${buckets[n]}`).join(" · ")}\nMixed (subset): ${s.mixed_groups}`;
          let offset = 18;
          buckets.forEach((count, n) => { const height = count / peak * 18; offset -= height;
            if (count) box(s.ts, samples[i + 1]?.ts ?? b, index, COLORS[n], detail, height, offset); });
        });
      } else {
        for (const s of events) if (s.event === row.event) box(s.ts, s.ts, index, row.event === "reject" ? "#e5484d" : "#efa000",
          `${s.event} · ${new Date(s.ts * 1000).toLocaleTimeString()}\nPool ${s.trainer_model_id ?? "all"} · batch ${s.rollout_id ?? "—"} · ${s.reason ?? ""}`);
      }
    });
    table.replaceChildren(...events.filter(e => e.ts >= a && e.ts <= b && !["put", "get", "snapshot"].includes(e.event)).slice(-100).reverse().map(e =>
      el("tr", {}, [new Date(e.ts * 1000).toLocaleTimeString(), e.event, e.trainer_model_id ?? "all", `queued ${e.length}`, e.rollout_id ?? "", e.reason ?? ""].map(v => el("td", {}, [String(v)])))));
  }
  canvas.onmousemove = e => {
    const rect = canvas.getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top;
    const hit = hits.find(h => x >= h.x && x <= h.end && y >= h.y && y <= h.y + h.height);
    hit ? showTooltip(e.clientX, e.clientY, hit.detail) : hideTooltip();
  };
  canvas.onmouseleave = hideTooltip;
  return {root, refresh, redraw};
}
