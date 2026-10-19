import { api } from "./api.js";
import { el, fmtNum } from "./app.js";
import { hideTooltip, showTooltip } from "./charts.js";

// margins match views_timeline/carpet so all overview panes share an x domain
const M_LEFT = 156;
const M_RIGHT = 14;
const X_BUCKETS = 600;
const PANE_H = 96;
const BAND_FILL = "rgba(47, 111, 235, 0.16)";
const MUTED = "#667080";
const BAD = "#d1442e";
const ACCENT = "#2f6feb";

// createFleet returns {root, refresh({t0?, t1?})} — the scale-invariant
// replacement for the per-rank carpet above the lane-count threshold:
// stacked phase composition (fraction of lanes per phase) plus the util
// distribution band (p10-p90 + median + min) across the whole fleet.
export function createFleet(opts) {
  let data = null;
  let request = 0;
  const composition = el("canvas", { style: `width: 100%; height: ${PANE_H}px; display: block` });
  const band = el("canvas", { style: `width: 100%; height: ${PANE_H}px; display: block` });
  const header = el("div", { class: "controls" }, [el("span", { class: "muted" }, ["fleet overview"])]);
  const root = el("div", { class: "panel" }, [header, composition,
    ...(opts.timeBar ? [opts.timeBar("Fleet phases")] : []), band,
    ...(opts.timeBar ? [opts.timeBar("Fleet utilization")] : []),
  ]);

  async function refresh(range = {}) {
    const current = ++request;
    const response = await api("/api/timeline/fleet", { x_buckets: X_BUCKETS, t0: range.t0, t1: range.t1 });
    if (current !== request) return;
    data = response;
    header.replaceChildren(
      el("span", { class: "muted" }, [`fleet overview · ${data.lanes} lanes`]),
      el("span", { class: "muted", style: "margin-left: auto" }, [
        "top: phase composition · bottom: util p10–p90 band, median, min",
      ]),
    );
    redraw();
  }

  const setup = (canvas) => {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    return { ctx, width: rect.width, height: rect.height };
  };

  function redraw() {
    if (!data) return;
    const range = opts.viewRange?.() ?? data;
    for (const canvas of [composition, band]) {
      canvas.setAttribute("data-time-start", range.t0);
      canvas.setAttribute("data-time-end", range.t1);
    }
    drawComposition();
    drawBand();
  }

  function drawComposition() {
    const { ctx, width, height } = setup(composition);
    ctx.clearRect(0, 0, width, height);
    ctx.font = "11px ui-monospace, monospace";
    const names = (data.palette || []).filter((name) => name && data.composition[name]);
    if (!names.length) {
      ctx.fillStyle = MUTED;
      ctx.fillText("waiting for phase data…", M_LEFT, 16);
      return;
    }
    const plotW = width - M_LEFT - M_RIGHT;
    const range = opts.viewRange?.() ?? data;
    const colW = plotW / data.x_buckets * (data.t1 - data.t0) / Math.max(range.t1 - range.t0, 1e-9);
    const offset = M_LEFT + (data.t0 - range.t0) / Math.max(range.t1 - range.t0, 1e-9) * plotW;
    ctx.save(); ctx.beginPath(); ctx.rect(M_LEFT, 0, plotW, height); ctx.clip();
    for (let x = 0; x < data.x_buckets; x++) {
      let y = height;
      for (const name of names) {
        const frac = data.composition[name][x];
        if (!frac) continue;
        const h = frac * height;
        ctx.fillStyle = opts.phaseColors[name] ?? "#98a1b0";
        ctx.fillRect(offset + x * colW, y - h, Math.max(colW, 1), h);
        y -= h;
      }
    }
    ctx.restore();
    ctx.fillStyle = MUTED;
    ctx.fillText("phase %", 8, height / 2);
  }

  function drawBand() {
    const { ctx, width, height } = setup(band);
    ctx.clearRect(0, 0, width, height);
    ctx.font = "11px ui-monospace, monospace";
    const plotW = width - M_LEFT - M_RIGHT;
    const range = opts.viewRange?.() ?? data;
    const scale = plotW / Math.max(range.t1 - range.t0, 1e-9);
    const X = x => M_LEFT + (data.t0 + (x + 0.5) / data.x_buckets * (data.t1 - data.t0) - range.t0) * scale;
    const Y = (util) => height - (util / 100) * (height - 8) - 2;

    ctx.fillStyle = MUTED;
    ctx.fillText("util %", 8, height / 2);
    for (const tick of [0, 50, 100]) {
      ctx.fillText(String(tick), M_LEFT - 28, Y(tick) + 4);
    }

    // p10-p90 band as per-column strips (null-safe without path bookkeeping)
    const { p10, p50, p90, min } = data.band;
    const colW = (data.t1 - data.t0) / data.x_buckets * scale;
    ctx.save(); ctx.beginPath(); ctx.rect(M_LEFT, 0, plotW, height); ctx.clip();
    ctx.fillStyle = BAND_FILL;
    for (let x = 0; x < data.x_buckets; x++) {
      if (p10[x] === null || p90[x] === null) continue;
      ctx.fillRect(X(x) - colW / 2, Y(p90[x]), Math.max(colW, 1), Y(p10[x]) - Y(p90[x]) + 1);
    }
    const line = (values, color, dash) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.2;
      ctx.setLineDash(dash);
      ctx.beginPath();
      // connect across empty buckets: sparse sampling must not shred the line
      let pen = false;
      values.forEach((value, x) => {
        if (value === null) return;
        pen ? ctx.lineTo(X(x), Y(value)) : ctx.moveTo(X(x), Y(value));
        pen = true;
      });
      ctx.stroke();
      ctx.setLineDash([]);
    };
    line(p50, ACCENT, []);
    line(min, BAD, [3, 3]);
    ctx.restore();
  }

  const bucketAt = (canvas, clientX) => {
    const rect = canvas.getBoundingClientRect();
    const ratio = (clientX - rect.left - M_LEFT) / (rect.width - M_LEFT - M_RIGHT);
    if (ratio < 0 || ratio > 1) return null;
    const range = opts.viewRange?.() ?? data;
    const time = range.t0 + ratio * (range.t1 - range.t0);
    const x = Math.floor((time - data.t0) / Math.max(data.t1 - data.t0, 1e-9) * data.x_buckets);
    return x >= 0 && x < data.x_buckets ? x : null;
  };
  composition.onmousemove = (ev) => {
    if (!data) return;
    const x = bucketAt(composition, ev.clientX);
    if (x === null) return hideTooltip();
    const t = data.t0 + ((x + 0.5) / data.x_buckets) * (data.t1 - data.t0);
    const lines = [`+${fmtNum(t - opts.runStart())}s`];
    for (const name of data.palette || []) {
      const frac = name && data.composition[name] ? data.composition[name][x] : 0;
      if (frac > 0.002) lines.push(`${name}: ${(frac * 100).toFixed(1)}%`);
    }
    showTooltip(ev.clientX, ev.clientY, lines.join("\n"));
  };
  band.onmousemove = (ev) => {
    if (!data) return;
    const x = bucketAt(band, ev.clientX);
    if (x === null || data.band.p50[x] === null) return hideTooltip();
    const t = data.t0 + ((x + 0.5) / data.x_buckets) * (data.t1 - data.t0);
    showTooltip(
      ev.clientX,
      ev.clientY,
      [
        `+${fmtNum(t - opts.runStart())}s`,
        `median ${fmtNum(data.band.p50[x])}%`,
        `p10–p90 ${fmtNum(data.band.p10[x])}–${fmtNum(data.band.p90[x])}%`,
        `min ${fmtNum(data.band.min[x])}%`,
      ].join("\n"),
    );
  };
  composition.onmouseleave = hideTooltip;
  band.onmouseleave = hideTooltip;

  return { root, refresh, redraw, destroy: () => { request++; } };
}
