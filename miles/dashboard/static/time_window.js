import { el } from "./app.js";

// Every time chart on a page shares this viewport and its follow-mode bounds.
export function createTimeWindow({ bounds, origin, maxSpan = 4 * 3600 } = {}) {
  let start = 0, end = 1, t0 = 0, t1 = 1, ready = false, version = 0;
  const listeners = new Set();
  const get = () => ({ start, end, t0, t1, origin: origin ?? start, maxSpan, ready, version });
  const emit = () => { version++; for (const listener of listeners) listener(get()); };
  function setWindow(a, b) {
    if (!ready || !Number.isFinite(a) || !Number.isFinite(b)) return false;
    const span = Math.min(Math.max(b - a, Math.min(1, end - start)), maxSpan, end - start);
    a = Math.max(start, Math.min(a, end - span)); b = a + span;
    if (Math.abs(a - t0) < 1e-6 && Math.abs(b - t1) < 1e-6) return false;
    [t0, t1] = [a, b]; emit(); return true;
  }
  function updateBounds(range) {
    if (!range || !range.every(Number.isFinite) || range[1] <= range[0]) return;
    if (ready && start === range[0] && end === range[1]) return;
    const trailing = !ready || t1 >= end - 1e-6;
    const span = Math.min(ready ? t1 - t0 : maxSpan, maxSpan, range[1] - range[0]);
    [start, end] = range;
    t0 = trailing ? end - span : Math.max(start, Math.min(t0, end - span));
    t1 = t0 + span; ready = true; emit();
  }
  updateBounds(bounds);
  return { get, setWindow, updateBounds,
    latest: () => setWindow(end - maxSpan, end),
    subscribe: listener => { listeners.add(listener); return () => listeners.delete(listener); },
  };
}

const elapsed = seconds => {
  const s = Math.max(0, Math.round(seconds));
  return `+${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};

export function createTimeBar(controller, label) {
  const thumb = el("span", { class: "time-thumb" }, ["⋮⋮"]);
  const track = el("div", { class: "time-track", role: "scrollbar", tabindex: "0",
    "aria-label": `${label} time range`, "aria-orientation": "horizontal" }, [thumb]);
  const rangeLabel = el("span", { class: "time-range" });
  function zoom(factor) {
    const { t0, t1, start, end, maxSpan } = controller.get(), center = (t0 + t1) / 2;
    const span = Math.min((t1 - t0) * factor, maxSpan, end - start);
    controller.setWindow(center - span / 2, center + span / 2);
  }
  const root = el("div", { class: "time-navigation", "data-time-bar": label }, [
    track, el("div", { class: "time-controls" }, [rangeLabel,
      el("button", { onclick: () => zoom(2), "aria-label": `${label} zoom out` }, ["−"]),
      el("button", { onclick: () => zoom(0.5), "aria-label": `${label} zoom in` }, ["+"]),
    ]),
  ]);
  function update({ start, end, t0, t1, origin, ready }) {
    const full = Math.max(end - start, 1e-9);
    thumb.style.left = `${100 * (t0 - start) / full}%`;
    thumb.style.width = `${100 * (t1 - t0) / full}%`;
    track.setAttribute("aria-valuemin", start);
    track.setAttribute("aria-valuemax", Math.max(start, end - (t1 - t0)));
    track.setAttribute("aria-valuenow", t0);
    track.setAttribute("aria-valuetext", `${elapsed(t0 - origin)} — ${elapsed(t1 - origin)}`);
    track.setAttribute("aria-disabled", String(!ready || t1 - t0 >= full));
    root.setAttribute("data-time-start", t0);
    root.setAttribute("data-time-end", t1);
    rangeLabel.textContent = ready ? `${elapsed(t0 - origin)} — ${elapsed(t1 - origin)} · drag to sync time` : "No time data";
  }
  let drag = null;
  track.onpointerdown = ev => {
    if (ev.button !== 0 || !controller.get().ready) return;
    ev.preventDefault();
    const rect = track.getBoundingClientRect(), state = controller.get();
    if (ev.target !== thumb) {
      const a = state.start + (ev.clientX - rect.left) / rect.width * (state.end - state.start) - (state.t1 - state.t0) / 2;
      controller.setWindow(a, a + state.t1 - state.t0);
    }
    const { t0, t1, start, end } = controller.get();
    drag = { x: ev.clientX, t0, span: t1 - t0, secondsPerPixel: (end - start) / rect.width };
    track.setPointerCapture(ev.pointerId);
  };
  track.onpointermove = ev => {
    if (!drag) return;
    const a = drag.t0 + (ev.clientX - drag.x) * drag.secondsPerPixel;
    controller.setWindow(a, a + drag.span);
  };
  track.onpointerup = track.onpointercancel = track.onlostpointercapture = () => { drag = null; };
  track.onkeydown = ev => {
    const { start, end, t0, t1 } = controller.get(), span = t1 - t0;
    const a = { ArrowLeft: t0 - span / 10, ArrowRight: t0 + span / 10,
      PageUp: t0 - span, PageDown: t0 + span, Home: start, End: end - span }[ev.key];
    if (a === undefined) return;
    ev.preventDefault(); controller.setWindow(a, a + span);
  };
  const unsubscribe = controller.subscribe(update);
  update(controller.get());
  return { root, destroy: unsubscribe };
}
