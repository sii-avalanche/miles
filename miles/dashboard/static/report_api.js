// Offline adapter over the native MetricStore query results embedded at export time.
export async function getMeta() { return REPORT.meta; }
const laneKey = lane => `${lane.node}:${lane.gpu}`;
function selected(grammar) {
  const lanes = REPORT.topology.lanes;
  if (!grammar || grammar.trim() === "all") return new Set(lanes.map(laneKey));
  const predicates = grammar.split(",").filter(t => t.trim()).map(raw => {
    const token = raw.trim(), pos = token.indexOf(":"), kind = token.slice(0, pos), value = token.slice(pos + 1);
    if (pos < 0 || !value) throw new Error(`Invalid lane selector: ${token}`);
    if (kind === "g" || kind === "rank") {
      const range = value.match(/^(\d+)(?:-(\d+))?$/);
      if (!range) throw new Error(`Invalid lane range: ${token}`);
      const lo = +range[1], hi = +(range[2] ?? range[1]);
      return l => (kind === "g" ? [l.index] : l.ranks).some(n => n >= lo && n <= hi);
    }
    if (kind === "node") return l => l.node === value;
    if (kind === "gpu") return l => laneKey(l) === value;
    if (kind === "role") return l => l.roles.includes(value);
    if (kind === "engine") return l => l.engine_addrs.some(a => a.includes(value));
    if (kind === "every" && /^\d+$/.test(value) && +value > 0) return l => l.index % +value === 0;
    throw new Error(`Unknown lane selector: ${token}`);
  });
  return new Set(lanes.filter(l => predicates.some(p => p(l))).map(laneKey));
}
function fleetWindow(params) {
  const data = REPORT.fleet, width = (data.t1 - data.t0) / data.x_buckets;
  if (!(width > 0)) return data;
  const lo = Math.max(0, Math.min(data.x_buckets - 1, Math.floor(((params.t0 ?? data.t0) - data.t0) / width)));
  const hi = Math.min(data.x_buckets, Math.max(lo + 1, Math.ceil(((params.t1 ?? data.t1) - data.t0) / width)));
  const slice = group => Object.fromEntries(Object.entries(group).map(([k, values]) => [k, values.slice(lo, hi)]));
  return {...data, t0: data.t0 + lo * width, t1: data.t0 + hi * width, x_buckets: hi - lo,
    composition: slice(data.composition), band: slice(data.band)};
}
export async function api(path, params = {}) {
  const lanes = selected(params.lanes);
  switch (path) {
    case "/api/meta": return REPORT.meta;
    case "/api/timeline/topology": return REPORT.topology;
    case "/api/timeline/phases": return {phases: REPORT.phases.filter(p => lanes.has(laneKey(p)))};
    case "/api/timeline/gpu": return {lanes: Object.fromEntries(Object.entries(REPORT.gpu).filter(([key]) => lanes.has(key)))};
    case "/api/timeline/gpu_processes": return {processes: REPORT.processes.filter(p => lanes.has(laneKey(p)))};
    case "/api/timeline/engine_series": return {series: REPORT.engines[params.metric] ?? []};
    case "/api/timeline/fleet": return fleetWindow(params);
    case "/api/timeline/bubbles": return {bubbles: REPORT.bubbles};
    case "/api/timeline/data_pool": return {events: REPORT.data_pool};
    case "/api/timeline/outliers": return {outliers: (REPORT.outliers[params.criterion] ?? []).slice(0, params.top_k ?? 16)};
    case "/api/advisory": return REPORT.advisory;
    default: throw new Error(`Not available in the exported timeline: ${path}`);
  }
}
export async function apiBinary() { throw new Error("The exported timeline uses the native fleet overview."); }
