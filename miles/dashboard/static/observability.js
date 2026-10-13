import { APP_BASE } from "./api.js";

export function grafanaUrl(meta, range = null) {
  const path = meta.observability?.grafana_url;
  if (!path) return null;
  const url = new URL(path, APP_BASE);
  if (range) {
    url.searchParams.set("from", String(Math.round(range.t0 * 1000)));
    url.searchParams.set("to", String(Math.round(range.t1 * 1000)));
  }
  return url.href;
}
