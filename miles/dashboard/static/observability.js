import { APP_BASE } from "./api.js";

export function grafanaUrl(meta, range = null, { embed = false } = {}) {
  const path = meta.observability?.grafana_url;
  if (!path) return null;
  const url = new URL(path, APP_BASE);
  if (!["http:", "https:"].includes(url.protocol)) return null;
  if (embed && url.origin !== APP_BASE.origin) return null;
  if (range && range.ready !== false) {
    url.searchParams.set("from", String(Math.round(range.t0 * 1000)));
    url.searchParams.set("to", String(Math.round(range.t1 * 1000)));
  }
  if (embed) url.searchParams.set("kiosk", "1");
  return url.href;
}
