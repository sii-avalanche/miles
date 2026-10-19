const RETRIES_503 = 3;
// Derive the mounted run prefix from this module, including reverse-proxy prefixes.
export const APP_BASE = new URL("../", import.meta.url);

async function fetchOk(path, params, { retry503 = true } = {}) {
  const url = new URL(path.replace(/^\//, ""), APP_BASE);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) url.searchParams.set(k, v);
  }
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(url);
    if (res.status === 503 && retry503 && attempt < RETRIES_503) {
      await new Promise((r) => setTimeout(r, 1500));
      continue;
    }
    if (!res.ok) {
      let detail = "";
      try {
        detail = (await res.json()).detail ?? "";
      } catch {
        /* non-json error body */
      }
      throw new Error(`HTTP ${res.status}: ${detail || url.pathname}`);
    }
    return res;
  }
}

export async function api(path, params = {}, options = {}) {
  return (await fetchOk(path, params, options)).json();
}

// framed binary endpoints (/api/timeline/heatmap):
// [4-byte LE header length][header JSON][uint8 matrix, row-major]
export async function apiBinary(path, params = {}) {
  const buf = await (await fetchOk(path, params)).arrayBuffer();
  const headerLen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)));
  return { ...header, values: new Uint8Array(buf, 4 + headerLen) };
}

export async function getMeta() {
  return api("/api/meta");
}


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
