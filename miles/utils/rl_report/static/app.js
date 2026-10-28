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
