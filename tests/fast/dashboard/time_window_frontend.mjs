// node --experimental-vm-modules tests/fast/dashboard/time_window_frontend.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createContext, SourceTextModule, SyntheticModule } from "node:vm";

class Element {
  constructor(tag, attrs = {}, children = []) { this.tag = tag; this.attrs = attrs; this.children = children; this.style = {}; }
  setAttribute(name, value) { this.attrs[name] = String(value); }
}
const context = createContext({ console });
const mod = new SourceTextModule(readFileSync(new URL("../../../miles/dashboard/static/time_window.js", import.meta.url), "utf8"), { context });
await mod.link(() => new SyntheticModule(["el"], function () {
  this.setExport("el", (tag, attrs, children) => new Element(tag, attrs, children));
}, { context }));
await mod.evaluate();
const window = mod.namespace.createTimeWindow({ bounds: [1000, 21000], origin: 1000, maxSpan: 14400 });
const a = mod.namespace.createTimeBar(window, "flow"), b = mod.namespace.createTimeBar(window, "GPU");
const range = () => [window.get().t0, window.get().t1];
assert.deepEqual(range(), [6600, 21000]);
window.setWindow(1000, 8200);
assert.deepEqual(range(), [1000, 8200]);
assert.equal(a.root.attrs["data-time-start"], b.root.attrs["data-time-start"], "bars move together");
window.updateBounds([1000, 22000]);
assert.deepEqual(range(), [1000, 8200], "a historical viewport stays fixed during follow updates");
window.setWindow(14800, 22000);
window.updateBounds([1000, 23000]);
assert.deepEqual(range(), [15800, 23000], "a trailing viewport follows new data without changing its span");
window.setWindow(0, 60000);
assert.deepEqual(range(), [1000, 15400], "clamp oversized windows to both the run and fetch cap");
window.setWindow(24000, 25000);
assert.deepEqual(range(), [22000, 23000], "panning beyond the right edge preserves a bounded viewport");
const track = a.root.children[0];
track.onkeydown({ key: "Home", preventDefault() {} });
assert.deepEqual(range(), [1000, 2000]);
track.onkeydown({ key: "ArrowRight", preventDefault() {} });
assert.deepEqual(range(), [1100, 2100]);
const oldStart = b.root.attrs["data-time-start"];
b.destroy(); window.latest();
assert.equal(b.root.attrs["data-time-start"], oldStart, "removed bars stop receiving updates");
a.destroy();
const empty = mod.namespace.createTimeWindow();
empty.updateBounds([10, 10]);
assert.equal(empty.get().ready, false);
assert.equal(empty.setWindow(NaN, Infinity), false);
console.log("Shared time window passed: synchronized bars, bounded pan, live following, keyboard and cleanup.");
