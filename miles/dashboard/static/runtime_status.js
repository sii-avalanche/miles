import { el } from "./app.js";

const R = "runtime/rollout/";
export const ROLLOUT_STAGES = [
  { id: "generate", label: "Generation", prefix: R + "sample/generate", color: "#1baf7a" },
  { id: "reward", label: "Reward scoring", prefix: R + "sample/reward", color: "#8b5cc7" },
  { id: "persist", label: "Trace persistence", prefix: R + "record/persist", color: "#249d87" },
  { id: "database", label: "Database writes", prefix: R + "record", color: "#437aaa" },
];
export const STAGE_METRICS = ROLLOUT_STAGES.flatMap(stage => stage.id === "database"
  ? [stage.prefix + "/pending_writes", stage.prefix + "/in_flight_writes", stage.prefix + "/oldest_pending_seconds"]
  : [stage.prefix + "/waiting", stage.prefix + "/oldest_wait_seconds"]);

export function stageLabel(stage, info = {}) {
  return stage.id === "reward" && info.kind === "lmaaj" ? "LMaaJ scoring" : stage.label;
}

export function stageSegments(series, t0, t1, maxGapSeconds, now = Date.now() / 1000) {
  const samples = (series?.ts ?? []).map((ts, i) => ({ ts, value: series.y[i] }))
    .filter(p => Number.isFinite(p.ts)).sort((a, b) => a.ts - b.ts);
  return samples.flatMap((sample, i) => {
    if (!Number.isFinite(sample.value) || sample.value < 0) return [];
    const start = Math.max(sample.ts, t0);
    const end = Math.min(samples[i + 1]?.ts ?? sample.ts, sample.ts + maxGapSeconds, t1, now);
    return end > start ? [{ ...sample, start, end }] : [];
  });
}

const duration = value => Number.isFinite(value) ? (value >= 60 ? `${(value / 60).toFixed(1)} min` : `${value.toFixed(1)} s`) : "unknown";
const PHASE_NAMES = { step: "Step", wait_rollout: "Waiting for rollout", train: "Training", update_weights: "Weight sync", initial_weight_sync: "Initial weight sync", checkpoint: "Checkpoint" };
const STRATEGY_NAMES = { one_step_off_policy: "One-step off-policy", fully_async: "Fully async", synchronous: "Synchronous rollout → train", custom: "Custom rollout strategy", unknown: "Unknown (launch information missing)" };
const STRATEGY_SETTINGS = { num_steps_per_rollout: "Train steps per batch", update_weights_interval: "Weight publish interval", max_weight_staleness: "Maximum version lag", use_tis: "TIS", partial_rollout: "Partial rollout", colocate: "Shared training/inference GPUs", pause_generation_mode: "Generation pause mode" };

export function strategyDescription(strategy) {
  const descriptions = { one_step_off_policy: "Collect each batch while the previous batch trains.", fully_async: "Generate continuously in the background and collect training batches from the data buffer.", synchronous: "Complete rollout before training." };
  const settings = Object.entries(strategy?.settings || {}).filter(([key]) => key in STRATEGY_SETTINGS)
    .map(([key, value]) => `${STRATEGY_SETTINGS[key]}=${typeof value === "boolean" ? (value ? "on" : "off") : value}`).join(" · ");
  return [descriptions[strategy?.kind], settings].filter(Boolean).join(" ");
}

export function createRuntimeStatus() {
  const title = el("p", { class: "muted" });
  const strategyTitle = el("p");
  const strategyNote = el("p", { class: "muted", style: "font-size:12px" });
  const root = el("div", { class: "panel" }, [el("h3", {}, ["Training strategy and current phase"]), strategyTitle, strategyNote, title]);
  function update(data) {
    if (!data) {
      title.textContent = "Current phase unavailable. Waiting for the next refresh.";
      strategyTitle.textContent = "Training strategy: waiting for launch information";
      strategyNote.textContent = "";
      return;
    }
    const { driver, reward_info: info = {} } = data;
    const strategy = data.training_strategy;
    strategyTitle.textContent = `Training strategy: ${STRATEGY_NAMES[strategy?.kind] || STRATEGY_NAMES.unknown}`;
    strategyTitle.title = strategy?.rollout_function_path || "Rollout entrypoint not recorded";
    strategyNote.textContent = strategyDescription(strategy);
    const phases = driver.fresh ? driver.active_phases.filter(p => p.name !== "step") : [];
    title.textContent = (driver.fresh
      ? phases.length ? `Driver: ${phases.map(p => `${PHASE_NAMES[p.name] || p.name} ${duration(p.oldest_seconds)}`).join(", ")}` : "Driver: no active phase observed"
      : "Driver sample is stale; current phase unknown") + (info.model_name ? ` · Judge: ${info.model_name}` : "");
  }
  update(null);
  return { root, update };
}
