import { el } from "./app.js";

const R = "runtime/rollout/";
export const ROLLOUT_STAGES = [
  { id: "generate", label: "生成", prefix: R + "sample/generate", color: "#1baf7a" },
  { id: "reward", label: "Reward 评分", prefix: R + "sample/reward", color: "#8b5cc7" },
  { id: "persist", label: "Trace 落盘", prefix: R + "record/persist", color: "#249d87" },
  { id: "database", label: "数据库写入", prefix: R + "record", color: "#437aaa" },
];
export const STAGE_METRICS = ROLLOUT_STAGES.flatMap(stage => stage.id === "database"
  ? [stage.prefix + "/pending_writes", stage.prefix + "/in_flight_writes", stage.prefix + "/oldest_pending_seconds"]
  : [stage.prefix + "/waiting", stage.prefix + "/oldest_wait_seconds"]);

export function stageLabel(stage, info = {}) {
  return stage.id === "reward" && info.kind === "lmaaj" ? "LMaaJ 评分" : stage.label;
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

const count = value => Number.isFinite(value) ? String(Math.round(value)) : "未知";
const duration = value => Number.isFinite(value) ? (value >= 60 ? `${(value / 60).toFixed(1)} 分钟` : `${value.toFixed(1)} 秒`) : "未知";
const PHASE_NAMES = { step: "本轮", wait_rollout: "等待 rollout", train: "训练", update_weights: "同步权重", initial_weight_sync: "首次同步权重", checkpoint: "保存 checkpoint" };
const STRATEGY_NAMES = { one_step_off_policy: "One-step off-policy", fully_async: "Fully async", synchronous: "同步 rollout → train", custom: "自定义 rollout 策略", unknown: "未知（启动信息不足）" };
const STRATEGY_SETTINGS = { num_steps_per_rollout: "每批训练步数", update_weights_interval: "权重发布间隔", max_weight_staleness: "最大版本落后", use_tis: "TIS", partial_rollout: "Partial rollout", colocate: "训练/推理共卡", pause_generation_mode: "暂停生成方式" };

export function strategyDescription(strategy) {
  const descriptions = { one_step_off_policy: "按批收集，下一批生成与当前批训练重叠。", fully_async: "持续后台生成，通过 data buffer 收集训练批次。", synchronous: "先完成 rollout，再执行训练。" };
  return (descriptions[strategy?.kind] || "") + Object.entries(strategy?.settings || {}).filter(([key]) => key in STRATEGY_SETTINGS)
    .map(([key, value]) => `${STRATEGY_SETTINGS[key]}=${typeof value === "boolean" ? (value ? "开启" : "关闭") : value}`).join(" · ");
}

export function createRuntimeStatus() {
  const title = el("p", { class: "muted" });
  const strategyTitle = el("p");
  const strategyNote = el("p", { class: "muted", style: "font-size:12px" });
  const grid = el("div", { class: "statgrid" });
  const note = el("p", { class: "muted", style: "font-size:12px" });
  const root = el("div", { class: "panel" }, [el("h3", {}, ["训练策略与当前执行阶段"]), strategyTitle, strategyNote, title, grid, note]);
  function update(data) {
    if (!data) {
      title.textContent = "暂时无法获取当前阶段，等待下次刷新。";
      strategyTitle.textContent = "训练策略：等待启动信息"; strategyNote.textContent = "";
      grid.replaceChildren(); note.textContent = ""; return;
    }
    const { rollout, driver, reward_info: info = {} } = data;
    const strategy = data.training_strategy;
    strategyTitle.textContent = `训练策略：${STRATEGY_NAMES[strategy?.kind] || STRATEGY_NAMES.unknown}`;
    strategyTitle.title = strategy?.rollout_function_path || "实际 rollout 入口未记录";
    strategyNote.textContent = strategyDescription(strategy);
    const freshness = rollout.observed_at == null ? "未记录 rollout 监控" :
      `${rollout.fresh ? "最新采样" : "采样已过期，当前状态未知"}：${duration(rollout.age_seconds)}前`;
    const phases = driver.fresh ? driver.active_phases.filter(p => p.name !== "step") : [];
    title.textContent = freshness + (phases.length ? ` · Driver：${phases.map(p => `${PHASE_NAMES[p.name] || p.name} ${duration(p.oldest_seconds)}`).join("、")}` : "") +
      (info.model_name ? ` · Judge：${info.model_name}` : "");
    grid.replaceChildren(...ROLLOUT_STAGES.map(spec => {
      const stage = rollout.stages.find(s => s.id === spec.id) || {};
      const busy = stage.active > 0 || stage.pending > 0;
      const state = !rollout.fresh ? "当前未知" : stage.active == null ? "未记录" : busy ? "进行中" : "空闲";
      const details = spec.id === "database" ? `写入中 ${count(stage.active)} · 待写 ${count(stage.pending)}` : `阶段内 ${count(stage.active)} 条`;
      const finished = spec.id === "database" ? "成功写入累计" : "阶段结束累计";
      return el("div", { class: "stat", style: `border-top:3px solid ${rollout.fresh && busy ? spec.color : "#ded6ca"}` }, [
        el("div", { class: "k" }, [stageLabel(spec, info)]),
        el("div", { class: "v" }, [state]), el("div", {}, [details]),
        el("div", { class: "muted" }, [`最长等待 ${duration(stage.oldest_seconds)}`]),
        el("div", { class: "muted" }, [`${finished} ${count(stage.finished_total)}`]),
      ]);
    }));
    note.textContent = "当前状态按最新采样显示，与下方历史时间窗口独立。阶段可以并行；阶段结束累计包含失败和取消，不能直接当作成功评分数。" +
      (info.kind === "lmaaj" ? " LMaaJ 阶段包含等待并发名额、API 请求及重试，阶段内数量并非实际 HTTP 并发数。" : "") +
      (info.concurrency != null ? ` Judge 配置并发上限 ${info.concurrency}。` : "") +
      (info.timeout != null ? ` Judge 单次超时 ${duration(info.timeout)}${info.max_retries != null ? `，最多 ${info.max_retries + 1} 次尝试` : ""}。` : "");
  }
  update(null);
  return { root, update };
}
