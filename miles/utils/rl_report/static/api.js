// Offline adapter for the original Miles timeline. No fetch or API credentials.
const events = REPORT.events;
const metrics = REPORT.series;
const times = [...Object.values(metrics).flatMap(s=>s.ts.length?[s.ts[0],s.ts.at(-1)]:[]), ...events.flatMap(e=>[e.t0,e.t1])].filter(Number.isFinite);
const t0 = times.reduce((a,b)=>Math.min(a,b),Infinity);
const t1 = times.reduce((a,b)=>Math.max(a,b),-Infinity);
const gap = Math.max(30, REPORT.interval*3);
const lanes = [], gpu = {}, engines = {};
for (const [key,s] of Object.entries(metrics)) {
  const match = key.match(/\/engines\/(train|inference)\/([^/]+)\/devices\/([^/]+)\/gpu_(\d+)\/gpu_util_pct$/);
  if (match) {
    const [,role,engine,node,n] = match, id = `${node}:${n}`;
    if (gpu[id]) continue;
    lanes.push({node,gpu:+n,role,engine,index:lanes.length});
    const mem = metrics[key.replace(/gpu_util_pct$/, "gpu_memory_gb")];
    const byTime = new Map((mem?.ts ?? []).map((t,i)=>[t,mem.y[i]*1024]));
    gpu[id] = {ts:s.ts,util:s.y,mem_mb:s.ts.map(t=>byTime.get(t) ?? null)};
  }
  const engine = key.match(/\/engines\/inference\/([^/]+)\/(running_requests|generated_tokens_per_s|kv_usage_mean|gpu_util_mean_pct|gpu_memory_max_gb)$/);
  if (engine) (engines[engine[1]] ??= {})[engine[2]] = {addr:engine[1],ts:s.ts,value:s.y};
}
// Old runs only have aggregate engine metrics; label them as aggregates.
if (!Object.keys(engines).length && metrics["runtime/driver/fleet/inference/running_requests"]) {
  const s=metrics["runtime/driver/fleet/inference/running_requests"];
  engines["All inference engines (aggregate)"]={running_requests:{addr:"All inference engines (aggregate)",ts:s.ts,value:s.y}};
}
// Historical W&B snapshots did not retain phase events. Show only observed spans,
// bounded by the sampler gap; never infer unobserved short syncs.
for (const [key,s] of Object.entries(metrics)) {
  const m=key.match(/^runtime\/(driver|rollout)\/phase\/([^/]+)\/waiting$/);
  if (!m || events.some(e=>e.role===m[1] && e.name===m[2])) continue;
  const age=metrics[key.replace(/waiting$/, "oldest_wait_seconds")];
  const ageMap=new Map((age?.ts??[]).map((t,i)=>[t,age.y[i]]));
  s.ts.forEach((t,i)=>{if(s.y[i]>0) events.push({kind:"phase",role:m[1],name:m[2],t0:t-Math.min(gap,ageMap.get(t)??0),t1:t,status:"sampled",rollout_id:null})});
}
const phaseNames={train:"actor_train",critic_train:"actor_train",wait_rollout:"train_wait",collect_batch:"rollout",drain_batch:"rollout",update_weights:"update_weights",initial_weight_sync:"update_weights"};
// Role-level stage spans are not per-rank timings; this is explicitly labelled.
const phases = lanes.flatMap(l=>events.filter(e=>phaseNames[e.name] && e.t1 != null &&
  (l.role==="train" ? e.role==="driver" : e.role==="rollout" || e.name.includes("weight")))
  .map(e=>({...e,node:l.node,gpu:l.gpu,name:phaseNames[e.name],rank:-1})));
export const meta = {mode:"snapshot",start_ts:Number.isFinite(t0)?t0:0,time_range:Number.isFinite(t0)?[t0,Math.max(t0+1,t1)]:null,
  metric_keys:Object.keys(metrics), runtime_interval_s:REPORT.interval,engine_interval_s:REPORT.interval,
  capabilities:{has_timeline:true,has_engine_series:Object.keys(engines).length>0,use_utilization_overview:true,max_window_s:Math.max(1,t1-t0)}};
function selected(spec) {
  if (!spec) return lanes;
  const terms=spec.split(",").map(t=>t.trim());
  return lanes.filter(l=>terms.some(t=>{
    if(t===`gpu:${l.node}:${l.gpu}` || t===`node:${l.node}` || t===`role:${l.role}` || t===`engine:${l.engine}`) return true;
    const m=t.match(/^g:(\d+)(?:-(\d+))?$/);
    if(m) return l.index>=+m[1] && l.index<=+(m[2]??m[1]);
    const every=t.match(/^every:(\d+)$/); return every && +every[1]>0 && l.index%+every[1]===0;
  }));
}
function fleet(p) {
  const a=p.t0??t0,b=p.t1??t1,n=p.x_buckets??600;
  const band={p10:[],p50:[],p90:[],min:[]},composition={};
  const palette=[...new Set(phases.map(p=>p.name))];
  palette.forEach(k=>composition[k]=[]);
  const cursors={};
  for(let i=0;i<n;i++) {
    const t=a+(i+.5)*(b-a)/n,values=[];
    for(const [key,s] of Object.entries(gpu)) {
      let j=cursors[key]??0;
      while(j+1<s.ts.length && s.ts[j+1]<=t) j++;
      cursors[key]=j;
      if(s.ts[j]<=t && t-s.ts[j]<=gap && Number.isFinite(s.util[j])) values.push(s.util[j]);
    }
    values.sort((a,b)=>a-b);
    for(const [k,q] of [["p10",.1],["p50",.5],["p90",.9],["min",0]]) band[k].push(values.length?values[Math.floor((values.length-1)*q)]:null);
    for(const name of palette) {
      const active=new Set(phases.filter(p=>p.name===name && p.t0<=t && t<p.t1).map(p=>`${p.node}:${p.gpu}`));
      composition[name].push(lanes.length?active.size/lanes.length:0);
    }
  }
  return {lanes:lanes.length,t0:a,t1:b,x_buckets:n,palette,composition,band};
}
export async function api(path,p={}) {
  const chosen=selected(p.lanes), ids=new Set(chosen.map(l=>`${l.node}:${l.gpu}`));
  switch(path) {
    case "/api/meta": return meta;
    case "/api/timeline/topology": return {lanes,windows:[{t0,t1,engines:Object.keys(engines).map(addr=>({
      addr,gpus:lanes.filter(l=>l.role==="inference" && l.engine===addr).map(l=>[l.node,l.gpu])
    }))}]};
    case "/api/timeline/bubbles": return {bubbles:[]};
    case "/api/timeline/gpu_processes": return {processes:[]};
    case "/api/advisory": return {mfu:modelUtilization(),advisories:[{level:"info",message:"Snapshot of recorded data. GPU phase strips show role-level orchestration, not per-rank kernel timings. Historical runs may lack exact events or per-device telemetry."}]};
    case "/api/timeline/phases": return {phases:phases.filter(p=>ids.has(`${p.node}:${p.gpu}`))};
    case "/api/timeline/gpu": return {lanes:Object.fromEntries(Object.entries(gpu).filter(([k])=>ids.has(k)))};
    case "/api/runtime/events": return {events};
    case "/api/metrics": return metrics;
    case "/api/timeline/engine_series": {
      const key={sglang_num_running_reqs:"running_requests",sglang_gen_throughput:"generated_tokens_per_s",sglang_token_usage:"kv_usage_mean"}[p.metric];
      return {series:Object.values(engines).filter(e=>e[key]).map(e=>{
        const s=e[key], util=e.gpu_util_mean_pct, memory=e.gpu_memory_max_gb;
        const u=new Map((util?.ts??[]).map((t,i)=>[t,util.value[i]]));
        const m=new Map((memory?.ts??[]).map((t,i)=>[t,memory.value[i]]));
        return {...s,gpu_util:s.ts.map(t=>u.get(t)),gpu_memory:s.ts.map(t=>m.get(t))};
      })};
    }
    case "/api/timeline/fleet": return fleet(p);
    case "/api/timeline/outliers": return {outliers:lanes.map(l=>({...l,mean:gpu[`${l.node}:${l.gpu}`].util.reduce((a,b)=>a+b,0)/gpu[`${l.node}:${l.gpu}`].util.length})).sort((a,b)=>a.mean-b.mean).slice(0,p.top_k??8)};
    default: throw Error(`Unavailable in snapshot: ${path}`);
  }
}
export async function apiBinary() {throw Error("Heatmap is replaced by fleet overview in standalone reports");}

// Only actual observations appear in these cards; absent values stay absent.
const cards = [
  ["Completed rollouts", "runtime/driver/progress/completed_rollouts"],
  ["Target rollouts", "runtime/driver/progress/target_rollouts"],
  ["Queued groups", "runtime/rollout/buffer/queued_groups"],
  ["Pool capacity", "runtime/rollout/buffer/capacity_groups"],
  ["Batch ready groups", "runtime/rollout/batch/ready_groups"],
  ["Batch target groups", "runtime/rollout/batch/target_groups"],
  ["Collection progress", "runtime/rollout/drain/progress_fraction"],
  ["Train GPU memory max (GiB)", "runtime/driver/fleet/train/gpu_memory_max_gb"],
  ["Inference GPU memory max (GiB)", "runtime/driver/fleet/inference/gpu_memory_max_gb"],
  ["Train GPU mean (%)", "runtime/driver/fleet/train/gpu_util_mean_pct"],
  ["Inference GPU mean (%)", "runtime/driver/fleet/inference/gpu_util_mean_pct"],
  ["Train GPU coverage", "runtime/driver/fleet/train/coverage_fraction"],
  ["Inference GPU coverage", "runtime/driver/fleet/inference/coverage_fraction"],
];
const summary = document.getElementById("snapshot-summary");
if(summary) {
  for(const [label,key] of cards) {
    const s=metrics[key]; if(!s?.y.length) continue;
    const card=document.createElement("div");card.className="stat";
    const value=document.createElement("div");value.className="v";value.textContent=Number(s.y.at(-1).toFixed(2));
    const caption=document.createElement("div");caption.className="k";caption.textContent=label;
    card.title="Observed at "+new Date(s.ts.at(-1)*1000).toLocaleString();card.append(value,caption);summary.append(card);
  }
}

function modelUtilization() {
  const s=metrics["perf/actor_train_mfu"],peak=metrics["perf/mfu_peak_tflops"];
  if(!s?.y.length || !peak?.y.length) return null;
  return {latest:s.y.at(-1),mean:s.y.reduce((a,b)=>a+b,0)/s.y.length,steps:s.y.length,peak:peak.y.at(-1)};
}
