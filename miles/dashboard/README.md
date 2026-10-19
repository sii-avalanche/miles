# Miles Dashboard

实现已归入 [`miles/monitoring/`](../monitoring/README.md) 的四层目录。
本目录仅保留兼容入口与指向唯一前端资源的链接。

训练监控使用一条主链：采集 → Collector → JSONL → HTTP API → 页面。
Miles 展示训练过程、样本和轨迹；Grafana 展示通用监控与告警，W&B 用于实验对比。

## 启动

统一入口是 `python -m miles.dashboard`，按用途选择子命令：

```bash
# 多任务看板：生产环境使用密码环境变量或密码文件
export AVATRAIN_DASHBOARD_PASSWORD_FILE=/path/to/password
python -m miles.dashboard serve --runs-dir /root/AvaTrain/runs --port 7788

# 单个任务：只读取已有遥测，不需要开启完整 tensor dump
python -m miles.dashboard view --dashboard-dir /path/to/run/logs/dashboard --follow

# 手动关联已有任务；AvaTrain 启动时会自动写入关联
python -m miles.dashboard register j-example /path/to/config/run_20261016_100000

# 本地演示，需要 Miles 源码仓库中的测试数据生成器
python -m miles.dashboard view --demo
```

多任务入口也支持 `--no-auth`，用于本机预览或已有认证的私有入口。
`--root-path` 适配反向代理前缀，`--grafana-upstream` 接入独立监控服务。
各子命令的完整参数见 `python -m miles.dashboard <子命令> --help`。
旧的 `miles.dashboard.serve` / `miles.dashboard.service` 命令已合并到此入口。

## 目录分工

| 位置 | 职责 |
|---|---|
| `__main__.py` | 统一命令行入口 |
| `args.py`、`backend.py`、`hooks.py` | 训练参数、tracking 后端、阶段事件和推理拓扑接入 |
| `collector.py` | 主节点收集器，管理采集器和定期落盘 |
| `gpu_sampler.py`、`sglang_scraper.py` | 获取 GPU 与 SGLang 原始指标 |
| `store.py`、`dump_reader.py` | 读取指标流，以及按需读取可选的训练/rollout dump |
| `catalog.py` | 发现运行目录，读写运行状态和任务关联 |
| `web/api.py`、`web/service.py` | 单运行 API；统一服务的多运行路由、认证、缓存、分析页、Prometheus 导出与 Grafana 代理/进程管理 |
| `web/status.py`、`web/diagnostics.py`、`web/group_history.py` | 当前状态与训练模式、诊断建议、样本组历史 |
| `provisioning/` | Grafana 看板与 Prometheus 告警规则 |
| `static/` | 页面和脚本；`analysis/` 包含公式渲染与第三方 KaTeX 资源 |

## 数据怎么流动

1. `init_tracking()` 根据 `use_miles_dashboard` 创建一个 `DashboardCollector`。
2. 训练进程上报标量与阶段事件；推理控制器把 router 和 engine 地址一起更新给 Collector。
3. Collector 管理每个 GPU 节点的采样器，主动拉取 SGLang 指标，并通过 `MetricStore.flush()` 写文件。
4. 多任务看板读取共享文件；访问某个任务的 API 时才加载该任务的数据，并限制缓存数量。
5. `/metrics` 从同一份遥测导出 Prometheus 数据，Grafana 通过 Prometheus 查询。

采集开关与间隔由训练参数决定：`use_miles_dashboard`、`dashboard_dir`、
`dashboard_gpu_sample_interval`、`dashboard_sglang_scrape_interval`、`dashboard_flush_interval`。
AvaTrain 默认将遥测放在 `runs/<配置>/<运行>/logs/dashboard/`。
完整样本 dump 单独由 `dump_details` 开启；指标采集不依赖它。

代码定位：运行时计数和耗时在 `miles/utils/perf_monitor.py`，训练标量通过
`tracking.log()` 上报。GPU/engine 明细采用小时分区，查询按时间窗口加载，避免每次扫描整个运行。
缺失或过期的遥测显示未知；阶段结束计数包含失败和取消，不等同于成功样本数。

验证与部署说明见 `tests/fast/dashboard/`、`docs/user-guide/performance-monitoring.md`
及 AvaTrain 的 `deploy/dashboard/README.md`。
