"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
data buffer (see ``fully_async_data_buffer.py``). Rollout production and training
consumption run in parallel, so per-iteration wall time moves from
``rollout_time + train_time`` toward ``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which requires the class-based
rollout API (the default; incompatible with ``MILES_USE_LEGACY_ROLLOUT_V1=1``).

Evaluation targets whatever ``GenerateState`` ``RolloutManager`` passes via
``RolloutFnEvalInput.generate_state`` (see ``miles/rollout/checkpoint_eval.py``
for how the dedicated-fleet state is built). When unset, eval shares the
rollout engines, pausing producer submissions for the duration of the
(blocking) eval.
"""

import asyncio
import logging
import time
from collections import Counter
from dataclasses import replace

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.fully_async_data_buffer import (
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
    DefaultMultiDataBuffer,
    Group,
    add_data_buffer_arguments,
    first_sample,
)
from miles.rollout.filter_hub.common_filters import group_weight_version_stats
from miles.rollout.continuous_sec import METADATA_KEY
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.function_registry import load_function
from miles.utils.perf_monitor import get_monitor, workload_of, workload_targets
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

NO_PROGRESS_WARN_SECS = 30.0


class FullyAsyncRolloutFn(BaseRolloutFn):
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Which finished groups reach training is the
    data buffer's call (see ``fully_async_data_buffer.py``); this class assembles
    what it hands back into a batch.
    """

    add_arguments = staticmethod(add_data_buffer_arguments)

    def __init__(self, input: RolloutFnConstructorInput):
        super().__init__(input)
        self.args = input.args
        self.data_source = input.data_source
        self.curriculum = getattr(input.data_source, "curriculum", None)
        self.state = GenerateState(input.args)
        # default to sample level backfill for fully async rollout
        self._scheduler = make_submission_scheduler(input.args, default="sample")
        assert input.args.async_unused_samples_handler in ("retry", "drop")
        # applied to every group we do not train on; "drop" discards instead of recycling
        self._handle_unused = (
            self._recycle if input.args.async_unused_samples_handler == "retry" else (lambda prompt_group: None)
        )
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._worker: asyncio.Task | None = None
        self._eval_prompt_dataset_cache: dict = {}
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._sync_paused = False
        self._eval_paused = False
        self._generation_tasks: set[asyncio.Task] = set()
        self._curriculum_available = asyncio.Event()
        self._curriculum_available.set()
        self._output: DataBuffer | None = None
        self._monitor = get_monitor(input.args)
        self._pending_groups: dict[asyncio.Task, tuple[float, str]] = {}
        self._last_group_completed: float | None = None
        self._heartbeat: asyncio.Task | None = None
        self._last_heartbeat: float | None = None
        self._workload_targets = dict(workload_targets(self.args))
        self._published_versions: dict[str | None, int] = {}
        self._monitor.register("", self._perf_snapshot)

    def set_weight_version(self, version: int, trainer_model_id: str | None = None) -> None:
        self._published_versions[trainer_model_id] = version
        if self._output is not None:
            self._output.set_weight_version(version, trainer_model_id=trainer_model_id)

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            default_buffer_cls = (
                DefaultMultiDataBuffer if resolve_megatron_config(self.args).is_multi_policy else DefaultDataBuffer
            )
            buffer_cls = load_function(self.args.custom_async_data_buffer_path) or default_buffer_cls
            self._output = buffer_cls(
                DataBufferConstructorInput(
                    args=self.args, unused_handler_fn=self._handle_unused,
                    group_observer=self._observe_curriculum_group if self.curriculum is not None else None,
                )
            )
            for model_id, version in self._published_versions.items():
                self._output.set_weight_version(version, trainer_model_id=model_id)
            self._worker = asyncio.create_task(self._worker_loop())
            if self._monitor.enabled:
                self._last_heartbeat = time.monotonic()
                self._heartbeat = asyncio.create_task(self._monitor_event_loop())
            logger.info("Started fully-async rollout worker")
        with self._monitor.phase("drain_batch", rollout_id=input.rollout_id):
            return await self._drain(input)

    def _perf_snapshot(self) -> dict[str, float]:
        pending = self._pending_groups.copy()
        live = [started for task, started in pending.items() if not task.done()]
        metrics = {
            "in_flight_groups": len(live),
            "completed_pending_groups": len(pending) - len(live),
            "oldest_in_flight_seconds": max((time.monotonic() - t for t, _ in live), default=0.0),
            "group_submission_budget": self._max_in_flight_groups(),
            "sample_budget": self._max_in_flight_groups() * self.args.n_samples_per_prompt,
            "paused": int(not self._producer_resumed.is_set()),
        }
        if hasattr(self._scheduler, "samples_in_flight"):
            metrics["in_flight_samples"] = self._scheduler.samples_in_flight
        if self._last_group_completed is not None:
            metrics["seconds_since_group_completed"] = time.monotonic() - self._last_group_completed
        if self._last_heartbeat is not None:
            metrics["event_loop/heartbeat_age_seconds"] = time.monotonic() - self._last_heartbeat
        for name, target in (self._workload_targets | {"other": self._workload_targets.get("other", 0)}).items():
            metrics[f"workload/{name}/target_fraction"] = target
            metrics[f"workload/{name}/in_flight_group_fraction"] = (
                sum(workload == name for _, workload in live) / len(live) if live else 0
            )
        # Workload gauges live alongside the buffer and recorder, not under producer/.
        return {
            key if key.startswith(("workload/", "event_loop/")) else f"producer/{key}": value
            for key, value in metrics.items()
        }

    async def _monitor_event_loop(self) -> None:
        while True:
            started = time.monotonic()
            await asyncio.sleep(1)
            self._last_heartbeat = time.monotonic()
            self._monitor.update({"event_loop/lag_seconds": max(0, self._last_heartbeat - started - 1)})

    async def dispose(self) -> None:
        if (worker := self._worker) is None:
            return
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_end_worker(worker), worker.get_loop()))
        if self._heartbeat is not None:
            await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(_end_worker(self._heartbeat), self._heartbeat.get_loop())
            )
        if (avacore := getattr(self.state, "avacore", None)) is not None:
            await avacore.dispose()

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        if input.generate_state is not None:
            results = await run_eval_datasets(input.generate_state, self._eval_prompt_dataset_cache)
            return RolloutFnEvalOutput(data=results)

        logger.info("Pausing fully-async producer submissions for shared-engine eval")
        self._eval_paused = True
        self._producer_resumed.clear()
        try:
            results = await run_eval_datasets(self.state, self._eval_prompt_dataset_cache)
        finally:
            self._eval_paused = False
            if not self._sync_paused:
                self._producer_resumed.set()
            logger.info("Resumed fully-async producer submissions after eval")
        return RolloutFnEvalOutput(data=results)

    # -------------------------- producer --------------------------

    async def pause_for_weight_sync(self) -> None:
        """Drain generation, not buffer delivery: a full buffer must not deadlock sync."""
        await self._on_producer_loop(self._pause_for_weight_sync())

    async def _pause_for_weight_sync(self) -> None:
        self._sync_paused = True
        self._producer_resumed.clear()
        # Dispatch and this snapshot execute on the same loop, with no await between
        # the gate check and registering a task. Every group's RM + SEC feedback is
        # inside its generation task; buffer.put is deliberately outside it.
        if self._generation_tasks:
            await asyncio.gather(*tuple(self._generation_tasks))

    async def resume_after_weight_sync(self) -> None:
        await self._on_producer_loop(self._resume_after_weight_sync())

    async def _resume_after_weight_sync(self) -> None:
        self._sync_paused = False
        if not self._eval_paused:
            self._producer_resumed.set()

    async def _on_producer_loop(self, coroutine):
        loop = self._worker.get_loop() if self._worker is not None else asyncio.get_running_loop()
        if loop is asyncio.get_running_loop():
            return await coroutine
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coroutine, loop))

    def _observe_curriculum_group(self, entry, outcome):
        self.curriculum.observe_decision(entry, outcome)
        self._curriculum_available.set()

    def _max_in_flight_groups(self) -> int:
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    def _submit_one_group(self) -> tuple[asyncio.Task, list[Sample]] | None:
        samples = self.data_source.get_samples(1)
        if not samples:
            self._curriculum_available.clear()
            return None
        self._scheduler.on_submit(samples)
        [prompt_group] = samples
        if self.curriculum is not None:
            self.curriculum.dispatch(prompt_group, self._published_versions.get(None))
        task = asyncio.create_task(self._generate_group(prompt_group))
        self._generation_tasks.add(task)
        task.add_done_callback(self._generation_tasks.discard)
        return task, prompt_group

    async def _generate_group(self, prompt_group: list[Sample]) -> DataBufferInput:
        started = time.monotonic()
        sample_done = self._scheduler.sample_done_callback
        if self.curriculum is not None:
            def sample_done():
                self.curriculum.sample_done(prompt_group)
                if self._scheduler.sample_done_callback is not None:
                    self._scheduler.sample_done_callback()

        try:
            result = await generate_and_rm_group(
                self.state,
                prompt_group,
                sampling_params=self.state.sampling_params.copy(),
                evaluation=False,
                sample_done_callback=sample_done,
            )
            if self.curriculum is not None:
                if self.curriculum.complete(prompt_group, result, reward_key=self.args.reward_key):
                    self._monitor.event("curriculum_feedback", details=first_sample(result).metadata[METADATA_KEY])
        except BaseException:
            if self.curriculum is not None:
                self.curriculum.failed(prompt_group)
                self._curriculum_available.set()
            raise
        self._last_group_completed = time.monotonic()
        self._monitor.observe("producer/group_generation", self._last_group_completed - started)
        self._monitor.increment("producer/groups_completed_total")
        return DataBufferInput(prompt_group=prompt_group, group=result)

    async def _worker_loop(self) -> None:
        active: dict[asyncio.Task, list[Sample]] = {}
        try:
            await self._run_worker_loop(active)
        finally:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            self._pending_groups.clear()

    async def _run_worker_loop(self, active: dict[asyncio.Task, list[Sample]]) -> None:
        while True:
            await self._producer_resumed.wait()
            while self._producer_resumed.is_set() and self._scheduler.has_capacity(
                pending_groups=len(active), group_budget=self._max_in_flight_groups()
            ):
                # SEC keeps completed oversampling work, so bound its pending groups
                # as well as the existing sample-level concurrency budget.
                if self.curriculum is not None and len(active) >= self._max_in_flight_groups():
                    break
                submission = self._submit_one_group()
                if submission is None:
                    break
                task, prompt_group = submission
                active[task] = prompt_group
                if self._monitor.enabled:
                    self._pending_groups[task] = (time.monotonic(), workload_of(prompt_group[0], self.args))
                self._monitor.increment("producer/groups_submitted_total")
                if self.curriculum is not None:
                    await asyncio.sleep(0)
            if not active:
                await self._curriculum_available.wait()
                continue
            if self.curriculum is not None:
                # SEC also bounds pending groups. A sample-completion wakeup cannot
                # open a group slot; waiting on that already-set event would spin.
                done, _ = await asyncio.wait(set(active), return_when=asyncio.FIRST_COMPLETED)
            else:
                done, _ = await self._scheduler.wait_for_progress(set(active))
            for task in done:
                entry = self._collect_group_result(task, active.pop(task))
                self._pending_groups.pop(task, None)
                await self._output.put(entry)

    def _collect_group_result(self, task: asyncio.Task, prompt_group: list[Sample]) -> DataBufferInput:
        if not task.cancelled():
            return task.result()

        logger.warning(
            "Rollout group was cancelled; marking samples aborted: indices=%s",
            [sample.index for sample in prompt_group],
        )
        return DataBufferInput(
            prompt_group=prompt_group,
            group=[replace(sample, status=Sample.Status.ABORTED) for sample in prompt_group],
        )

    # -------------------------- consumer --------------------------

    async def _next_group(self, *, current_version: int | None, trainer_model_id: str | None) -> DataBufferInput:
        queue_get = asyncio.create_task(
            self._output.get(current_version=current_version, trainer_model_id=trainer_model_id)
        )
        try:
            while True:
                done, _ = await asyncio.wait(
                    {queue_get, self._worker},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=NO_PROGRESS_WARN_SECS,
                )
                # Checked before the queue: the worker loop never returns normally, so a
                # dead worker fails the step now instead of after its backlog drains.
                if self._worker in done:
                    if self._worker.cancelled():
                        raise RuntimeError("fully-async rollout was disposed while a step waited for groups")
                    self._worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if queue_get in done:
                    return queue_get.result()
                logger.warning(f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s")
        finally:
            if not queue_get.done():
                queue_get.cancel()

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        consumed_prompts = []
        do_print = True
        workload_counts: dict[str, int] = {}
        group_workloads: dict[int, str] = {}
        prefix = "drain" if input.trainer_model_id is None else f"drain/{input.trainer_model_id}"
        self._monitor.update(
            {
                f"{prefix}/rollout_id": input.rollout_id,
                f"{prefix}/target_groups": target_data_size,
                f"{prefix}/collected_groups": 0,
                f"{prefix}/progress_fraction": 0,
            }
        )
        self._monitor.update({f"workload/{name}/collection_progress_fraction": 0 for name in self._workload_targets})

        while len(data) < target_data_size:
            entry = await self._next_group(
                current_version=input.weight_version, trainer_model_id=input.trainer_model_id
            )
            assert len(entry.group) == args.n_samples_per_prompt

            if do_print:
                sample = first_sample(entry.group)
                logger.info(
                    "First rollout sample: text_preview=%s, label=%s, reward_summary=%s",
                    sample_text_preview(sample),
                    str(sample.label)[:100],
                    reward_log_summary(sample.reward),
                )
                do_print = False

            data.append(entry.group)
            consumed_prompts.append(entry.prompt_group)
            workload = workload_of(entry.prompt_group[0], args)
            workload_counts[workload] = workload_counts.get(workload, 0) + 1
            if self._monitor.enabled:
                group_workloads[id(entry.group)] = workload
            self._monitor.update(
                {
                    f"{prefix}/collected_groups": len(data),
                    f"{prefix}/progress_fraction": len(data) / target_data_size,
                }
            )
            if (target := self._workload_targets.get(workload)) is not None:
                self._monitor.update(
                    {
                        f"workload/{workload}/collection_progress_fraction": workload_counts[workload]
                        / (target_data_size * target)
                    }
                )

        sample = first_sample(data[-1])
        logger.info(
            "Finish rollout: text_preview=%s, label=%s, reward_summary=%s",
            sample_text_preview(sample),
            str(sample.label)[:100],
            reward_log_summary(sample.reward),
        )

        data.sort(key=lambda group: first_sample(group).index)

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        if self._monitor.enabled:
            versions = Counter()
            workload_versions: dict[str, Counter] = {}
            mixed = 0
            for group in data:
                stats = group_weight_version_stats(group)
                version = str(stats.oldest_version) if stats.oldest_version is not None else "unknown"
                versions[version] += 1
                workload = group_workloads.get(id(group), workload_of(first_sample(group), args))
                workload_versions.setdefault(workload, Counter())[version] += 1
                mixed += int(stats.oldest_version != stats.newest_version)
            self._monitor.event(
                "batch_ready",
                rollout_id=input.rollout_id,
                details=dict(
                    current_version=input.weight_version,
                    trainer_model_id=input.trainer_model_id,
                    groups=len(data),
                    versions=dict(versions),
                    mixed_groups=mixed,
                    workload_versions={name: dict(counter) for name, counter in workload_versions.items()},
                    groups_before_batch_filter=sum(workload_counts.values()),
                    groups_removed_by_batch_filter=max(0, sum(workload_counts.values()) - len(data)),
                    workload_groups_before_batch_filter=workload_counts,
                ),
            )

        metrics = self._output.get_metrics(input.trainer_model_id)
        if self.curriculum is not None:
            for prompt_group in consumed_prompts:
                self.curriculum.release(prompt_group)
            self._curriculum_available.set()
            metrics.update({f"rollout/curriculum/{key}": value for key, value in self.curriculum.metrics().items()})
        return RolloutFnTrainOutput(samples=data, metrics=metrics)

    def _recycle(self, prompt_group: list[Sample]) -> None:
        self._monitor.increment("producer/recycled_groups_total")
        for sample in prompt_group:
            sample.reset_for_retry()
        self.data_source.add_samples([prompt_group])


async def _end_worker(worker: asyncio.Task) -> None:
    worker.cancel()
    await asyncio.wait({worker})
