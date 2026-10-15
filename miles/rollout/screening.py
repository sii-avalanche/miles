"""Full-dataset binary-reward screening using the standard inference rollout pipeline."""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import (
    InferenceRolloutFn,
    generate_and_rm_group,
)
from miles.utils.types import Sample
from miles.rollout.screening_resume import completed_groups, validate_recording_sources

logger = logging.getLogger(__name__)


class ScreeningStopped(RuntimeError):
    """Repeated system failures stopped the sweep; do not submit more samples."""


class SweepExport:
    """Append durable group decisions; publish a final dataset only after full success."""

    def __init__(self, source, output, n_samples, *, early_stop=False, max_consecutive_errors=8, resume_from=None):
        if max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be positive")
        self.source = Path(source)
        content = self.source.read_bytes()
        self.rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        if not self.rows or n_samples < 2:
            raise ValueError("Screening requires nonempty input and at least two samples per prompt")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        for name in ("kept.partial.jsonl", "groups.jsonl", "failed.jsonl"):
            (self.output / name).touch()
        self.n_samples = n_samples
        self.early_stop = early_stop
        self.max_consecutive_errors = max_consecutive_errors
        self.consecutive_errors = 0
        self.last_error = None
        self.stop_reason = None
        self.started = time.monotonic()
        self.responses_scored = 0
        self.responses_attempted = 0
        self.unrequested_samples = 0
        self.early_stopped_groups = 0
        self.counts = dict(processed=0, kept=0, all_zero=0, all_one=0, failed=0)
        self.seen = set()
        self.sha256 = hashlib.sha256(content).hexdigest()
        self.resume_from = str(Path(resume_from).resolve()) if resume_from else None
        self.inherited_groups = completed_groups(
            self.resume_from, source_sha256=self.sha256, total=len(self.rows), n_samples=n_samples, early_stop=early_stop
        ) if self.resume_from else []
        self._seed_completed()
        self._write_summary(False)

    def _seed_completed(self):
        # These files are new; batch and fsync the initial seed before scheduling work.
        with (self.output / "groups.jsonl").open("a") as groups, (self.output / "kept.partial.jsonl").open("a") as kept:
            for group in self.inherited_groups:
                index, rewards, decision = group["source_index"], group["rewards"], group["decision"]
                groups.write(json.dumps(group) + "\n")
                if decision == "kept":
                    kept.write(json.dumps(self.rows[index], ensure_ascii=False) + "\n")
                self.seen.add(index)
                self.counts[decision] += 1
                self.counts["processed"] += 1
                self.responses_attempted += len(rewards)
                self.responses_scored += len(rewards)
                if self.early_stop and len(rewards) < self.n_samples:
                    self.early_stopped_groups += 1
                    self.unrequested_samples += self.n_samples - len(rewards)
            for stream in (groups, kept):
                stream.flush()
                os.fsync(stream.fileno())

    def _append(self, name, value):
        with (self.output / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def raise_if_stopped(self):
        if self.stop_reason is not None:
            raise ScreeningStopped(self.stop_reason)

    def record(self, index, rewards=None, error=None, *, attempted_samples=None):
        self.raise_if_stopped()
        if index in self.seen or not 0 <= index < len(self.rows):
            raise ValueError(f"Duplicate or out-of-range input index: {index}")
        valid_rewards = rewards is not None and all(r in (0, 1) for r in rewards)
        mixed = valid_rewards and 0 in rewards and 1 in rewards
        if error is None:
            if not valid_rewards or not (
                len(rewards) == self.n_samples
                or (self.early_stop and mixed and 2 <= len(rewards) < self.n_samples)
            ):
                error = "Incomplete group or non-binary reward"
        if error is not None:
            decision = "failed"
            self._append("failed.jsonl", {"source_index": index, "error": str(error), "record": self.rows[index]})
        else:
            decision = "kept" if 0 in rewards and 1 in rewards else ("all_one" if rewards[0] == 1 else "all_zero")
            if decision == "kept":
                self._append("kept.partial.jsonl", self.rows[index])
        details = {}
        if self.early_stop:
            attempted = len(rewards or []) if attempted_samples is None else attempted_samples
            stopped = decision == "kept" and len(rewards) < self.n_samples
            skipped = self.n_samples - attempted if stopped else 0
            self.responses_attempted += attempted
            self.responses_scored += sum(r in (0, 1) for r in (rewards or []))
            self.unrequested_samples += skipped
            self.early_stopped_groups += int(stopped)
            details = {"attempted_samples": attempted, "early_stopped": stopped, "unrequested_samples": skipped}
        self._append("groups.jsonl", {"source_index": index, "rewards": rewards, "decision": decision, **details})
        self.seen.add(index)
        self.counts["processed"] += 1
        self.counts[decision] += 1
        if error is None:
            self.consecutive_errors = 0
            self.last_error = None
        else:
            error = str(error)
            self.consecutive_errors = self.consecutive_errors + 1 if error == self.last_error else 1
            self.last_error = error
            if self.consecutive_errors >= self.max_consecutive_errors:
                self.stop_reason = f"Screening stopped after {self.consecutive_errors} consecutive identical errors: {error}"
                self.checkpoint()
                self.raise_if_stopped()

    def _write_summary(self, complete):
        summary = {
            "source": str(self.source),
            "source_sha256": self.sha256,
            "total": len(self.rows),
            "samples_per_prompt": self.n_samples,
            **self.counts,
            "complete": complete,
            "max_consecutive_errors": self.max_consecutive_errors,
            "consecutive_errors": self.consecutive_errors,
            "last_error": self.last_error,
            "stop_reason": self.stop_reason,
            "resume_from": self.resume_from,
            "resumed_groups": len(self.inherited_groups),
        }
        if self.early_stop:
            elapsed = time.monotonic() - self.started
            summary.update(
                early_stop=True,
                max_samples_per_prompt=self.n_samples,
                responses_attempted=self.responses_attempted,
                responses_scored=self.responses_scored,
                unrequested_samples=self.unrequested_samples,
                early_stopped_groups=self.early_stopped_groups,
                elapsed_seconds=elapsed,
                processed_prompts_per_second=(self.counts["processed"] - len(self.inherited_groups)) / elapsed if elapsed else 0,
            )
        temp = self.output / "summary.tmp"
        temp.write_text(json.dumps(summary, indent=2) + "\n")
        temp.replace(self.output / "summary.json")

    def checkpoint(self):
        self._write_summary(False)
        logger.info("Full dataset screening: %s / %s", self.counts, len(self.rows))

    def finish(self):
        complete = len(self.seen) == len(self.rows) and not self.counts["failed"]
        if complete:
            partial = self.output / "kept.partial.jsonl"
            partial.touch(exist_ok=True)
            partial.rename(self.output / "kept.jsonl")
        self._write_summary(complete)
        if not complete:
            raise RuntimeError(f"Screening incomplete; inspect {self.output / 'summary.json'} and failed.jsonl")


class FullDatasetRollout(InferenceRolloutFn):
    """One rollout call drains the finite dataset; only the last chunk reaches debug logging."""

    early_stop = False

    def __init__(self, input):
        args = input.args
        if not args.debug_rollout_only or args.fully_async or args.num_rollout != 1:
            raise ValueError("FullDatasetRollout requires debug_rollout_only=true, fully_async=false, num_rollout=1")
        if args.rollout_shuffle or args.dynamic_sampling_filter_path or args.data_prefilter_path:
            raise ValueError("Full-pass screening requires unshuffled, unfiltered input")
        if not args.disable_rollout_trim_samples:
            raise ValueError("Disable rollout trimming for full coverage")
        if args.start_rollout_id not in (None, 0) or getattr(args, "requested_load", args.load):
            raise ValueError("Start a fresh screening run; checkpoint resume is not supported")
        super().__init__(input)
        self.export = SweepExport(
            args.prompt_data,
            Path(os.environ["RUN_DIR"]) / "rollout_sweep",
            args.n_samples_per_prompt,
            early_stop=self.early_stop,
            resume_from=os.environ.get("AVATRAIN_SWEEP_RESUME_FROM"),
        )
        # Miles derives a prompt limit from rollout_max_context_len even when
        # none was configured. Verify actual coverage instead of rejecting this
        # implicit limit; a genuinely filtered dataset must still fail.
        if len(self.data_source.dataset) != len(self.export.rows):
            raise ValueError("Loaded dataset differs from the source JSONL")
        self._inherit_task = None

    async def _begin_sweep(self, input):
        if input.rollout_id != 0 or getattr(self, "_sweep_started", False):
            raise ValueError("A full sweep can only run once")
        self._sweep_started = True
        self._inherit_task = None
        if self.export.inherited_groups:
            # Imported lazily because the AvaCore adapter loads model libraries.
            from miles.rollout.avacore_rollout import AvaCoreRollout

            groups = self.export.inherited_groups
            manifests = await asyncio.to_thread(
                validate_recording_sources, groups, collection=Path(self.state.args.prompt_data).stem,
                policy_checkpoint=getattr(self.state.args, "hf_checkpoint", None),
            )
            if not self.state.args.avacore_config.get("record"):
                raise ValueError("Resuming screening requires database rollout recording")
            if getattr(self.state, "avacore", None) is None:
                self.state.avacore = AvaCoreRollout(self.state.args, self.state.args.avacore_config)
            self._inherit_task = asyncio.create_task(
                self.state.avacore.inherit_completed(groups, manifests, self.state.sampling_params)
            )

    def _check_inheritance(self):
        if self._inherit_task is not None and self._inherit_task.done():
            self._inherit_task.result()

    async def _finish_inheritance(self, *, cancel=False):
        if self._inherit_task is None:
            return
        if cancel:
            self._inherit_task.cancel()
            await asyncio.gather(self._inherit_task, return_exceptions=True)
        else:
            await self._inherit_task

    async def _call_train(self, input):
        await self._begin_sweep(input)
        try:
            samples = await self._screen_all()
            await self._finish_inheritance()
            self.export.finish()
            return RolloutFnTrainOutput(samples=samples, metrics=self.export.counts.copy())
        finally:
            await self._finish_inheritance(cancel=True)

    async def _screen_all(self):
        args = self.state.args
        from miles.utils import dumper_utils

        await dumper_utils.configure_sglang(args)
        last_good = None
        total = len(self.export.rows)
        for offset in range(0, total, args.rollout_batch_size):
            last_good = None
            count = min(args.rollout_batch_size, total - offset)
            groups = self.data_source.get_samples(count)
            if len(groups) != count:
                raise ValueError("Data source returned an incomplete chunk")
            for index, group in enumerate(groups, offset):
                if any(s.metadata.get("_index") != index for s in group):
                    raise ValueError("Data source reordered or repeated an input row")

            async def process(index, group):
                self.export.raise_if_stopped()
                self._check_inheritance()
                if index in self.export.seen:
                    return None
                try:
                    result = await generate_and_rm_group(self.state, group, self.state.sampling_params)
                    if len(result) != args.n_samples_per_prompt or any(
                        not isinstance(s, Sample)
                        or s.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED, Sample.Status.REPEATED)
                        for s in result
                    ):
                        raise ValueError("Incomplete, aborted, or failed generation group")
                    rewards = [s.get_reward_value(args) for s in result]
                    if any(r not in (0, 1) for r in rewards):
                        raise ValueError("LMaaJ must return binary rewards")
                except Exception as error:
                    # Infrastructure failures are not evidence of zero reward.
                    self.export.record(index, error=f"{type(error).__name__}: {error}")
                    logger.warning("Screening failed at input row %s: %s", index, error)
                    return None
                self.export.record(index, rewards)
                return result

            tasks = [asyncio.create_task(process(i, group)) for i, group in enumerate(groups, offset)]
            try:
                results = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self.export.checkpoint()
            good = [result for result in results if result is not None]
            if good:
                last_good = good
        # No training occurs. Retain only one chunk of tensors for the framework's
        # final debug logging; every group's decision is already durable on disk.
        return last_good or []
