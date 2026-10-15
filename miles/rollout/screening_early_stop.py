"""Screen many prompts concurrently, with at most one outstanding sample per prompt."""

import asyncio
import logging
import uuid

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.generate_utils.generate_endpoint_utils import policy_uses_routing_key
from miles.rollout.inference_rollout.inference_rollout_common import generate_and_rm
from miles.utils.types import Sample
from miles.rollout.screening import FullDatasetRollout, ScreeningStopped

logger = logging.getLogger(__name__)


class EarlyStopDatasetRollout(FullDatasetRollout):
    early_stop = True

    def __init__(self, input):
        if input.args.group_rm:
            raise ValueError("Early stopping requires per-response rewards, not group_rm")
        if input.args.rollout_batch_size < 1:
            raise ValueError("rollout_batch_size must be positive")
        super().__init__(input)

    async def _screen_group(self, index, group):
        args = self.state.args
        rewards, completed = [], []
        attempted = 0
        try:
            for position, sample in enumerate(group):
                self.export.raise_if_stopped()
                sampling_params = self.state.sampling_params.copy()
                if getattr(args, "sglang_enable_deterministic_inference", False):
                    sampling_params["sampling_seed"] = args.rollout_seed + position
                if policy_uses_routing_key(args) and sample.routing_key is None:
                    sample.routing_key = str(uuid.uuid4())
                attempted += 1
                result = await generate_and_rm(self.state, sample, sampling_params)
                if not isinstance(result, Sample) or result.status not in (
                    Sample.Status.COMPLETED, Sample.Status.TRUNCATED, Sample.Status.REPEATED
                ):
                    raise ValueError("Incomplete, aborted, or failed generation")
                reward = result.get_reward_value(args)
                if reward not in (0, 1):
                    raise ValueError("LMaaJ must return binary rewards")
                rewards.append(reward)
                completed.append(result)
                if 0 in rewards and 1 in rewards:
                    # No sibling request is in flight: the unused samples were
                    # never submitted, so savings do not depend on remote abort.
                    break
        except ScreeningStopped:
            raise
        except Exception as error:
            self.export.record(
                index, rewards, error=f"{type(error).__name__}: {error}", attempted_samples=attempted
            )
            logger.warning("Screening failed at input row %s: %s", index, error)
            return None
        self.export.record(index, rewards, attempted_samples=attempted)
        return completed

    async def _call_train(self, input):
        await self._begin_sweep(input)
        from miles.utils import dumper_utils

        args = self.state.args
        try:
            await dumper_utils.configure_sglang(args)
        except BaseException:
            await self._finish_inheritance(cancel=True)
            raise
        indices = iter(range(len(self.export.rows)))
        last_good = None

        async def worker():
            nonlocal last_good
            for index in indices:
                self.export.raise_if_stopped()
                self._check_inheritance()
                # There is no await between claiming a row and getting its group;
                # all workers share this event loop, so the source never wraps or
                # skips a row. Finished prompts immediately make room for new ones.
                groups = self.data_source.get_samples(1)
                if len(groups) != 1 or len(groups[0]) != args.n_samples_per_prompt:
                    raise ValueError("Data source returned an incomplete group")
                group = groups[0]
                if any(s.metadata.get("_index") != index for s in group):
                    raise ValueError("Data source reordered or repeated an input row")
                if index in self.export.seen:
                    continue
                result = await self._screen_group(index, group)
                if result:
                    last_good = result
                if self.export.counts["processed"] % 64 == 0:
                    self.export.checkpoint()

        tasks = [asyncio.create_task(worker()) for _ in range(min(args.rollout_batch_size, len(self.export.rows)))]
        try:
            await asyncio.gather(*tasks)
            await self._finish_inheritance()
        finally:
            for task in tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._finish_inheritance(cancel=True)
            self.export.checkpoint()
        self.export.finish()
        # The driver only logs this representative group in debug_rollout_only.
        # All decisions and actual sample counts are in the export, never padded.
        return RolloutFnTrainOutput(
            samples=[last_good] if last_good else [],
            metrics={
                **self.export.counts,
                "screening/responses_scored": self.export.responses_scored,
                "screening/unrequested_samples": self.export.unrequested_samples,
                "screening/early_stopped_groups": self.export.early_stopped_groups,
            },
        )
