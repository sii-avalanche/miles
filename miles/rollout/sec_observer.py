"""SEC feedback for the existing batch rollout; never schedules model work."""

import logging

logger = logging.getLogger(__name__)


class SECBatchObserver:
    def __init__(self, curriculum, policy_version, reward_key):
        if policy_version is None:
            raise ValueError("SEC feedback requires a published policy version")
        self.curriculum = curriculum
        self.policy_version = policy_version
        self.reward_key = reward_key
        self._groups = {}
        self._observed = set()

    def watch(self, task, prompts):
        # Called synchronously after task creation, before the event loop can run
        # it. Sampling parameters and generation tasks remain owned by rollout.
        self.curriculum.dispatch(prompts, self.policy_version)
        self._groups[task] = prompts
        task.add_done_callback(self._completed)

    def _completed(self, task):
        if task in self._observed:
            return
        self._observed.add(task)
        prompts = self._groups[task]
        try:
            if task.cancelled():
                self.curriculum.failed(prompts)
            else:
                self.curriculum.complete(prompts, task.result(), reward_key=self.reward_key)
        except Exception:
            self.curriculum.failed(prompts)
            logger.exception("SEC feedback failed for a completed rollout group")

    def close(self):
        # Observe already-settled tasks even if their callbacks are still queued.
        # Do not cancel, drain, backfill, reorder or reschedule any model request.
        for task, prompts in self._groups.items():
            if task.done():
                self._completed(task)
            elif task not in self._observed:
                self._observed.add(task)
                self.curriculum.failed(prompts)
            self.curriculum.release(prompts)
