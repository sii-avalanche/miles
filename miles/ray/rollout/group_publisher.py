"""Publish accepted groups once; keep their native object-store references alive."""

import json
from threading import Lock

from miles.ray.rollout.train_data_conversion import (
    ROLLOUT_DATA_VALUE_SPEC,
    convert_samples_to_train_data,
)
from miles.utils import object_store
from miles.utils.data import RolloutDataPack
from miles.utils.group_stream import GroupStream, write_manifest
from miles.utils.weight_version import assert_samples_weight_version_sane


class GroupPublisher:
    def __init__(self, args, rollout_id):
        self._lock = Lock()
        self._closed = False
        self.args = args
        self.stream = GroupStream.create(
            args.group_packing_directory,
            rollout_id=rollout_id,
            count=args.rollout_batch_size,
            size=args.n_samples_per_prompt,
            timeout=args.group_packing_timeout,
        )
        self.packs = []
        self.sample_indices = set()

    def publish(self, samples):
        with self._lock:
            if self._closed:
                raise RuntimeError("Group publisher is closed")
            self._publish(samples)

    def _publish(self, samples):
        self.stream.check_failure()
        batch = self.stream.batch
        if len(self.packs) >= batch["count"] or len(samples) != batch["size"]:
            raise ValueError("Expected exactly the configured number of complete groups")
        if any(isinstance(sample, list) for sample in samples):
            raise ValueError("Streaming packing does not support nested/compacted samples")
        indices = [sample.index for sample in samples]
        if None in indices or len(set(indices)) != len(indices) or self.sample_indices.intersection(indices):
            raise ValueError("Duplicate or missing sample identity in streamed batch")
        if samples[0].group_index is None or len({sample.group_index for sample in samples}) != 1:
            raise ValueError("A streamed GRPO group must contain one prompt group")
        assert_samples_weight_version_sane(self.args, samples=samples)
        data = convert_samples_to_train_data(
            self.args,
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        if data.get("multimodal_train_inputs") is not None:
            raise ValueError("Initial streaming packing integration supports text-only data")
        if len(data.get("rollout_log_probs", [])) != len(samples):
            raise ValueError("Every streamed sample needs rollout log probabilities")
        for sample in samples:
            if len(sample.rollout_log_probs) != sample.response_length:
                raise ValueError("Rollout log probabilities do not match response length")
        data["group_stream"] = str(self.stream.directory)
        pack = RolloutDataPack(
            data_ref=object_store.get_instance().put(data, value_spec=ROLLOUT_DATA_VALUE_SPEC),
            sample_indices=indices,
            group_stream=str(self.stream.directory),
        )
        index = len(self.packs)
        self.packs.append(pack)
        self.sample_indices.update(indices)
        write_manifest(
            self.stream.directory / f"group-{index}.json",
            dict(
                index=index,
                rollout_id=batch["rollout_id"],
                lengths=[len(s.tokens) for s in samples],
                pack=pack.model_dump(mode="json"),
            ),
        )

    def finish(self, samples):
        self.stream.check_failure()
        if len(self.packs) != self.stream.batch["count"]:
            raise ValueError("Producer returned without publishing all groups")
        indices = [s.index for s in samples]
        if len(indices) != len(self.sample_indices) or set(indices) != self.sample_indices:
            raise ValueError("Late rollout processing changed the already-published batch")
        write_manifest(self.stream.directory / "complete.json", {"samples": len(indices)})

    def release(self):
        with self._lock:
            self._closed = True
            self._release()

    def _release(self):
        # Driver releases consumed batches; disposal also owns failed/prefetched
        # batches which never reached the driver's normal cleanup path.
        if not (self.stream.directory / "released.json").exists():
            for pack in self.packs:
                for ref in pack.data_refs:
                    object_store.get_instance().remove(ref)
            write_manifest(self.stream.directory / "released.json", {})
        self.packs.clear()


def release_group_stream(args, directory):
    stream = GroupStream(directory, timeout=args.group_packing_timeout)
    # The driver already releases the first reference through the native path.
    for path in sorted(stream.directory.glob("group-*.json")):
        if path.name == "group-0.json":
            continue
        pack = RolloutDataPack.model_validate(json.loads(path.read_text())["pack"])
        for ref in pack.data_refs:
            object_store.get_instance().remove(ref)
    write_manifest(stream.directory / "released.json", {})
