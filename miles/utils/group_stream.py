"""Small atomic manifests on shared storage; tensors stay in the native object store.

One directory owns one optimizer batch. A failed producer is terminal: consumers
must not commit gradients from the prefix that happened to arrive before failure.
"""

import json
import time
import uuid
from pathlib import Path


def write_manifest(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


class GroupStream:
    def __init__(self, directory, *, timeout):
        self.directory = Path(directory)
        if timeout <= 0:
            raise ValueError("Group wait timeout must be positive")
        self.timeout = timeout

    @classmethod
    def create(cls, parent, *, rollout_id, count, size, timeout):
        if count < 1 or size < 1:
            raise ValueError("A stream needs positive group count and size")
        directory = Path(parent) / f"{rollout_id}-{uuid.uuid4().hex}"
        directory.mkdir(parents=True, exist_ok=False)
        write_manifest(directory / "batch.json", dict(rollout_id=rollout_id, count=count, size=size))
        return cls(directory, timeout=timeout)

    @property
    def batch(self):
        return json.loads((self.directory / "batch.json").read_text())

    def check_failure(self):
        path = self.directory / "failed.json"
        if path.exists():
            raise RuntimeError(f"Group producer failed: {path.read_text()}")

    def ready(self, start):
        self.check_failure()
        rows = []
        for index in range(start, self.batch["count"]):
            path = self.directory / f"group-{index}.json"
            if not path.exists():
                break
            row = json.loads(path.read_text())
            if row["index"] != index or row["rollout_id"] != self.batch["rollout_id"]:
                raise ValueError("Group manifest crosses optimizer batches or reorders groups")
            rows.append(row)
        return rows

    def wait(self, start):
        deadline = time.monotonic() + self.timeout
        while not (rows := self.ready(start)):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"No group {start} published in {self.directory}")
            time.sleep(0.02)
        return rows

    def wait_finished(self):
        deadline = time.monotonic() + self.timeout
        while not (self.directory / "complete.json").exists():
            self.check_failure()
            if time.monotonic() >= deadline:
                raise TimeoutError("Group producer did not acknowledge the complete batch")
            time.sleep(0.02)
        self.check_failure()
        if len(self.ready(0)) != self.batch["count"]:
            raise ValueError("Producer completed without every expected group")

    def fail(self, error):
        write_manifest(self.directory / "failed.json", dict(error=repr(error)))
