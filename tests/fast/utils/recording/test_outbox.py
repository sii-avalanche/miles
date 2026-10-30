import asyncio
import builtins
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from miles.utils.recording.outbox import DurableRolloutOutbox, _clear_error_frames


class OutboxTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_write_replays_after_restart(self):
        failed = asyncio.Event()

        async def unavailable(record):
            failed.set()
            raise ConnectionError("database unavailable")

        with tempfile.TemporaryDirectory() as directory:
            outbox = DurableRolloutOutbox(Path(directory), unavailable, workers=1)
            record = {"query_id": "q", "trial_id": 0, "payload": "trace"}
            path = await outbox.enqueue(record)
            self.assertEqual(outbox._read(path), record)
            await outbox.start()
            await asyncio.wait_for(failed.wait(), 1)
            self.assertFalse(await outbox.close(timeout=0.01))
            self.assertTrue(path.exists())

            received = []

            async def recovered(record):
                received.append(record)

            restarted = DurableRolloutOutbox(Path(directory), recovered, workers=1)
            await restarted.start()
            self.assertTrue(await restarted.close(timeout=1))
            self.assertEqual(received, [record])
            self.assertFalse(path.exists())
            self.assertTrue((restarted.committed / path.name).exists())

    async def test_retry_clears_writer_locals_and_preserves_record(self):
        attempts, errors, received = [], [], []
        failed = asyncio.Event()

        async def write(record):
            attempts.append(1)
            if len(attempts) == 1:
                raise ConnectionError("database unavailable")
            received.append(record)

        def event(kind, error):
            if kind == "failed":
                errors.append(error)
                failed.set()

        with tempfile.TemporaryDirectory() as directory:
            outbox = DurableRolloutOutbox(Path(directory), write, workers=1, retry_initial_seconds=0.1, on_event=event)
            record = {"query_id": "q", "trial_id": 1, "payload": "trace"}
            await outbox.enqueue(record)
            await outbox.start()
            await asyncio.wait_for(failed.wait(), 1)
            self.assertIsNone(errors[0].__traceback__.tb_frame.f_locals["record"])
            writer_frame = errors[0].__traceback__.tb_next.tb_frame
            self.assertEqual(writer_frame.f_locals, {})
            self.assertTrue(await outbox.close(timeout=1))
            self.assertEqual(received, [record])

    async def test_trial_ids_survive_restart_and_seed(self):
        async def write(record):
            pass

        with tempfile.TemporaryDirectory() as directory:
            sample = SimpleNamespace(metadata={"_index": "q"}, epoch=0, group_index=0, index=0)
            outbox = DurableRolloutOutbox(Path(directory), write)
            await outbox.seed_trials([("q", 3)], n=2)
            self.assertEqual(await outbox.allocate_trial(sample, 2), 4)
            restarted = DurableRolloutOutbox(Path(directory), write)
            self.assertEqual(await restarted.allocate_trial(sample, 2), 6)

    async def test_conflicting_snapshot_is_rejected(self):
        async def write(record):
            pass

        with tempfile.TemporaryDirectory() as directory:
            outbox = DurableRolloutOutbox(Path(directory), write)
            record = {"query_id": "q", "trial_id": 0, "payload": "original"}
            path = await outbox.enqueue(record)
            self.assertEqual(await outbox.enqueue(record), path)
            with self.assertRaises(ValueError):
                await outbox.enqueue(record | {"payload": "changed"})
            self.assertEqual(outbox._read(path), record)

    @unittest.skipUnless(hasattr(builtins, "BaseExceptionGroup"), "Exception groups require Python 3.11")
    def test_exception_group_with_cycle_releases_frames(self):
        def fail():
            payload = bytearray(1024)
            raise ValueError(len(payload))

        try:
            fail()
        except ValueError as error:
            frame = error.__traceback__.tb_next.tb_frame
            group = builtins.BaseExceptionGroup("failures", [error, KeyboardInterrupt()])
            error.__cause__ = group
            _clear_error_frames(group)
            self.assertEqual(frame.f_locals, {})
