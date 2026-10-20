"""Replay durable rollout snapshots into PostgreSQL.

The outbox lives in miles.utils.recording.outbox; these imports preserve existing callers.
"""

import argparse
import asyncio
import json
import logging
import os
import tomllib
from pathlib import Path

from miles.utils.recording.outbox import DurableRolloutOutbox as DurableRolloutOutbox
from miles.utils.recording.outbox import _atomic_json as _atomic_json

logger = logging.getLogger(__name__)


async def replay(path: Path, *, postgres: str, workers: int, timeout: float) -> None:
    # Keep the replay command usable without importing GPU rollout adapters.
    from ava_core.core import Schema
    from ava_core.store.postgres import PostgresBackend
    from miles.rollout.avacore_trace import write_record

    manifest = json.loads((path / "run.json").read_text())
    # Old flattened snapshots need their original tokenizer to split exact turn renderings.
    config_path = path.parent / "config.toml"
    config = tomllib.loads(config_path.read_text()) if config_path.exists() else {}
    legacy_tokenizer = None
    if any("trace_graph" not in DurableRolloutOutbox._read(p) for p in (path / "pending").glob("*.json.gz")):
        from transformers import AutoTokenizer

        legacy_tokenizer = AutoTokenizer.from_pretrained(config["train"]["hf_checkpoint"], local_files_only=True)
    async with PostgresBackend(postgres, min_size=8, max_size=64) as store:
        store.pool.timeout = 120
        async with store.rl_run(**manifest, config=config, schema=Schema(), resume=True) as run:

            async def write(record):
                await write_record(run, record, legacy_tokenizer=legacy_tokenizer)

            failures = 0

            def report(event, error):
                nonlocal failures
                if event == "failed":
                    failures += 1
                    if failures in (1, 10) or failures % 100 == 0:
                        logger.warning("Replay write failed %d times; records retained: %r", failures, error)

            outbox = DurableRolloutOutbox(path, write, workers=workers, on_event=report)
            await outbox.start()
            if not await outbox.close(timeout):
                raise RuntimeError(f"Uncommitted rollout records remain in {outbox.pending}; rerun this command")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay durable AvaCore rollout recordings into PostgreSQL")
    parser.add_argument("path", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    asyncio.run(replay(args.path, postgres=os.environ["POSTGRES"], workers=args.workers, timeout=args.timeout))


if __name__ == "__main__":
    main()
