# Durable AvaCore rollout recording

When AvaCore `[record]` is configured, training rollouts await a local snapshot
before returning. A bounded worker pool writes these snapshots to PostgreSQL;
failed writes retry with exponential backoff (1–30 seconds). Retries release
exception frames and decoded records before waiting. Evaluation is not recorded.

Snapshots live in `record.spool_dir`, defaulting to
`$RUN_DIR/rollout_records` (or `./rollout_records` without `RUN_DIR`):

- `pending/`: compressed JSON snapshots waiting for database acknowledgement.
- `committed/`: acknowledged snapshots retained for inspection and recovery.
- `trials/`: persistent trial allocation counters.
- `run.json`: model, run, collection and sampling provenance.

Restart with the same directory and run identity to replay pending snapshots.
Use a durable mount and one directory per rollout actor. A directory belonging to
another model, run or collection is rejected. Keep committed snapshots until
recovery and inspection are no longer needed; they are not automatically deleted.

Existing settings under `[record]` are restored:

| Setting | Default | Purpose |
| --- | --- | --- |
| `spool_dir` | See above | Durable snapshot directory |
| `write_workers` | 16 | Concurrent database deliveries |
| `pool_min_size` / `pool_max_size` | 8 / 64 | Database connection pool bounds |
| `pool_timeout` | 120 seconds | Connection acquisition timeout |
| `shutdown_timeout` | 300 seconds | Time allowed to drain pending deliveries |
| `expected_run_id` | Unset | Validate the existing database run identity |
| `resume` | `true` | Seed local trial counters from existing database records |

On first use with `resume=true`, startup waits for the database to seed trial
counters before generating. Later starts use the persisted seed marker. Use
`resume=false` only for a fresh run with no existing database trials; this skips
counter seeding, not database run lookup. After startup, a database outage does
not block accepting snapshots on disk. Disk write errors still fail the rollout.

Shutdown drains deliveries within the configured timeout. If the database remains
unavailable, pending snapshots stay on disk and shutdown reports an error instead
of declaring recording complete.
