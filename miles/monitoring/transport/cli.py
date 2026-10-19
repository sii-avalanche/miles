"""Serve multiple runs, view one run, or associate a run with a job ID."""

import argparse
import os
import tempfile
from pathlib import Path


def make_demo_dir(target: Path) -> Path:
    """Generate a local example using the repository's test fixtures."""
    try:
        from tests.fast.dashboard.dummy_dump import dump_dummy_run
        from tests.fast.dashboard.dummy_telemetry import dump_dummy_telemetry
    except ImportError as error:
        raise SystemExit("--demo requires the Miles repository's tests directory") from error
    dump_dummy_run(target, steps=3, num_prompts=8, n_samples_per_prompt=4, max_response_len=48)
    dump_dummy_telemetry(target, steps=3, samples_per_step=32)
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m miles.dashboard", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve a directory of runs with authentication")
    serve.add_argument("--runs-dir", type=Path, default=Path("runs"))
    serve.add_argument("--cache-dir", type=Path, default=Path("/tmp/avatrain-dashboard-cache"))
    serve.add_argument("--username", default=os.getenv("AVATRAIN_DASHBOARD_USERNAME", "admin"))
    serve.add_argument("--max-runs", type=int, default=3)
    serve.add_argument("--root-path", default="", help="External URL prefix stripped by a reverse proxy")
    serve.add_argument("--no-auth", action="store_true", help="Disable authentication for a private entrypoint")
    serve.add_argument("--monitoring", action="store_true", default=os.getenv("AVATRAIN_MONITORING") == "1")
    serve.add_argument("--monitoring-dir", type=Path, default=os.getenv("AVATRAIN_MONITORING_DIR"))
    commands.add_parser("init-db", help="Initialize the shared dashboard schema using AVATRAIN_DASHBOARD_POSTGRES")
    sync = commands.add_parser("sync", help="Publish or resume publication of an existing run's telemetry")
    sync.add_argument("run_dir", type=Path)
    sync.add_argument("--follow", action="store_true", help="Retry publication every five seconds")
    view = commands.add_parser("view", help="View one run's telemetry and optional tensor dumps")
    view.add_argument("--dump-details", type=Path)
    view.add_argument("--dashboard-dir", type=Path)
    view.add_argument("--cache-dir", type=Path)
    view.add_argument("--follow", action="store_true")
    view.add_argument("--tensor-lru", type=int, default=2)
    view.add_argument("--demo", action="store_true", help="Generate demo data from repository fixtures")
    view.add_argument("--use-utilization-overview", action="store_true")
    view.add_argument("--low-mfu-threshold", type=float, help="Override the diagnostic rule's default MFU threshold")
    for command in (serve, view):
        command.add_argument("--host", default="0.0.0.0")
        command.add_argument("--port", type=int, default=7788)
    register = commands.add_parser("register", help="Associate an existing run with a job ID")
    register.add_argument("job_id")
    register.add_argument("run_dir", type=Path)
    return parser


def _view_app(args, parser):
    # Tensor readers and torch are only needed when viewing an individual run.
    from miles.monitoring.aggregation.diagnostics import DEFAULT_LOW_MFU
    from miles.monitoring.transport.api import make_app
    from miles.monitoring.transport.dump_reader import DumpReader
    from miles.monitoring.transport.service import follow_updates
    from miles.monitoring.transport.store import MetricStore

    if args.demo:
        dump_dir = make_demo_dir(Path(tempfile.mkdtemp(prefix="miles_dashboard_demo_")))
    elif args.dump_details or args.dashboard_dir:
        dump_dir = args.dump_details or args.dashboard_dir.parent
    else:
        parser.error("view requires --dashboard-dir, --dump-details or --demo")
    dashboard_dir = args.dashboard_dir or dump_dir / "dashboard"
    if not dump_dir.is_dir() or not dashboard_dir.is_dir():
        parser.error("The dump and dashboard directories must exist")
    store = MetricStore.load(dashboard_dir)
    app = make_app(
        store,
        DumpReader(dump_dir, cache_dir=args.cache_dir, tensor_lru=args.tensor_lru),
        follow=args.follow,
        use_utilization_overview=args.use_utilization_overview,
        low_mfu=DEFAULT_LOW_MFU if args.low_mfu_threshold is None else args.low_mfu_threshold,
    )
    if args.follow:
        app.router.lifespan_context = lambda app: follow_updates(store.follow)
    return app


def _service_app(args, parser):
    from miles.monitoring.transport.service import make_service

    password = None
    if not args.no_auth:
        path = os.getenv("AVATRAIN_DASHBOARD_PASSWORD_FILE")
        password = Path(path).read_text().rstrip("\r\n") if path else os.getenv("AVATRAIN_DASHBOARD_PASSWORD")
        if not password:
            parser.error("Set AVATRAIN_DASHBOARD_PASSWORD[_FILE] or use --no-auth for a private entrypoint")
    return make_service(
        args.runs_dir,
        cache_dir=args.cache_dir,
        password=password,
        username=args.username,
        max_runs=args.max_runs,
        monitoring=args.monitoring,
        monitoring_dir=args.monitoring_dir,
        port=args.port,
        root_path=args.root_path,
        postgres=os.getenv("AVATRAIN_DASHBOARD_POSTGRES"),
    )


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command in ("init-db", "sync"):
        from miles.monitoring.transport.postgres import DashboardDatabase
        from miles.monitoring.transport.publisher import TelemetryPublisher

        dsn = os.getenv("AVATRAIN_DASHBOARD_POSTGRES")
        if not dsn:
            parser.error("Set AVATRAIN_DASHBOARD_POSTGRES")
        database = DashboardDatabase(dsn)
        if args.command == "init-db":
            database.initialize()
            print("Shared dashboard schema initialized")
        else:
            publisher = TelemetryPublisher(database, args.run_dir.resolve())
            if args.follow:
                import time

                publisher.start()
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    publisher.stop()
            else:
                print(f"Published {publisher.sync()} bytes")
        return
    if args.command == "register":
        from miles.monitoring.transport.catalog import write_manifest

        if not args.run_dir.is_dir():
            parser.error("run-dir must exist")
        write_manifest(args.run_dir, job_id=args.job_id)
        print(f"Registered {args.job_id}: {args.run_dir.name}")
        return
    import uvicorn

    app = _service_app(args, parser) if args.command == "serve" else _view_app(args, parser)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        root_path=getattr(args, "root_path", ""),
        access_log=args.command == "view",
    )


if __name__ == "__main__":
    main()
