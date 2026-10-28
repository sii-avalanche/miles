"""Background snapshot publisher. Only the driver writes the shared run file."""

import logging
import gzip
import json
import hashlib
import time
from collections import deque
from pathlib import Path

from miles.utils.rl_report import render_report, report_metrics, report_url

logger = logging.getLogger(__name__)


class ReportPublisher:
    def __init__(self, *, interval):
        self.interval = interval
        self.records = deque(maxlen=20000)
        self.last_upload = 0.0
        self.dropped = 0
        self.media_path = None

    def publish(self, records, *, force=False):
        # W&B is optional; imported only in the background W&B monitoring path.
        import wandb

        self.dropped += max(0, len(self.records) + len(records) - self.records.maxlen)
        self.records.extend(records)
        if wandb.run is None:
            return
        archive = Path(wandb.run.dir) / "rl-telemetry.jsonl.gz"
        with gzip.open(archive, "at", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        if not self.last_upload:
            wandb.run.save(str(archive), base_path=wandb.run.dir, policy="live")
        if not force and time.monotonic() - self.last_upload < 60:
            return
        title = wandb.run.name or wandb.run.id
        if self.dropped:
            title += f" (recent window; {self.dropped} older records omitted)"
        path = Path(wandb.run.dir) / "rl-report.html"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            render_report(list(self.records), title=title, run_url=wandb.run.url or "", interval=self.interval)
        )
        temporary.replace(path)
        # The stable file path is the run entry point. Html metadata allows the
        # single-run media viewer to render it without a separate dashboard service.
        wandb.run.save(str(path), base_path=wandb.run.dir, policy="live")
        if self.media_path is None:
            media = wandb.Html(str(path), inject=False)
            # One native media record registers the key with W&B's viewer. All
            # later snapshots replace this same run file, avoiding O(n²) storage.
            wandb.run.log({"rl_report": media})
            self.media_path = media.to_json(wandb.run)["path"]
        media_file = Path(wandb.run.dir) / self.media_path
        media_file.write_bytes(path.read_bytes())
        wandb.run.save(str(media_file), base_path=wandb.run.dir, policy="live")
        wandb.run.summary["rl_report"] = {
            "_type": "html-file",
            "path": self.media_path,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        if wandb.run.url:
            url = report_url(wandb.run.url)
            wandb.run.summary["rl_report_url"] = url
            if "[RL timeline]" not in (wandb.run.notes or ""):
                wandb.run.notes = (wandb.run.notes or "") + f"\n\n[RL timeline]({url})"
        self.last_upload = time.monotonic()


def report_record(metrics):
    return {"kind": "metrics", "ts": time.time(), "metrics": report_metrics(metrics)}
