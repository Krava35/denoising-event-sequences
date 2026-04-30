import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsLogger:
    def __init__(self, output_dir: str, experiment_name: str) -> None:
        self._dir = Path(output_dir) / experiment_name
        self._dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self._dir / "metrics.jsonl"
        logger.info("MetricsLogger initialised at %s", self._dir)

    def _write(self, record: dict) -> None:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self._metrics_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @property
    def metrics_path(self) -> Path:
        return self._metrics_path

    @staticmethod
    def _format_metrics(metrics: dict) -> str:
        parts: list[str] = []
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.6g}")
            else:
                parts.append(f"{key}={value}")
        return " ".join(parts)

    def log_step(self, step: int, metrics: dict) -> None:
        self._write({"step": step, **metrics})
        logger.info("step=%s %s", step, self._format_metrics(metrics))

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        self._write({"epoch": epoch, **metrics})
        logger.info("epoch=%s %s", epoch, self._format_metrics(metrics))

    def log_config(self, config: dict) -> None:
        path = self._dir / "config_snapshot.json"
        with path.open("w") as f:
            json.dump(
                {"saved_at": datetime.now(timezone.utc).isoformat(), **config},
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("Config snapshot saved to %s", path)

    def to_csv(self, output_path: str) -> None:
        records = []
        with self._metrics_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            logger.warning("No metrics to export")
            return

        fieldnames: list[str] = []
        seen: set[str] = set()
        for rec in records:
            for key in rec:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        logger.info("Metrics exported to %s", output_path)


if __name__ == "__main__":
    import os
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")

    with tempfile.TemporaryDirectory() as tmpdir:
        ml = MetricsLogger(output_dir=tmpdir, experiment_name="test_run")

        ml.log_step(step=100, metrics={"loss_total": 0.43, "lr": 3e-4})
        ml.log_step(step=200, metrics={"loss_total": 0.31, "lr": 3e-4})
        ml.log_epoch(epoch=1, metrics={"val_loss": 0.35, "val_f1": 0.72})

        ml.log_config({"model": {"hidden_dim": 256}, "training": {"batch_size": 128}})

        # verify JSONL content
        metrics_path = os.path.join(tmpdir, "test_run", "metrics.jsonl")
        with open(metrics_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 3
        assert lines[0]["step"] == 100
        assert lines[0]["loss_total"] == 0.43
        assert "timestamp" in lines[0]
        assert lines[2]["epoch"] == 1
        assert lines[2]["val_f1"] == 0.72

        # verify config snapshot
        snap_path = os.path.join(tmpdir, "test_run", "config_snapshot.json")
        with open(snap_path) as f:
            snap = json.load(f)
        assert snap["model"]["hidden_dim"] == 256
        assert "saved_at" in snap

        # verify CSV export
        csv_path = os.path.join(tmpdir, "test_run", "metrics.csv")
        ml.to_csv(csv_path)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert rows[0]["loss_total"] == "0.43"
        assert rows[2]["val_f1"] == "0.72"
        # step-only rows must not crash on epoch column (DictWriter fills missing as empty)
        assert rows[0].get("epoch", "") == ""

    print("All checks passed.")
