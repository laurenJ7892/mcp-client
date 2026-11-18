"""Append one JSON object per line for a single run."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from model_client.errors import ClientError


def log_directory() -> Path:
    """The `logs` directory next to this project's `pyproject.toml`."""
    start = Path(__file__).resolve().parent
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent / "logs"
    raise ClientError("log", "Could not find the model-client project.")


class Transcript:
    """A JSONL file of one client run."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S-%f")
        self.path = directory / f"{stamp}.jsonl"

    def write(self, event: str, **fields: Any) -> None:
        record = {"event": event, **fields}
        line = json.dumps(record, ensure_ascii=False)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            raise ClientError(
                "log", f"Could not write the log: {exc.strerror}"
            ) from exc

    def record(self, exc: ClientError) -> None:
        """Write one `final` error line. A second call for the same error returns without writing."""
        if exc.recorded:
            return
        self.write("final", stop="error", source=exc.source, message=exc.message)
        exc.recorded = True
