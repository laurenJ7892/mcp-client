"""Append one JSON object per line for a single run."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

from model_client.errors import ClientError, failure_kind

_TOOL_OUTPUT = {"content", "message", "structured_content"}


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
        """Create one JSONL file in `directory`.

        The directory mode is `0700`. The file mode is `0600`.
        """
        try:
            directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o700)
        except OSError as exc:
            raise ClientError(
                "log", f"Could not write the log: {exc.strerror}"
            ) from exc
        stamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S-%f")
        self.path = directory / f"{stamp}.jsonl"
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            os.chmod(self.path, 0o600)
        except OSError as exc:
            raise ClientError(
                "log", f"Could not write the log: {exc.strerror}"
            ) from exc

    def tools_listed(
        self, *, model: str, instructions: str, tools: list[dict[str, Any]]
    ) -> None:
        """Record the model, the instructions, and the tools."""
        self.write(
            "tools_listed",
            model=model,
            instructions=instructions,
            tools=tools,
        )

    def model_response(
        self, round_index: int, content: str | None, tool_calls: list[Any]
    ) -> None:
        """Record one model reply."""
        self.write(
            "model_response",
            round=round_index,
            content=content,
            tool_calls=[_call_record(call) for call in tool_calls],
        )

    def tool_result(self, round_index: int, call: Any, fields: dict[str, Any]) -> None:
        """Record the tool name, `ok`, `code`, and `kind`."""
        kind = failure_kind(fields)
        logged = {
            key: value for key, value in fields.items() if key not in _TOOL_OUTPUT
        }
        if kind is not None:
            logged["kind"] = kind.value
        self.write(
            "tool_result",
            round=round_index,
            name=call.name,
            **logged,
        )

    def approval(self, round_index: int, name: str, *, allowed: bool) -> None:
        """Record a yes or no answer for an approval refusal."""
        self.write("approval", round=round_index, name=name, allowed=allowed)

    def final(self, content: str) -> None:
        """Record the answer returned to the caller."""
        self.write("final", stop="final", content=content)

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


def report(log: Transcript, error: ClientError) -> NoReturn:
    """Write `error` to the log, then raise it.

    When the write fails, raise the original error.
    """
    try:
        log.record(error)
    except ClientError:
        pass
    raise error


def _call_record(call: Any) -> dict[str, Any]:
    """Return the call id and the tool name.

    The record includes `error` when the arguments did not parse.
    """
    record: dict[str, Any] = {"id": call.id, "name": call.name}
    if call.error:
        record["error"] = call.error
    return record
