"""Errors the command prints as `source: message`."""

from __future__ import annotations

from typing import Literal

from mcp.shared.exceptions import MCPError

Source = Literal["gateway", "model", "log", "loop"]


class ClientError(Exception):
    """A failure the command prints and the log records.

    `source` names the part that failed. `message` is the text after that name.
    `recorded` becomes true after `record` writes a `final` line for this error.
    """

    def __init__(self, source: Source, message: str) -> None:
        self.source: Source = source
        self.message = message
        self.recorded = False
        super().__init__(f"{source}: {message}")


def as_client_error(exc: BaseException) -> ClientError:
    """Return a ClientError for `exc`.

    When `exc` is a ClientError, return it. A task group can wrap the real
    failure. When the group holds one ClientError, return that error. For any
    other failure, return a gateway ClientError.
    """
    if isinstance(exc, ClientError):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        errors = [as_client_error(item) for item in exc.exceptions]
        if len(errors) == 1:
            return errors[0]
        return ClientError(
            "gateway",
            "; ".join(error.message for error in errors),
        )
    if isinstance(exc, MCPError):
        return ClientError("gateway", f"{exc.message} ({exc.code})")
    return ClientError("gateway", str(exc))
