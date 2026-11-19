"""Errors the command prints as `source: message`."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from mcp.shared.exceptions import MCPError

Source = Literal["gateway", "model", "log", "loop"]

# Gateway codes. -32001 is also the MCP request-timeout code. The message
# separates a timeout from `tool_definition_changed`.
BUSY_CODE = -32000
TIMEOUT_CODE = -32001
DEFINITION_CHANGED_CODE = -32001
NOT_PINNED_CODE = -32002
PINS_FILE_CODE = -32003
PIN_GATE_CODE = -32004
NEEDS_APPROVAL_CODE = -32005
READ_ONLY_CODE = -32006
INVALID_PARAMS_CODE = -32602

_POLICY_CODES = {
    DEFINITION_CHANGED_CODE,
    NOT_PINNED_CODE,
    PINS_FILE_CODE,
    PIN_GATE_CODE,
    NEEDS_APPROVAL_CODE,
    READ_ONLY_CODE,
}


class FailureKind(StrEnum):
    """How the client treats one failed tool call.

    `transient` is a timeout or a full MaxConcurrentCalls gate. The transport
    calls the tool once more. The model does not see that failure.
    `arguments` is a bad call. The loop sends it to the model once.
    `policy` is a pin, read-only, or approval refusal. The loop does not
    retry it on its own.
    """

    TRANSIENT = "transient"
    ARGUMENTS = "arguments"
    POLICY = "policy"


def failure_kind(fields: Mapping[str, Any]) -> FailureKind | None:
    """Classify a tool result. Return None when the call succeeded.

    A failure that is none of the three kinds also returns None.
    """
    if fields.get("ok") is not False:
        return None
    code = fields.get("code")
    message = fields.get("message")
    text = message if isinstance(message, str) else ""
    if code == BUSY_CODE and text.startswith("Too many calls are already running"):
        return FailureKind.TRANSIENT
    if code == TIMEOUT_CODE and "timed out" in text:
        return FailureKind.TRANSIENT
    if code in _POLICY_CODES:
        return FailureKind.POLICY
    if code == INVALID_PARAMS_CODE or code is None:
        return FailureKind.ARGUMENTS
    return None


def needs_approval(fields: Mapping[str, Any]) -> bool:
    """Return True when this failed call is waiting for a person to approve it.

    The failure is policy. The loop asks once. It does not retry the call on
    its own, and the transport does not retry it either.
    """
    return fields.get("ok") is False and fields.get("code") == NEEDS_APPROVAL_CODE


def tool_error(fields: Mapping[str, Any], name: str) -> ClientError:
    """A ClientError for a tool result that stops the run."""
    message = fields.get("message")
    detail = message if isinstance(message, str) and message else name
    source: Source = (
        "gateway" if "code" in fields or fields.get("is_error") is True else "loop"
    )
    return ClientError(source, detail)


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
