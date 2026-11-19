"""Run a tool call once more, or hand the failure back, or stop."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from typing import Any

from model_client.errors import (
    ClientError,
    FailureKind,
    failure_kind,
    needs_approval,
    tool_error,
)
from model_client.log import Transcript
from model_client.tools import call_tool

Approve = Callable[[str, dict[str, Any]], bool]


async def carry_out(
    session: Any,
    call: Any,
    *,
    log: Transcript,
    round_index: int,
    approve: Approve,
    repairs: int,
) -> tuple[str, int]:
    """Run one call. Return text for the model and the repair count.

    An approval refusal asks once, then runs the call again or stops. A first
    bad argument is returned so the model can fix it. A success keeps that
    count. A second bad argument, a policy refusal, or any other failure
    raises.
    """
    text, fields = await _perform(session, call)
    log.tool_result(round_index, call, fields)
    if needs_approval(fields):
        text, fields = await _after_approval(
            session,
            call,
            approve=approve,
            log=log,
            round_index=round_index,
        )
    kind = failure_kind(fields)
    if kind is FailureKind.ARGUMENTS and repairs < 1:
        return text, repairs + 1
    if kind is not None or fields.get("ok") is False:
        raise tool_error(fields, call.name)
    return text, repairs


def confirm_approval(name: str, arguments: dict[str, Any]) -> bool:
    """Read yes or no from stdin. Return True only for yes.

    stderr shows the tool name, then `Arguments:`, then `Allow this call? [y/n]`.
    An end of file returns False.
    """
    shown = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    question = (
        f"{name} needs human approval.\nArguments: {shown}\nAllow this call? [y/n] "
    )
    while True:
        print(question, end="", file=sys.stderr, flush=True)
        try:
            answer = input().strip().lower()
        except EOFError:
            print(file=sys.stderr)
            return False
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False


async def _perform(session: Any, call: Any) -> tuple[str, dict[str, Any]]:
    if call.error or call.arguments is None:
        message = call.error or "Tool arguments must be a JSON object."
        return message, {"ok": False, "message": message}
    return await _attempt(session, call.name, call.arguments)


async def _attempt(
    session: Any, name: str, arguments: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Call the gateway. A timeout or a busy gate gets one more call."""
    text, fields = await call_tool(session, name, arguments)
    if failure_kind(fields) is not FailureKind.TRANSIENT:
        return text, fields
    text, fields = await call_tool(session, name, arguments)
    if failure_kind(fields) is FailureKind.TRANSIENT:
        raise tool_error(fields, name)
    return text, {**fields, "retried": True}


async def _after_approval(
    session: Any,
    call: Any,
    *,
    approve: Approve,
    log: Transcript,
    round_index: int,
) -> tuple[str, dict[str, Any]]:
    """Ask once. Run the call again, or stop.

    A no answer raises a loop error. A second approval error raises a
    gateway error.
    """
    accepted = await asyncio.to_thread(approve, call.name, call.arguments) is True
    log.approval(round_index, call.name, allowed=accepted)
    if not accepted:
        raise ClientError("loop", f"Declined {call.name}.")
    text, fields = await _attempt(session, call.name, call.arguments)
    log.tool_result(round_index, call, fields)
    if needs_approval(fields):
        raise tool_error(fields, call.name)
    return text, fields
