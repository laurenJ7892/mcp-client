"""List gateway tools, ask the model, and run the calls it makes."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any, NoReturn

from model_client.errors import ClientError, as_client_error
from model_client.gateway import (
    call_tool,
    function_tools,
    gateway_directory,
    list_tools,
    open_gateway,
    tool_records,
)
from model_client.log import Transcript
from model_client.model import MODEL, ModelReply, ask, build_client

MAX_TRIES = 4
SYSTEM = (
    "Answer by calling the listed tools. Send the user's text as the tool argument."
)


async def run(prompt: str, log: Transcript) -> str:
    """Run one prompt through the live gateway and the live model."""
    held: ClientError | None = None
    try:
        directory = gateway_directory()
        client = build_client()
    except ClientError as exc:
        held = exc
    except Exception as exc:  # noqa: BLE001
        held = as_client_error(exc)
    else:

        def complete(
            messages: list[dict[str, Any]], tools: list[dict[str, Any]]
        ) -> ModelReply:
            return ask(client, messages, tools)

        try:
            tries = max_tries()
            async with open_gateway(directory) as session:
                try:
                    return await converse(
                        prompt,
                        session=session,
                        complete=complete,
                        log=log,
                        max_tries=tries,
                    )
                except ClientError as exc:
                    held = exc
        except ClientError as exc:
            held = exc
        except Exception as exc:  # noqa: BLE001
            held = as_client_error(exc)
    if held is None:
        held = ClientError("loop", "The loop ended without a final answer.")
    report(log, held)


async def converse(
    prompt: str,
    *,
    session: Any,
    complete: Callable[[list[dict[str, Any]], list[dict[str, Any]]], ModelReply],
    log: Transcript,
    max_tries: int = MAX_TRIES,
) -> str:
    """Offer `session`'s tools to the model and carry out each call."""
    try:
        return await _converse(
            prompt,
            session=session,
            complete=complete,
            log=log,
            max_tries=max_tries,
        )
    except ClientError as exc:
        report(log, exc)


async def _converse(
    prompt: str,
    *,
    session: Any,
    complete: Callable[[list[dict[str, Any]], list[dict[str, Any]]], ModelReply],
    log: Transcript,
    max_tries: int,
) -> str:
    tools = await list_tools(session)
    log.write(
        "tools_listed",
        model=MODEL,
        prompt=prompt,
        instructions=SYSTEM,
        tools=tool_records(tools),
    )
    functions = function_tools(tools)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    failures: list[str] = []
    gateway_refused = False
    for round_index in range(1, max_tries + 1):
        reply = await asyncio.to_thread(complete, messages, functions)
        log.write(
            "model_response",
            round=round_index,
            content=reply.content,
            tool_calls=[_call_record(call) for call in reply.tool_calls],
        )
        messages.append(reply.message())
        if not reply.tool_calls:
            if failures:
                raise _failure_error(failures, gateway_refused)
            text = reply.content or ""
            log.write("final", stop="final", content=text)
            return text
        if round_index == max_tries:
            if failures:
                raise _failure_error(failures, gateway_refused)
            pending = ", ".join(call.name for call in reply.tool_calls)
            raise ClientError(
                "loop",
                f"Stopped after {max_tries} tries with {pending} still pending.",
            )
        for call in reply.tool_calls:
            text, fields = await _perform(session, call)
            log.write(
                "tool_result",
                round=round_index,
                name=call.name,
                arguments=(
                    call.arguments if call.arguments is not None else call.raw_arguments
                ),
                **fields,
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
            if fields.get("ok") is False:
                message = fields.get("message")
                failures.append(
                    message if isinstance(message, str) and message else call.name
                )
                if "code" in fields or fields.get("is_error") is True:
                    gateway_refused = True
    raise ClientError("loop", "The loop ended without a final answer.")


def max_tries() -> int:
    """Return the try limit for this run.

    Read `TRIES` from the environment. Use 4 when `TRIES` is unset.
    """
    raw = os.environ.get("TRIES")
    if raw is None or raw.strip() == "":
        return MAX_TRIES
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1 or value > MAX_TRIES:
        raise ClientError("loop", "TRIES must be an integer from 1 to 4.")
    return value


def _failure_error(failures: list[str], gateway_refused: bool) -> ClientError:
    source = "gateway" if gateway_refused else "loop"
    return ClientError(source, "; ".join(failures))


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
    arguments = call.arguments if call.arguments is not None else call.raw_arguments
    record: dict[str, Any] = {"id": call.id, "name": call.name, "arguments": arguments}
    if call.error:
        record["error"] = call.error
    return record


async def _perform(session: Any, call: Any) -> tuple[str, dict[str, Any]]:
    if call.error or call.arguments is None:
        message = call.error or "Tool arguments must be a JSON object."
        return message, {"ok": False, "message": message}
    return await call_tool(session, call.name, call.arguments)
