"""Ask the model in turns and carry out the calls it makes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from model_client.errors import ClientError, as_client_error
from model_client.gateway import gateway_directory, open_gateway
from model_client.log import Transcript, report
from model_client.model import MODEL, ModelReply, ask, build_client
from model_client.retry import Approve, carry_out, confirm_approval
from model_client.tools import function_tools, list_tools, tool_records

MAX_TRIES = 4
SYSTEM = (
    "Answer by calling the listed tools. Send the user's text as the tool argument."
)
Complete = Callable[[list[dict[str, Any]], list[dict[str, Any]]], ModelReply]


async def run(prompt: str, log: Transcript, api_key: str, tries: int) -> str:
    """Run one prompt through the live gateway and the live model."""
    held: ClientError | None = None
    try:
        directory = gateway_directory()
        client = build_client(api_key)
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
    complete: Complete,
    log: Transcript,
    max_tries: int = MAX_TRIES,
    approve: Approve | None = None,
) -> str:
    """Offer `session`'s tools to the model and carry out each call."""
    try:
        return await _converse(
            prompt,
            session=session,
            complete=complete,
            log=log,
            max_tries=max_tries,
            approve=confirm_approval if approve is None else approve,
        )
    except ClientError as exc:
        report(log, exc)
    except Exception as exc:  # noqa: BLE001
        report(log, as_client_error(exc))


async def _converse(
    prompt: str,
    *,
    session: Any,
    complete: Complete,
    log: Transcript,
    max_tries: int,
    approve: Approve,
) -> str:
    messages, functions = await _prepare(session, prompt, log)
    repairs = 0
    for round_index in range(1, max_tries + 1):
        reply = await _ask(complete, messages, functions, log, round_index)
        if not reply.tool_calls:
            return _answer(log, reply)
        _stop_on_last_try(round_index, max_tries, reply)
        repairs = await _use_tools(
            session,
            reply,
            messages,
            log=log,
            round_index=round_index,
            approve=approve,
            repairs=repairs,
        )
    raise ClientError("loop", "The loop ended without a final answer.")


async def _prepare(
    session: Any, prompt: str, log: Transcript
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """List the gateway tools and build the first messages."""
    tools = await list_tools(session)
    log.tools_listed(
        model=MODEL,
        instructions=SYSTEM,
        tools=tool_records(tools),
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    return messages, function_tools(tools)


async def _ask(
    complete: Complete,
    messages: list[dict[str, Any]],
    functions: list[dict[str, Any]],
    log: Transcript,
    round_index: int,
) -> ModelReply:
    """Record one model reply and keep it on the message list."""
    reply = await asyncio.to_thread(complete, messages, functions)
    log.model_response(round_index, reply.content, reply.tool_calls)
    messages.append(reply.message())
    return reply


def _answer(log: Transcript, reply: ModelReply) -> str:
    """Record the model's text and return it."""
    text = reply.content or ""
    log.final(text)
    return text


def _stop_on_last_try(round_index: int, max_tries: int, reply: ModelReply) -> None:
    """Stop when this try still asks for tools and no try remains."""
    if round_index < max_tries:
        return
    pending = ", ".join(call.name for call in reply.tool_calls)
    raise ClientError(
        "loop",
        f"Stopped after {max_tries} tries with {pending} still pending.",
    )


async def _use_tools(
    session: Any,
    reply: ModelReply,
    messages: list[dict[str, Any]],
    *,
    log: Transcript,
    round_index: int,
    approve: Approve,
    repairs: int,
) -> int:
    """Run each call in this reply. Return the repair count."""
    for call in reply.tool_calls:
        text, repairs = await carry_out(
            session,
            call,
            log=log,
            round_index=round_index,
            approve=approve,
            repairs=repairs,
        )
        messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
    return repairs
