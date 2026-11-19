"""Ask an xAI model, using the gateway tools as functions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import APIError, APIStatusError, OpenAI

from model_client.errors import ClientError

XAI_BASE_URL = "https://api.x.ai/v1"
MODEL = "grok-4.7"


@dataclass(frozen=True)
class ToolCall:
    """One function call from the model."""

    id: str
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str
    error: str | None


@dataclass(frozen=True)
class ModelReply:
    """Text and tool calls from one completion."""

    content: str | None
    tool_calls: list[ToolCall]

    def message(self) -> dict[str, Any]:
        """The assistant message to send back on the next turn."""
        payload: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return payload


def build_client(api_key: str) -> OpenAI:
    """Return an OpenAI client for the xAI chat completions endpoint."""
    if not api_key:
        raise ClientError("model", "Set XAI_API_KEY.")
    return OpenAI(api_key=api_key, base_url=XAI_BASE_URL, timeout=180.0)


def ask(
    client: OpenAI,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> ModelReply:
    """One chat completion. Tool choice stays with the model."""
    kwargs: dict[str, Any] = {"model": MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
        kwargs["parallel_tool_calls"] = False
    try:
        response = client.chat.completions.create(**kwargs)
    except APIStatusError as exc:
        raise ClientError("model", error_message(exc)) from exc
    except APIError as exc:
        raise ClientError("model", str(exc)) from exc
    if not response.choices:
        raise ClientError("model", "The model returned no choices.")
    return _reply(response.choices[0].message)


def error_message(exc: APIStatusError) -> str:
    """Return the provider's error text from the response body.

    This stays next to `ask`. The body belongs to the OpenAI client, and
    `errors` does not import that client.
    """
    response = getattr(exc, "response", None)
    body: Any = None
    if response is not None:
        try:
            body = response.json()
        except (AttributeError, UnicodeDecodeError, ValueError):
            body = None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, str) and error:
            return error
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message:
        return message
    return str(exc)


def _reply(message: Any) -> ModelReply:
    calls = []
    for call in message.tool_calls or []:
        raw = call.function.arguments or "{}"
        arguments, error = _parse_arguments(raw)
        calls.append(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=arguments,
                raw_arguments=raw if isinstance(raw, str) else json.dumps(raw),
                error=error,
            )
        )
    return ModelReply(content=_text(message.content), tool_calls=calls)


def _parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(raw, dict):
        return raw, None
    if not isinstance(raw, str):
        return None, "Tool arguments must be a JSON object."
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Tool arguments are not valid JSON: {exc.msg}."
    if not isinstance(parsed, dict):
        return None, "Tool arguments must be a JSON object."
    return parsed, None


def _text(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = (
                part.get("text")
                if isinstance(part, dict)
                else getattr(part, "text", None)
            )
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content)
