"""Call the word-counter gateway over stdio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError
from mcp.types import Tool

from model_client.errors import ClientError

_GATEWAY_PACKAGE = Path("agent-trust-gateway") / "gateway"


def gateway_directory() -> Path:
    """Return `agent-trust-gateway/gateway` above this package."""
    return _discover_gateway()


def server_parameters(directory: Path) -> StdioServerParameters:
    """Parameters that start `python -m gateway.server` in `directory`."""
    return StdioServerParameters(
        command="uv",
        args=[
            "run",
            "--directory",
            str(directory),
            "python",
            "-m",
            "gateway.server",
        ],
    )


def open_gateway(directory: Path) -> Client:
    """Return a client. The `async with` block starts the gateway."""
    return Client(server_parameters(directory))


async def list_tools(session: Any) -> list[Tool]:
    """Follow `tools/list` cursors until the server has no further page."""
    tools: list[Tool] = []
    cursor: str | None = None
    while True:
        try:
            page = await session.list_tools(cursor=cursor)
        except MCPError as exc:
            raise ClientError(
                "gateway", f"tools/list failed ({exc.code}): {exc.message}"
            ) from exc
        except Exception as exc:
            raise ClientError("gateway", f"tools/list failed: {exc}") from exc
        tools.extend(page.tools)
        cursor = page.next_cursor
        if not cursor:
            return tools


def tool_records(tools: list[Tool]) -> list[dict[str, Any]]:
    """Names, descriptions, and input schemas offered to the model."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]


def function_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Chat-completions function tools for an xAI request."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


async def call_tool(
    session: Any, name: str, arguments: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Call one tool. Return text for the model and fields for the log.

    A JSON-RPC error returns JSON with `code` and `message`. An `is_error`
    result returns the tool content.
    """
    try:
        result = await session.call_tool(name, arguments)
    except MCPError as exc:
        text = json.dumps({"code": exc.code, "message": exc.message})
        return text, {"ok": False, "code": exc.code, "message": exc.message}
    except Exception as exc:
        raise ClientError("gateway", str(exc)) from exc
    if result.is_error:
        text = _content_text(result)
        return text, {"ok": False, "is_error": True, "message": text}
    if result.structured_content is not None:
        text = json.dumps(result.structured_content)
        return text, {"ok": True, "structured_content": result.structured_content}
    text = _content_text(result)
    return text, {"ok": True, "content": text}


def _discover_gateway() -> Path:
    start = Path(__file__).resolve().parent
    for parent in (start, *start.parents):
        candidate = parent / _GATEWAY_PACKAGE
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ClientError(
        "gateway",
        "The gateway checkout is missing (agent-trust-gateway/gateway).",
    )


def _content_text(result: Any) -> str:
    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)
