"""List and call paths the loop tests do not reach."""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace

from mcp.shared.exceptions import MCPError
from mcp.types import Tool

from model_client.errors import ClientError
from model_client.tools import call_tool, list_tools

COUNT_WORDS = Tool(
    name="count_words",
    description="Count words by splitting on whitespace.",
    input_schema={"type": "object"},
)
MOST_COMMON = Tool(
    name="most_common_word",
    description="Return the word that appears most often.",
    input_schema={"type": "object"},
)


class _Pages:
    def __init__(self) -> None:
        self.cursors: list[str | None] = []

    async def list_tools(self, cursor: str | None = None) -> SimpleNamespace:
        self.cursors.append(cursor)
        if cursor is None:
            return SimpleNamespace(tools=[COUNT_WORDS], next_cursor="page-2")
        return SimpleNamespace(tools=[MOST_COMMON], next_cursor=None)


class _ListFailure:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def list_tools(self, cursor: str | None = None) -> SimpleNamespace:
        raise self._error


class _CallFailure:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
        raise self._error


class _ContentResult:
    async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
        return SimpleNamespace(
            is_error=False,
            structured_content=None,
            content=[
                SimpleNamespace(text="one"),
                SimpleNamespace(),
                SimpleNamespace(text="two"),
            ],
        )


class ToolTests(unittest.TestCase):
    def test_a_second_page_is_listed(self) -> None:
        session = _Pages()
        tools = asyncio.run(list_tools(session))

        self.assertEqual(session.cursors, [None, "page-2"])
        self.assertEqual(
            [tool.name for tool in tools], ["count_words", "most_common_word"]
        )

    def test_a_list_error_names_the_gateway(self) -> None:
        session = _ListFailure(MCPError(-32603, "internal error"))
        with self.assertRaises(ClientError) as raised:
            asyncio.run(list_tools(session))

        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(
            raised.exception.message, "tools/list failed (-32603): internal error"
        )

    def test_a_list_crash_names_the_gateway(self) -> None:
        session = _ListFailure(RuntimeError("connection closed"))
        with self.assertRaises(ClientError) as raised:
            asyncio.run(list_tools(session))

        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(
            raised.exception.message, "tools/list failed: connection closed"
        )

    def test_a_timeout_returns_request_timed_out(self) -> None:
        session = _CallFailure(TimeoutError())
        text, fields = asyncio.run(call_tool(session, "count_words", {"text": "one"}))

        self.assertEqual(
            fields,
            {"ok": False, "code": -32001, "message": "Request timed out."},
        )
        self.assertEqual(
            json.loads(text),
            {"code": -32001, "message": "Request timed out."},
        )

    def test_content_is_kept_when_there_is_no_structured_content(self) -> None:
        text, fields = asyncio.run(
            call_tool(_ContentResult(), "count_words", {"text": "one"})
        )

        self.assertEqual(text, "one\ntwo")
        self.assertEqual(fields, {"ok": True, "content": "one\ntwo"})
