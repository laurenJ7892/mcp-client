"""The client lists the live gateway and records a real refusal."""

from __future__ import annotations

import asyncio
import json
import unittest

from model_client.errors import ClientError
from model_client.gateway import (
    call_tool,
    gateway_directory,
    list_tools,
    open_gateway,
)


def _directory():
    try:
        return gateway_directory()
    except ClientError as exc:
        raise unittest.SkipTest(str(exc)) from exc


class GatewayTests(unittest.TestCase):
    def test_list_and_refusal(self) -> None:
        directory = _directory()

        async def run() -> tuple[list[str], dict, dict]:
            async with open_gateway(directory) as session:
                tools = await list_tools(session)
                names = [tool.name for tool in tools]
                allowed_text, allowed = await call_tool(
                    session, "count_words", {"text": "one two three"}
                )
                refused_text, refused = await call_tool(
                    session, "most_common_word", {"text": "one two one"}
                )
            return (
                names,
                {**allowed, "text": allowed_text},
                {**refused, "text": refused_text},
            )

        names, allowed, refused = asyncio.run(run())
        self.assertIn("count_words", names)
        self.assertIn("most_common_word", names)
        self.assertEqual(allowed["ok"], True)
        self.assertEqual(json.loads(allowed["text"]), {"count": 3})
        self.assertEqual(refused["ok"], False)
        self.assertEqual(refused["code"], -32005)
        self.assertEqual(refused["message"], "most_common_word needs human approval.")
        handed = json.loads(refused["text"])
        self.assertEqual(handed["message"], "most_common_word needs human approval.")
