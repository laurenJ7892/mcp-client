"""The client lists the live gateway and records a real refusal."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from model_client.errors import ClientError
from model_client.gateway import _discover_gateway, gateway_directory, open_gateway
from model_client.tools import call_tool, list_tools


def _start(root: Path) -> Path:
    project = root / "python" / "model_client"
    start = project / "src" / "model_client"
    start.mkdir(parents=True)
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    return start


def _checkout(parent: Path, *, name: str = "gateway", server: bool = True) -> Path:
    gateway = parent / "agent-trust-gateway" / "gateway"
    gateway.mkdir(parents=True)
    (gateway / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\n', encoding="utf-8"
    )
    if server:
        source = gateway / "src" / "gateway"
        source.mkdir(parents=True)
        (source / "server.py").write_text("", encoding="utf-8")
    return gateway.resolve()


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


class DiscoveryTests(unittest.TestCase):
    def test_a_checkout_inside_the_project_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = _start(root)
            planted = (
                root
                / "python"
                / "model_client"
                / "src"
                / "agent-trust-gateway"
                / "gateway"
            )
            planted.mkdir(parents=True)
            (planted / "pyproject.toml").write_text("", encoding="utf-8")
            real = _checkout(root)
            found = _discover_gateway(start)

        self.assertEqual(found, real)

    def test_a_nearer_checkout_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = _start(root)
            _checkout(root / "python")
            pinned = _checkout(root)
            found = _discover_gateway(start)

        self.assertEqual(found, pinned)

    def test_a_symlink_checkout_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = _start(root)
            real = _checkout(root / "elsewhere")
            (root / "agent-trust-gateway").symlink_to(
                real.parent, target_is_directory=True
            )
            with self.assertRaises(ClientError) as raised:
                _discover_gateway(start)

        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(raised.exception.message, "The gateway checkout is a symlink.")

    def test_a_wrong_project_name_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = _start(root)
            _checkout(root, name="other")
            with self.assertRaises(ClientError) as raised:
                _discover_gateway(start)

        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(
            raised.exception.message, "The gateway project name is not gateway."
        )

    def test_a_checkout_without_the_server_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = _start(root)
            _checkout(root, server=False)
            with self.assertRaises(ClientError) as raised:
                _discover_gateway(start)

        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(
            raised.exception.message,
            "The gateway checkout has no src/gateway/server.py.",
        )

    def test_a_missing_checkout_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            start = _start(Path(directory))
            with self.assertRaises(ClientError) as raised:
                _discover_gateway(start)

        self.assertEqual(raised.exception.source, "gateway")
        self.assertIn("agent-trust-gateway/gateway", raised.exception.message)
