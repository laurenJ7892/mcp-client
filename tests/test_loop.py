"""The loop sends each gateway result to the model, including a refusal."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mcp.shared.exceptions import MCPError
from mcp.types import Tool

from model_client.errors import ClientError
from model_client.log import Transcript
from model_client.loop import converse, max_tries, report, run
from model_client.model import ModelReply, ToolCall, _apply_env, load_env

COUNT_WORDS = Tool(
    name="count_words",
    description="Count words by splitting on whitespace.",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)


class Script:
    def __init__(self, replies: list[ModelReply]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def __call__(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        self.seen.append(list(messages))
        return self.replies.pop(0)


class FakeSession:
    def __init__(self, result: SimpleNamespace | Exception) -> None:
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self, cursor: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(tools=[COUNT_WORDS], next_cursor=None)

    async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
        self.calls.append((name, arguments))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _call(name: str, arguments: dict, call_id: str = "call-1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments=json.dumps(arguments),
        error=None,
    )


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class LoopTests(unittest.TestCase):
    def test_success_is_handed_back(self) -> None:
        session = FakeSession(
            SimpleNamespace(
                is_error=False,
                structured_content={"count": 3},
                content=[],
            )
        )
        script = Script(
            [
                ModelReply(None, [_call("count_words", {"text": "one two three"})]),
                ModelReply("3", []),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            text = asyncio.run(
                converse(
                    "How many words are in 'one two three'?",
                    session=session,
                    complete=script,
                    log=log,
                )
            )
            events = _read(log.path)

        self.assertEqual(text, "3")
        self.assertEqual(session.calls, [("count_words", {"text": "one two three"})])
        listed = next(event for event in events if event["event"] == "tools_listed")
        result = next(event for event in events if event["event"] == "tool_result")
        final = next(event for event in events if event["event"] == "final")
        self.assertEqual(listed["tools"][0]["name"], "count_words")
        self.assertEqual(result["structured_content"], {"count": 3})
        self.assertEqual(final["stop"], "final")
        handed = script.seen[1][-1]
        self.assertEqual(handed["role"], "tool")
        self.assertEqual(json.loads(handed["content"]), {"count": 3})

    def test_jsonrpc_refusal_is_handed_back(self) -> None:
        session = FakeSession(
            MCPError(-32005, "most_common_word needs human approval.")
        )
        script = Script(
            [
                ModelReply(None, [_call("most_common_word", {"text": "one two one"})]),
                ModelReply("most_common_word needs human approval.", []),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            with self.assertRaises(ClientError) as raised:
                asyncio.run(
                    converse(
                        "What is the most common word in 'one two one'?",
                        session=session,
                        complete=script,
                        log=log,
                    )
                )
            events = _read(log.path)

        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(
            raised.exception.message, "most_common_word needs human approval."
        )
        self.assertFalse(any(event.get("stop") == "final" for event in events))
        result = next(event for event in events if event["event"] == "tool_result")
        self.assertEqual(result["ok"], False)
        self.assertEqual(result["code"], -32005)
        self.assertEqual(result["message"], "most_common_word needs human approval.")
        handed = json.loads(script.seen[1][-1]["content"])
        self.assertEqual(handed["code"], -32005)
        self.assertEqual(handed["message"], "most_common_word needs human approval.")

    def test_invalid_arguments_skip_the_gateway(self) -> None:
        session = FakeSession(SimpleNamespace())
        script = Script(
            [
                ModelReply(
                    None,
                    [
                        ToolCall(
                            id="call-1",
                            name="count_words",
                            arguments=None,
                            raw_arguments="not-json",
                            error="Tool arguments are not valid JSON: .",
                        )
                    ],
                ),
                ModelReply("The arguments were not valid.", []),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            with self.assertRaises(ClientError) as raised:
                asyncio.run(
                    converse(
                        "count",
                        session=session,
                        complete=script,
                        log=log,
                    )
                )
            events = _read(log.path)

        self.assertEqual(raised.exception.source, "loop")
        self.assertIn("not valid JSON", raised.exception.message)
        self.assertEqual(session.calls, [])
        result = next(event for event in events if event["event"] == "tool_result")
        self.assertEqual(result["ok"], False)
        self.assertIn("not valid JSON", result["message"])

    def test_max_rounds_leaves_the_last_call_unrun(self) -> None:
        session = FakeSession(
            SimpleNamespace(is_error=False, structured_content={"count": 1}, content=[])
        )
        call = _call("count_words", {"text": "one"})
        script = Script([ModelReply(None, [call]), ModelReply(None, [call])])
        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            with self.assertRaises(ClientError) as raised:
                asyncio.run(
                    converse(
                        "count",
                        session=session,
                        complete=script,
                        log=log,
                        max_tries=2,
                    )
                )
            events = _read(log.path)

        self.assertEqual(len(session.calls), 1)
        self.assertEqual(raised.exception.source, "loop")
        self.assertEqual(
            raised.exception.message,
            "Stopped after 2 tries with count_words still pending.",
        )
        final = next(event for event in events if event["event"] == "final")
        self.assertEqual(final["stop"], "error")
        self.assertEqual(final["message"], raised.exception.message)

    def test_max_tries_reports_an_earlier_refusal(self) -> None:
        session = FakeSession(
            MCPError(-32005, "most_common_word needs human approval.")
        )
        call = _call("most_common_word", {"text": "one two one"})
        script = Script([ModelReply(None, [call]), ModelReply(None, [call])])
        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            with self.assertRaises(ClientError) as raised:
                asyncio.run(
                    converse(
                        "What is the most common word in 'one two one'?",
                        session=session,
                        complete=script,
                        log=log,
                        max_tries=2,
                    )
                )
            events = _read(log.path)

        self.assertEqual(len(session.calls), 1)
        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(
            raised.exception.message, "most_common_word needs human approval."
        )
        self.assertFalse(any(event.get("stop") == "final" for event in events))

    def test_tries_defaults_to_four_and_rejects_a_higher_value(self) -> None:
        saved = os.environ.pop("TRIES", None)
        try:
            self.assertEqual(max_tries(), 4)
            os.environ["TRIES"] = "2"
            self.assertEqual(max_tries(), 2)
            os.environ["TRIES"] = "5"
            with self.assertRaises(ClientError) as raised:
                max_tries()
            self.assertEqual(raised.exception.source, "loop")
            self.assertEqual(
                raised.exception.message, "TRIES must be an integer from 1 to 4."
            )
            os.environ["TRIES"] = "nope"
            with self.assertRaises(ClientError) as raised_text:
                max_tries()
            self.assertEqual(
                raised_text.exception.message, "TRIES must be an integer from 1 to 4."
            )
        finally:
            if saved is None:
                os.environ.pop("TRIES", None)
            else:
                os.environ["TRIES"] = saved

    def test_model_error_is_logged(self) -> None:
        session = FakeSession(SimpleNamespace())

        def complete(messages: list[dict], tools: list[dict]) -> ModelReply:
            raise ClientError("model", "no credits")

        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            with self.assertRaises(ClientError) as raised:
                asyncio.run(
                    converse(
                        "count",
                        session=session,
                        complete=complete,
                        log=log,
                    )
                )
            events = _read(log.path)

        self.assertEqual(str(raised.exception), "model: no credits")
        self.assertEqual(session.calls, [])
        finals = [event for event in events if event["event"] == "final"]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["stop"], "error")
        self.assertEqual(finals[0]["source"], "model")
        self.assertEqual(finals[0]["message"], "no credits")

    def test_gateway_crash_stops_the_run(self) -> None:
        session = FakeSession(RuntimeError("connection closed"))
        script = Script(
            [
                ModelReply(None, [_call("count_words", {"text": "one"})]),
                ModelReply("should not run", []),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            with self.assertRaises(ClientError) as raised:
                asyncio.run(
                    converse(
                        "count",
                        session=session,
                        complete=script,
                        log=log,
                    )
                )
            events = _read(log.path)

        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(raised.exception.message, "connection closed")
        self.assertEqual(len(script.seen), 1)
        self.assertFalse(any(event["event"] == "tool_result" for event in events))
        final = next(event for event in events if event["event"] == "final")
        self.assertEqual(final["source"], "gateway")
        self.assertEqual(final["message"], "connection closed")

    def test_a_failed_log_write_keeps_the_original_error(self) -> None:
        class Broken(Transcript):
            def record(self, exc: ClientError) -> None:
                raise ClientError("log", "Could not write the log: No space left")

        error = ClientError("model", "no credits")
        with tempfile.TemporaryDirectory() as directory:
            log = Broken(Path(directory))
            with self.assertRaises(ClientError) as raised:
                report(log, error)

        self.assertIs(raised.exception, error)

    def test_unreadable_env_file_is_a_model_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = root / ".env"
            env.write_text("XAI_API_KEY=secret\n", encoding="utf-8")
            env.chmod(0)
            try:
                with self.assertRaises(ClientError) as raised:
                    _apply_env(env)
            finally:
                env.chmod(0o644)

        self.assertEqual(raised.exception.source, "model")
        self.assertIn(".env", raised.exception.message)
        self.assertNotIn("secret", raised.exception.message)

    def test_load_env_ignores_the_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text("", encoding="utf-8")
            (root / ".env").write_text("XAI_API_KEY=from-cwd\n", encoding="utf-8")
            previous = Path.cwd()
            saved = os.environ.pop("XAI_API_KEY", None)
            try:
                os.chdir(root)
                load_env()
                self.assertNotEqual(os.environ.get("XAI_API_KEY"), "from-cwd")
            finally:
                os.chdir(previous)
                if saved is None:
                    os.environ.pop("XAI_API_KEY", None)
                else:
                    os.environ["XAI_API_KEY"] = saved

    def test_session_startup_error_is_a_client_error(self) -> None:
        class Boom:
            async def __aenter__(self) -> None:
                raise RuntimeError("gateway did not start")

            async def __aexit__(self, *args: object) -> bool:
                return False

        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            with (
                patch(
                    "model_client.loop.gateway_directory",
                    return_value=Path(directory),
                ),
                patch("model_client.loop.build_client", return_value=object()),
                patch("model_client.loop.open_gateway", return_value=Boom()),
                self.assertRaises(ClientError) as raised,
            ):
                asyncio.run(run("count", log))
            events = _read(log.path)

        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(raised.exception.message, "gateway did not start")
        self.assertEqual(events[-1]["stop"], "error")
        self.assertEqual(events[-1]["message"], "gateway did not start")
