"""The loop sends each gateway result to the model, and asks before an approval retry."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mcp.shared.exceptions import MCPError
from mcp.types import Tool

from model_client import _apply_env, load_env, read_config
from model_client.errors import ClientError
from model_client.log import Transcript, report
from model_client.loop import converse, run
from model_client.model import ModelReply, ToolCall
from model_client.retry import confirm_approval

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
    def __init__(
        self, result: SimpleNamespace | Exception | list[SimpleNamespace | Exception]
    ) -> None:
        self._results = list(result) if isinstance(result, list) else [result]
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self, cursor: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(tools=[COUNT_WORDS], next_cursor=None)

    async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
        self.calls.append((name, arguments))
        result = self._results.pop(0) if len(self._results) > 1 else self._results[0]
        if isinstance(result, Exception):
            raise result
        return result


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
            raw = log.path.read_text(encoding="utf-8")
            file_mode = log.path.stat().st_mode & 0o777
            directory_mode = log.path.parent.stat().st_mode & 0o777

        self.assertEqual(text, "3")
        self.assertEqual(session.calls, [("count_words", {"text": "one two three"})])
        listed = next(event for event in events if event["event"] == "tools_listed")
        result = next(event for event in events if event["event"] == "tool_result")
        final = next(event for event in events if event["event"] == "final")
        self.assertEqual(listed["tools"][0]["name"], "count_words")
        self.assertNotIn("prompt", listed)
        self.assertNotIn("arguments", result)
        response = next(event for event in events if event["event"] == "model_response")
        self.assertNotIn("arguments", response["tool_calls"][0])
        self.assertNotIn("one two three", raw)
        self.assertEqual(file_mode, 0o600)
        self.assertEqual(directory_mode, 0o700)
        self.assertEqual(result["ok"], True)
        self.assertNotIn("structured_content", result)
        self.assertNotIn("content", result)
        self.assertNotIn("message", result)
        self.assertEqual(final["stop"], "final")
        handed = script.seen[1][-1]
        self.assertEqual(handed["role"], "tool")
        self.assertEqual(json.loads(handed["content"]), {"count": 3})

    def test_a_policy_refusal_stops_without_another_reply(self) -> None:
        session = FakeSession(MCPError(-32002, "tool_not_pinned"))
        script = Script(
            [
                ModelReply(None, [_call("most_common_word", {"text": "one two one"})]),
                ModelReply("should not run", []),
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

        self.assertEqual(len(script.seen), 1)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(raised.exception.message, "tool_not_pinned")
        result = next(event for event in events if event["event"] == "tool_result")
        self.assertEqual(result["kind"], "policy")
        self.assertEqual(result["code"], -32002)

    def test_declined_approval_stops_before_another_reply(self) -> None:
        session = FakeSession(
            MCPError(-32005, "most_common_word needs human approval.")
        )
        script = Script(
            [
                ModelReply(None, [_call("most_common_word", {"text": "one two one"})]),
                ModelReply("should not run", []),
            ]
        )
        asked: list[tuple[str, dict]] = []

        def approve(name: str, arguments: dict) -> bool:
            asked.append((name, arguments))
            return False

        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            with self.assertRaises(ClientError) as raised:
                asyncio.run(
                    converse(
                        "What is the most common word in 'one two one'?",
                        session=session,
                        complete=script,
                        log=log,
                        approve=approve,
                    )
                )
            events = _read(log.path)

        self.assertEqual(asked, [("most_common_word", {"text": "one two one"})])
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(len(script.seen), 1)
        self.assertEqual(raised.exception.source, "loop")
        self.assertEqual(raised.exception.message, "Declined most_common_word.")
        approval = next(event for event in events if event["event"] == "approval")
        self.assertEqual(approval["allowed"], False)
        self.assertEqual(approval["name"], "most_common_word")
        final = next(event for event in events if event["event"] == "final")
        self.assertEqual(final["stop"], "error")
        self.assertEqual(final["source"], "loop")
        self.assertEqual(final["message"], "Declined most_common_word.")

    def test_approved_call_is_retried_once(self) -> None:
        session = FakeSession(
            [
                MCPError(-32005, "most_common_word needs human approval."),
                SimpleNamespace(
                    is_error=False,
                    structured_content={"word": "one", "count": 2},
                    content=[],
                ),
            ]
        )
        script = Script(
            [
                ModelReply(None, [_call("most_common_word", {"text": "one two one"})]),
                ModelReply("one", []),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            text = asyncio.run(
                converse(
                    "What is the most common word in 'one two one'?",
                    session=session,
                    complete=script,
                    log=log,
                    approve=lambda name, arguments: True,
                )
            )
            events = _read(log.path)

        self.assertEqual(text, "one")
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(
            session.calls,
            [
                ("most_common_word", {"text": "one two one"}),
                ("most_common_word", {"text": "one two one"}),
            ],
        )
        approval = next(event for event in events if event["event"] == "approval")
        self.assertEqual(approval["allowed"], True)
        results = [event for event in events if event["event"] == "tool_result"]
        self.assertEqual(results[0]["ok"], False)
        self.assertEqual(results[0]["code"], -32005)
        self.assertNotIn("message", results[0])
        self.assertEqual(results[1]["ok"], True)
        self.assertNotIn("structured_content", results[1])
        handed = json.loads(script.seen[1][-1]["content"])
        self.assertEqual(handed, {"word": "one", "count": 2})

    def test_a_second_approval_error_stops_without_asking_again(self) -> None:
        session = FakeSession(
            MCPError(-32005, "most_common_word needs human approval.")
        )
        script = Script(
            [
                ModelReply(None, [_call("most_common_word", {"text": "one two one"})]),
                ModelReply("should not run", []),
            ]
        )
        asked = 0

        def approve(name: str, arguments: dict) -> bool:
            nonlocal asked
            asked += 1
            return True

        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            with self.assertRaises(ClientError) as raised:
                asyncio.run(
                    converse(
                        "What is the most common word in 'one two one'?",
                        session=session,
                        complete=script,
                        log=log,
                        approve=approve,
                    )
                )
            events = _read(log.path)

        self.assertEqual(asked, 1)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(len(script.seen), 1)
        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(
            raised.exception.message, "most_common_word needs human approval."
        )
        approvals = [event for event in events if event["event"] == "approval"]
        self.assertEqual(
            approvals,
            [
                {
                    "event": "approval",
                    "round": 1,
                    "name": "most_common_word",
                    "allowed": True,
                }
            ],
        )
        results = [event for event in events if event["event"] == "tool_result"]
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["code"] == -32005 for result in results))

    def test_confirm_approval_reads_yes_or_no(self) -> None:
        arguments = {"text": "one two one"}
        question = (
            "most_common_word needs human approval.\n"
            'Arguments: {"text": "one two one"}\n'
            "Allow this call? [y/n] "
        )
        with (
            patch("builtins.input", side_effect=["", "maybe", "y"]),
            patch("builtins.print") as printed,
        ):
            self.assertIs(confirm_approval("most_common_word", arguments), True)
        self.assertEqual(printed.call_count, 3)
        self.assertEqual(printed.call_args.args[0], question)
        self.assertIs(printed.call_args.kwargs["file"], sys.stderr)

        with (
            patch("builtins.input", return_value="NO"),
            patch("builtins.print"),
        ):
            self.assertIs(confirm_approval("most_common_word", arguments), False)

        with (
            patch("builtins.input", side_effect=EOFError),
            patch("builtins.print"),
        ):
            self.assertIs(confirm_approval("most_common_word", arguments), False)

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
            text = asyncio.run(
                converse(
                    "count",
                    session=session,
                    complete=script,
                    log=log,
                )
            )
            events = _read(log.path)

        self.assertEqual(text, "The arguments were not valid.")
        self.assertEqual(session.calls, [])
        result = next(event for event in events if event["event"] == "tool_result")
        self.assertEqual(result["kind"], "arguments")
        self.assertNotIn("message", result)
        self.assertIn("not valid JSON", script.seen[1][-1]["content"])

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

    def test_read_only_is_not_retried(self) -> None:
        session = FakeSession(
            MCPError(-32006, "write_file is blocked in read-only mode.")
        )
        script = Script(
            [
                ModelReply(None, [_call("write_file", {"text": "one"})]),
                ModelReply("should not run", []),
            ]
        )

        def approve(name: str, arguments: dict) -> bool:
            raise AssertionError(name)

        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            with self.assertRaises(ClientError) as raised:
                asyncio.run(
                    converse(
                        "write",
                        session=session,
                        complete=script,
                        log=log,
                        approve=approve,
                    )
                )
            events = _read(log.path)

        self.assertEqual(len(session.calls), 1)
        self.assertEqual(len(script.seen), 1)
        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(
            raised.exception.message, "write_file is blocked in read-only mode."
        )
        result = next(event for event in events if event["event"] == "tool_result")
        self.assertEqual(result["kind"], "policy")

    def test_a_busy_gate_is_retried_without_telling_the_model(self) -> None:
        session = FakeSession(
            [
                MCPError(-32000, "Too many calls are already running. The limit is 4."),
                SimpleNamespace(
                    is_error=False, structured_content={"count": 3}, content=[]
                ),
            ]
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
        self.assertEqual(len(session.calls), 2)
        handed = json.loads(script.seen[1][-1]["content"])
        self.assertEqual(handed, {"count": 3})
        result = next(event for event in events if event["event"] == "tool_result")
        self.assertEqual(result["ok"], True)
        self.assertEqual(result["retried"], True)
        self.assertNotIn("kind", result)

    def test_a_second_busy_failure_stops(self) -> None:
        session = FakeSession(
            MCPError(-32000, "Too many calls are already running. The limit is 4.")
        )
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

        self.assertEqual(len(session.calls), 2)
        self.assertEqual(len(script.seen), 1)
        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(
            raised.exception.message,
            "Too many calls are already running. The limit is 4.",
        )
        self.assertFalse(any(event["event"] == "tool_result" for event in events))

    def test_a_timeout_is_retried_and_a_pin_change_is_not(self) -> None:
        timed_out = FakeSession(
            [
                MCPError(-32001, "Request 'tools/call' timed out"),
                SimpleNamespace(
                    is_error=False, structured_content={"count": 1}, content=[]
                ),
            ]
        )
        changed = FakeSession(MCPError(-32001, "tool_definition_changed"))
        call = _call("count_words", {"text": "one"})
        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            text = asyncio.run(
                converse(
                    "count",
                    session=timed_out,
                    complete=Script([ModelReply(None, [call]), ModelReply("1", [])]),
                    log=log,
                )
            )
        self.assertEqual(text, "1")
        self.assertEqual(len(timed_out.calls), 2)

        script = Script([ModelReply(None, [call]), ModelReply("should not run", [])])
        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            with self.assertRaises(ClientError) as raised:
                asyncio.run(
                    converse(
                        "count",
                        session=changed,
                        complete=script,
                        log=log,
                    )
                )
        self.assertEqual(len(changed.calls), 1)
        self.assertEqual(len(script.seen), 1)
        self.assertEqual(raised.exception.message, "tool_definition_changed")

    def test_one_bad_argument_can_be_repaired(self) -> None:
        session = FakeSession(
            [
                SimpleNamespace(
                    is_error=True,
                    structured_content=None,
                    content=[SimpleNamespace(text="text is too long")],
                ),
                SimpleNamespace(
                    is_error=False, structured_content={"count": 1}, content=[]
                ),
            ]
        )
        script = Script(
            [
                ModelReply(None, [_call("count_words", {"text": "too long"})]),
                ModelReply(None, [_call("count_words", {"text": "one"})]),
                ModelReply("1", []),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            text = asyncio.run(
                converse(
                    "count",
                    session=session,
                    complete=script,
                    log=log,
                )
            )
            events = _read(log.path)

        self.assertEqual(text, "1")
        self.assertEqual(
            session.calls,
            [("count_words", {"text": "too long"}), ("count_words", {"text": "one"})],
        )
        self.assertEqual(script.seen[1][-1]["content"], "text is too long")
        kinds = [
            event.get("kind") for event in events if event["event"] == "tool_result"
        ]
        self.assertEqual(kinds, ["arguments", None])

    def test_a_second_bad_argument_stops(self) -> None:
        session = FakeSession(
            SimpleNamespace(
                is_error=True,
                structured_content=None,
                content=[SimpleNamespace(text="text is too long")],
            )
        )
        script = Script(
            [
                ModelReply(None, [_call("count_words", {"text": "too long"})]),
                ModelReply(None, [_call("count_words", {"text": "still too long"})]),
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

        self.assertEqual(len(session.calls), 2)
        self.assertEqual(len(script.seen), 2)
        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(raised.exception.message, "text is too long")

    def test_a_later_bad_argument_stops(self) -> None:
        session = FakeSession(
            [
                SimpleNamespace(
                    is_error=True,
                    structured_content=None,
                    content=[SimpleNamespace(text="text is too long")],
                ),
                SimpleNamespace(
                    is_error=False, structured_content={"count": 1}, content=[]
                ),
                SimpleNamespace(
                    is_error=True,
                    structured_content=None,
                    content=[SimpleNamespace(text="text is still too long")],
                ),
            ]
        )
        script = Script(
            [
                ModelReply(None, [_call("count_words", {"text": "too long"})]),
                ModelReply(None, [_call("count_words", {"text": "one"})]),
                ModelReply(None, [_call("count_words", {"text": "too long again"})]),
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

        self.assertEqual(len(session.calls), 3)
        self.assertEqual(len(script.seen), 3)
        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(raised.exception.message, "text is still too long")

    def test_tries_defaults_to_four_and_rejects_a_higher_value(self) -> None:
        saved = os.environ.pop("TRIES", None)
        try:
            self.assertEqual(read_config().tries, 4)
            os.environ["TRIES"] = "2"
            self.assertEqual(read_config().tries, 2)
            os.environ["TRIES"] = "5"
            with self.assertRaises(ClientError) as raised:
                read_config()
            self.assertEqual(raised.exception.source, "loop")
            self.assertEqual(
                raised.exception.message, "TRIES must be an integer from 1 to 4."
            )
            os.environ["TRIES"] = "nope"
            with self.assertRaises(ClientError) as raised_text:
                read_config()
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

    def test_a_model_crash_is_logged(self) -> None:
        session = FakeSession(SimpleNamespace())

        def complete(messages: list[dict], tools: list[dict]) -> ModelReply:
            raise RuntimeError("model crashed")

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

        self.assertEqual(str(raised.exception), "gateway: model crashed")
        finals = [event for event in events if event["event"] == "final"]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["stop"], "error")
        self.assertEqual(finals[0]["source"], "gateway")
        self.assertEqual(finals[0]["message"], "model crashed")

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
                asyncio.run(run("count", log, "test-key", 4))
            events = _read(log.path)

        self.assertEqual(raised.exception.source, "gateway")
        self.assertEqual(raised.exception.message, "gateway did not start")
        self.assertEqual(events[-1]["stop"], "error")
        self.assertEqual(events[-1]["message"], "gateway did not start")
