"""ClientError keeps one explanation when a task group wraps a failure."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mcp.shared.exceptions import MCPError

from model_client.errors import (
    ClientError,
    FailureKind,
    as_client_error,
    failure_kind,
    needs_approval,
)
from model_client.log import Transcript


class ErrorTests(unittest.TestCase):
    def test_exception_group_unwraps_a_client_error(self) -> None:
        inner = ClientError("model", "no credits")
        group = ExceptionGroup("unhandled", [inner])
        self.assertIs(as_client_error(group), inner)

    def test_mcp_error_names_the_gateway(self) -> None:
        error = as_client_error(
            MCPError(-32005, "most_common_word needs human approval.")
        )
        self.assertEqual(error.source, "gateway")
        self.assertEqual(
            error.message, "most_common_word needs human approval. (-32005)"
        )
        self.assertEqual(
            str(error),
            "gateway: most_common_word needs human approval. (-32005)",
        )

    def test_os_error_names_the_gateway(self) -> None:
        error = as_client_error(OSError("uv failed"))
        self.assertEqual(error.source, "gateway")
        self.assertIn("uv failed", error.message)

    def test_failure_kind_splits_timeout_busy_arguments_and_policy(self) -> None:
        busy = {
            "ok": False,
            "code": -32000,
            "message": "Too many calls are already running. The limit is 4.",
        }
        timeout = {
            "ok": False,
            "code": -32001,
            "message": "Request 'tools/call' timed out",
        }
        changed = {"ok": False, "code": -32001, "message": "tool_definition_changed"}
        pinned = {"ok": False, "code": -32002, "message": "tool_not_pinned"}
        approval = {
            "ok": False,
            "code": -32005,
            "message": "most_common_word needs human approval.",
        }
        readonly = {
            "ok": False,
            "code": -32006,
            "message": "write_file is blocked in read-only mode.",
        }
        invalid = {"ok": False, "code": -32602, "message": "text is required"}
        tool_error = {"ok": False, "is_error": True, "message": "text is too long"}
        local = {"ok": False, "message": "Tool arguments are not valid JSON: ."}
        closed = {"ok": False, "code": -32000, "message": "Connection closed"}

        self.assertIs(failure_kind(busy), FailureKind.TRANSIENT)
        self.assertIs(failure_kind(timeout), FailureKind.TRANSIENT)
        self.assertIs(failure_kind(changed), FailureKind.POLICY)
        self.assertIs(failure_kind(pinned), FailureKind.POLICY)
        self.assertIs(failure_kind(approval), FailureKind.POLICY)
        self.assertTrue(needs_approval(approval))
        self.assertFalse(needs_approval(readonly))
        self.assertFalse(needs_approval({"ok": True, "code": -32005}))
        self.assertIs(failure_kind(readonly), FailureKind.POLICY)
        self.assertIs(failure_kind(invalid), FailureKind.ARGUMENTS)
        self.assertIs(failure_kind(tool_error), FailureKind.ARGUMENTS)
        self.assertIs(failure_kind(local), FailureKind.ARGUMENTS)
        self.assertIsNone(failure_kind(closed))
        self.assertIsNone(
            failure_kind({"ok": True, "structured_content": {"count": 3}})
        )

    def test_record_writes_one_line(self) -> None:
        error = ClientError("model", "no credits")
        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            log.record(error)
            log.record(error)
            lines = log.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(error.recorded)
