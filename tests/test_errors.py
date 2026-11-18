"""ClientError keeps one explanation when a task group wraps a failure."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mcp.shared.exceptions import MCPError

from model_client.errors import ClientError, as_client_error
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

    def test_record_writes_one_line(self) -> None:
        error = ClientError("model", "no credits")
        with tempfile.TemporaryDirectory() as directory:
            log = Transcript(Path(directory))
            log.record(error)
            log.record(error)
            lines = log.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(error.recorded)
