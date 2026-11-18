"""Ask a model, and run its tool calls through the local MCP gateway."""

from __future__ import annotations

import argparse
import asyncio
import sys

from model_client.errors import ClientError
from model_client.log import Transcript, log_directory
from model_client.loop import run


def parse_prompt() -> str:
    """Read the prompt from the command line."""
    parser = argparse.ArgumentParser(
        prog="model-client",
        description=(
            "Ask a model, and run its tool calls through the local MCP gateway. "
            "Writes a JSONL log of the tool list, each model reply, and each gateway result."
        ),
    )
    parser.add_argument("prompt", help="What to ask the model")
    return parser.parse_args().prompt


def main() -> None:
    prompt = parse_prompt()
    log: Transcript | None = None
    try:
        log = Transcript(log_directory())
        text = asyncio.run(run(prompt, log))
    except ClientError as exc:
        if log is not None and log.path.is_file() and log.path.stat().st_size:
            print(log.path, file=sys.stderr)
        raise SystemExit(str(exc)) from None
    print(text)
    print(log.path, file=sys.stderr)
