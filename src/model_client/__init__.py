"""Ask a model, and run its tool calls through the local MCP gateway."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from model_client.errors import ClientError
from model_client.log import Transcript, log_directory
from model_client.loop import MAX_TRIES, run


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


@dataclass(frozen=True)
class Config:
    """The API key and the try limit for one run."""

    api_key: str
    tries: int


def load_env() -> None:
    """Load this project's `.env`. Set a variable only when it is unset."""
    start = Path(__file__).resolve().parent
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            env_file = parent / ".env"
            if env_file.is_file():
                _apply_env(env_file)
            return


def read_config() -> Config:
    """Read `XAI_API_KEY` and `TRIES`.

    A missing key is an empty string. A missing or blank `TRIES` is 4.
    """
    raw = os.environ.get("TRIES", "").strip() or str(MAX_TRIES)
    try:
        tries = int(raw)
    except ValueError:
        tries = 0
    if not 1 <= tries <= MAX_TRIES:
        raise ClientError("loop", f"TRIES must be an integer from 1 to {MAX_TRIES}.")
    return Config(api_key=os.environ.get("XAI_API_KEY", ""), tries=tries)


def main() -> None:
    prompt = parse_prompt()
    log: Transcript | None = None
    try:
        load_env()
        log = Transcript(log_directory())
        settings = read_config()
        text = asyncio.run(run(prompt, log, settings.api_key, settings.tries))
    except ClientError as exc:
        if log is not None and not exc.recorded:
            try:
                log.record(exc)
            except ClientError:
                pass
        if log is not None and log.path.is_file() and log.path.stat().st_size:
            print(log.path, file=sys.stderr)
        raise SystemExit(str(exc)) from None
    print(text)
    print(log.path, file=sys.stderr)


def _apply_env(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise ClientError("model", f"Could not read {path.name}: {detail}") from exc
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
