"""Start the word-counter gateway over stdio."""

from __future__ import annotations

import tomllib
from pathlib import Path

from mcp import Client, StdioServerParameters

from model_client.errors import ClientError

_GATEWAY_PACKAGE = Path("agent-trust-gateway") / "gateway"


def gateway_directory() -> Path:
    """Return `../../agent-trust-gateway/gateway` from this project."""
    return _discover_gateway(Path(__file__).resolve().parent)


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


def _discover_gateway(start: Path) -> Path:
    """Return `../../agent-trust-gateway/gateway` from this project.

    The directory must be real. The project name must be `gateway`.
    `src/gateway/server.py` must exist.
    """
    root = _project_root(start)
    gateway = root.parent.parent / _GATEWAY_PACKAGE
    package = gateway.parent
    if package.is_symlink() or gateway.is_symlink():
        raise ClientError("gateway", "The gateway checkout is a symlink.")
    if not gateway.is_dir():
        raise ClientError(
            "gateway",
            "The gateway checkout agent-trust-gateway/gateway is missing.",
        )
    if _project_name(gateway / "pyproject.toml") != "gateway":
        raise ClientError("gateway", "The gateway project name is not gateway.")
    server = gateway / "src" / "gateway" / "server.py"
    if not server.is_file():
        raise ClientError(
            "gateway", "The gateway checkout has no src/gateway/server.py."
        )
    return gateway.resolve()


def _project_name(path: Path) -> str | None:
    """Return the `[project].name` value, or None when the file is unusable."""
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project = data.get("project")
    if not isinstance(project, dict):
        return None
    name = project.get("name")
    if not isinstance(name, str):
        return None
    return name


def _project_root(start: Path) -> Path:
    """The directory that holds this project's `pyproject.toml` and `src`."""
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    raise ClientError("gateway", "Could not find the model-client project.")
