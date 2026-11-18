# Run the model client

`model-client` sends one prompt to `grok-4.7`. It runs each tool call the model asks for through the word-counter gateway.

You need Python 3.12. [uv](https://docs.astral.sh/uv/) installs the dependencies from `pyproject.toml` and `uv.lock`.

`model-client` looks for `agent-trust-gateway/gateway` in a parent directory. Put `XAI_API_KEY` in `.env` next to `pyproject.toml`. `model-client` loads that `.env`. Accept `count_words` and `most_common_word` before you run `model-client`. The steps are in [Run the word counter](../../agent-trust-gateway/gateway/README.md). If a tool has no pin, the gateway refuses the call with `tool_not_pinned`.

## Run the client

Run these commands in this directory.

```bash
uv sync
uv run model-client "How many words are in 'one two three'?"
```

`uv sync` installs the dependencies into `.venv`. `model-client` takes one argument, the prompt. That command prints the answer on stdout and the log path on stderr. The log is a JSONL file in `logs/`. Each line is one JSON object. The `event` field is `tools_listed`, `model_response`, `tool_result`, or `final`.

The gateway writes one timing line to stderr for each message it handles.

`model-client` exits with status code 0 when the run succeeds. On a `ClientError`, `model-client` prints `source: message` and exits with status code 1. `source` is `gateway`, `model`, `log`, or `loop`. A refused tool call is a `ClientError`. The command prints that refusal. The model's later text stays in the `model_response` log line.

Each reply from the model is one try. The client stops after 4 tries. Set `TRIES` in `.env` to an integer from 1 to 4. A lower value stops the run after that many replies. When a tool call fails on an earlier try and the model asks for another tool on the last try, the command prints that failure.

## Run the tests

Run this command in this directory.

```bash
uv run python -m unittest discover -s tests -v
```

The run ends with `OK` when every test passes. When a test fails, the process exits with status code 1.

## Lint and format

Run these commands in this directory.

```bash
uv run ruff check
uv run ruff format
```

`ruff check` prints `All checks passed!` when every check passes. When a check fails, `ruff check` exits with status code 1.

`ruff format` rewrites a file when its formatting differs from the ruff format. When every file matches, `ruff format` prints the count of files left unchanged and writes nothing.
