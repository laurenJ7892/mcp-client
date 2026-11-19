# Run the model client

`model-client` sends one prompt to `grok-4.7`. It runs each tool call the model asks for through the word-counter gateway.

You need Python 3.12. [uv](https://docs.astral.sh/uv/) installs the dependencies from `pyproject.toml` and `uv.lock`.

`model-client` starts `../../agent-trust-gateway/gateway` relative to this project. The checkout must be a real directory. Its project name must be `gateway`. It must contain `src/gateway/server.py`. A symlink is an error. `model-client` exits with status code 1. stderr prints `gateway:` and the reason.

Put `XAI_API_KEY` in `.env` next to `pyproject.toml`. `model-client` loads that `.env`. Accept `count_words` and `most_common_word` before you run `model-client`. The steps are in [Run the word counter](../../agent-trust-gateway/gateway/README.md). If a tool has no pin, the gateway refuses the call with `tool_not_pinned`.

## Run the client

Run these commands in this directory.

```bash
uv sync
uv run model-client "How many words are in 'one two three'?"
```

`uv sync` installs the dependencies into `.venv`. `model-client` takes one argument, the prompt. That command prints the answer on stdout and the log path on stderr. The log is a JSONL file in `logs/`. Each line is one JSON object. The `event` field is `tools_listed`, `model_response`, `tool_result`, `approval`, or `final`.

`model-client` creates the `logs` directory with mode `0700` and each JSONL file with mode `0600`. The JSONL file has no prompt and no tool arguments. A `tool_result` line records the tool name, `ok`, `code`, and `kind`. It has no `content`, no `message`, and no `structured_content`.

The gateway writes one timing line to stderr for each message it handles.

`model-client` exits with status code 0 when the run succeeds. On a `ClientError`, `model-client` prints `source: message` and exits with status code 1. `source` is `gateway`, `model`, `log`, or `loop`.

A tool failure has a `kind` on its `tool_result` line. A `transient` failure is a timeout or `Too many calls are already running.` The client calls that tool once more. The model does not see the first failure. A second failure of that kind ends the run with `gateway:` and the message. An `arguments` failure is invalid JSON, JSON-RPC `-32602`, or a tool result with `is_error`. The client sends that result to the model, and the model can call again. A second `arguments` failure ends the run. A `policy` failure is a pin refusal, read-only code `-32006`, or approval code `-32005`. The client does not retry a policy failure on its own. A pin refusal or a read-only refusal ends the run with `gateway:` and the gateway message.

An approval refusal pauses before another model reply. On stderr, the client prints the tool name, a line that starts with `Arguments:`, and `Allow this call? [y/n]`. The client reads the answer from stdin. `y` and `yes` run that call once more. `n`, `no`, and an end of file end the run with `loop: Declined name.` A second `-32005` from that retry ends the run with `gateway:` and the gateway message. The `approval` log line records `allowed` as true or false. The next model reply sees the retried result.

Each reply from the model is one try. The client stops after 4 tries. Set `TRIES` in `.env` to an integer from 1 to 4. A lower value stops the run after that many replies.

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
