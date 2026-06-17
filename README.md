# PyAgent

PyAgent is a lightweight coding agent with a terminal UI, streaming model responses, tool use, layered project instructions, and switchable model profiles.

It is built with [Textual](https://textual.textualize.io/) and supports both native Ollama chat endpoints and OpenAI-compatible `/v1/chat/completions` servers such as OpenAI and vLLM.

## Quickstart

### 1. Install

```bash
pip install pyagent-harness
```

For the optional HTTP API server, install the API extra:

```bash
pip install pyagent-harness[api]
```

### 2. Configure a model

PyAgent looks for model profiles at:

```text
~/.pyagent/profiles.json
```

Create a minimal Ollama profile:

```json
{
  "default_profile": "local-qwen",
  "profiles": {
    "local-qwen": {
      "provider": "ollama",
      "base_url": "http://localhost:11434",
      "model": "qwen2.5-coder:7b"
    }
  }
}
```

Or a minimal OpenAI profile:

```json
{
  "default_profile": "openai-mini",
  "profiles": {
    "openai-mini": {
      "provider": "openai",
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-4.1-mini",
      "api_key_env": "OPENAI_API_KEY"
    }
  }
}
```

Then run:

```bash
pyagent
```

You can also run one prompt and exit:

```bash
pyagent --prompt "Summarize this repository"
```

## What PyAgent does

- Provides a **streaming Textual TUI** for chat-based coding work.
- Supports **tool calling** for shell commands, file listing/search, text search, file reads/writes/appends/edits, and arithmetic.
- Supports **text-only mode** by disabling model tool calling for a session.
- Uses **Markdown rendering** for assistant and tool messages, with a plain-text fallback for fenced code blocks containing very long lines so transcript content does not get clipped.
- Loads **named model profiles** from JSON for easy switching between local and remote endpoints.
- Supports **Ollama** natively and **OpenAI-compatible** providers through the OpenAI Python SDK.
- Loads layered always-on instructions from user-global and project-local `AGENTS.md` files, with `.md` / `.skill` skills available for explicit or tool-driven loading.
- Supports persistent **custom tools and skills** under `~/.pyagent/`, safe from package upgrades.
- Includes optional **single-shot CLI**, **HTTP API**, **Python client**, and **browser-hosted TUI** modes.

## Installation

### Standard install

```bash
pip install pyagent-harness
```

### Install with HTTP API support

```bash
pip install pyagent-harness[api]
```

The API extra installs FastAPI and Uvicorn for `pyagent serve`.

### Developer install from a checkout

```bash
python -m pip install -e .
```

With API support:

```bash
python -m pip install -e '.[api]'
```

Non-editable local install:

```bash
python -m pip install .
```

With API support:

```bash
python -m pip install '.[api]'
```

If you only want dependencies without installing the package entry point:

```bash
pip install -r requirements.txt
```

## Model profiles

PyAgent loads named model profiles from JSON. By default it reads:

```text
~/.pyagent/profiles.json
```

Override that path with:

```bash
export PYAGENT_MODEL_PROFILES_PATH=/path/to/profiles.json
```

A sample profile file is included as [`models.example.json`](models.example.json).

### Profile file example

```json
{
  "default_profile": "local-qwen",
  "profiles": {
    "local-qwen": {
      "provider": "ollama",
      "base_url": "http://localhost:11434",
      "model": "qwen2.5-coder:7b"
    },
    "openai-gpt4": {
      "provider": "openai_compatible",
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-4.1",
      "api_key_env": "OPENAI_API_KEY"
    },
    "vllm-local": {
      "provider": "vllm",
      "base_url": "http://localhost:8000/v1",
      "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
      "api_key_env": "VLLM_API_KEY"
    }
  }
}
```

Supported provider values:

- `ollama`
- `openai_compatible`
- `openai`
- `vllm`

`openai` and `vllm` are treated as OpenAI-compatible providers.

OpenAI-compatible profiles use the OpenAI Python SDK with the Chat Completions API. PyAgent intentionally stays on `/v1/chat/completions`, not the newer Responses API, so it remains compatible with OpenAI-style servers such as OpenAI and vLLM.

### API keys

Profiles can specify either:

- `api_key` — inline secret value
- `api_key_env` — environment variable name to read at runtime

Using `api_key_env` is recommended. Inline secrets in config files age about as well as milk.

For local OpenAI-compatible servers that do not require authentication, omit both fields.

### Extra headers and HTTP transport options

Profiles may include:

- `headers` — extra HTTP headers to send with requests
- `httpx_kwargs` — keyword arguments passed to `httpx.Client` for OpenAI-compatible providers only
- `http_kwargs` — legacy alias for `httpx_kwargs`; still accepted, but not recommended

If both `httpx_kwargs` and `http_kwargs` are present, `httpx_kwargs` wins.

Example:

```json
{
  "default_profile": "local-vllm",
  "profiles": {
    "local-vllm": {
      "provider": "vllm",
      "base_url": "https://localhost:8000/v1",
      "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
      "httpx_kwargs": {
        "verify": false
      }
    }
  }
}
```

### Fallback profile from environment

If the profile file does not exist, PyAgent creates an implicit `default` profile from environment variables:

- `PYAGENT_PROFILE`
- `PYAGENT_PROVIDER`
- `PYAGENT_MODEL`
- `PYAGENT_BASE_URL`
- `PYAGENT_API_KEY`
- `PYAGENT_API_KEY_ENV`

## Running PyAgent

### Interactive TUI

```bash
pyagent
```

Or as a module:

```bash
python -m pyagent
```

Select a saved profile and optionally override its model for the current session:

```bash
pyagent --profile local-qwen
pyagent --profile openai-gpt4 --model gpt-4.1-mini
```

### Single-shot CLI

Run one prompt and exit:

```bash
pyagent --prompt "What files are in the current directory?"
```

Use a profile/model override:

```bash
pyagent --profile openai-gpt4 --prompt "Summarize README.md"
pyagent --profile openai-gpt4 --model gpt-4.1-mini --prompt "Review the current project"
```

Single-shot mode loads layered instruction context just like the TUI.

You can also load specific skills into the startup system prompt with `--skills`. Pass comma-separated scoped skill IDs (`user:<path>` or `project:<path>`). For backward compatibility, unscoped names resolve to user skills under `~/.pyagent/skills/` first:

```bash
pyagent --skills user:code-review.md,project:skills/testing.skill --prompt "Review this repository's testing strategy"
pyagent --skills code-review.md --prompt "Use my user code-review skill"
```

If any listed skill does not exist, PyAgent exits with an error. The `--skills` flag is currently supported only with `--prompt`.

### Browser-hosted TUI

PyAgent can expose the Textual app in a browser through `textual-serve`:

```bash
pyagent web
```

Optional bind and model/profile overrides:

```bash
pyagent web --host 0.0.0.0 --port 8000
pyagent web --profile local-qwen --model qwen2.5-coder:7b
```

This serves the normal `python -m pyagent` app through a small web server.

### HTTP API server

Install the API extra first:

```bash
pip install pyagent-harness[api]
```

Then start the server:

```bash
pyagent serve
```

Optional bind overrides:

```bash
pyagent serve --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` — basic health check
- `POST /run` — run a single non-streaming agent turn

Example request:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Summarize README.md",
    "messages": [
      {"role": "user", "content": "We already discussed installation."}
    ],
    "profile": "local-qwen",
    "model": "qwen2.5-coder:7b",
    "cwd": ".",
    "skills": ["code-review.md"]
  }'
```

Example response:

```json
{
  "response": "...",
  "profile": "local-qwen",
  "provider": "ollama",
  "model": "qwen2.5-coder:7b",
  "context_files": ["~/.pyagent/AGENTS.md", "AGENTS.md"]
}
```

The API uses the same profile selection, model override, context loading, and optional skill validation as single-shot CLI mode. Skills may be scoped IDs such as `user:code-review.md` or `project:skills/review.md`; unscoped names resolve to user skills first. You may pass prior conversation history in the optional `messages` field on `POST /run`; PyAgent preserves its own active system prompt and ignores incoming `system` messages so runtime instructions cannot be overridden by API callers.

If FastAPI or Uvicorn are missing, `pyagent serve` exits with a clear error.

### Python API client

PyAgent ships with a small synchronous HTTP client for the API. It uses the Python standard library, so the client does not require the server-side `api` extra just to make requests.

```python
from pyagent.client import PyAgentClient

client = PyAgentClient("http://127.0.0.1:8000")

print(client.health())

result = client.run(
    "Summarize README.md",
    messages=[{"role": "user", "content": "We already discussed installation."}],
    profile="local-qwen",
    model="qwen2.5-coder:7b",
    cwd=".",
    skills=["code-review.md"],
)

print(result.response)
print(result.profile, result.provider, result.model)
print(result.context_files)
```

Client details:

- `PyAgentClient.health()` returns the decoded `/health` JSON payload.
- `PyAgentClient.is_healthy()` returns `True` or `False` without raising on connection failures.
- `PyAgentClient.run(...)` returns a typed `RunResponse` object.
- `PyAgentClientError` is raised for HTTP errors, invalid JSON responses, connection failures, and timeouts.
- The default base URL is `http://127.0.0.1:8000`.

## Instructions, skills, and project context

PyAgent layers always-on instruction files into the active system prompt and keeps skills discoverable for explicit or model-driven loading.

Loaded first, as user-global context:

- `~/.pyagent/AGENTS.md`

Loaded next, from the current project:

- `AGENTS.md`

Available as skills, but **not loaded into the system prompt by default**:

- `~/.pyagent/skills/**/*.md`
- `~/.pyagent/skills/**/*.skill`
- `*.skill`
- `skills/**/*.md`
- `skills/**/*.skill`

Skills are plain text guidance files. The model can discover them with the built-in `list_skills` tool and load their contents as tool output with `load_skills`; this does not mutate the system prompt. Users can explicitly load skills into the system prompt with `/skills load <id-or-path>` in the TUI, `--skills` in single-shot CLI mode, or the API `skills` field.

Use `/context` in the TUI to inspect loaded instruction sources and context size. Use `/skills list` to inspect available skills. Use `/reload_context` to rescan `AGENTS.md` files and any skills explicitly loaded into the system prompt.

## Custom system prompt

PyAgent stores the base system prompt in a text file. By default:

```text
~/.pyagent/system_prompt.txt
```

On first run, PyAgent creates the file automatically if it does not already exist.

Override the location with:

```bash
export PYAGENT_SYSTEM_PROMPT_PATH="$HOME/.config/pyagent/my_prompt.txt"
pyagent
```

Or edit the default file directly:

```bash
mkdir -p ~/.pyagent
$EDITOR ~/.pyagent/system_prompt.txt
```

Notes:

- `/prompt` shows the currently active system prompt in the TUI.
- The system prompt is loaded when the conversation is initialized or reset. After editing it, use `/clear` to start a fresh conversation with the updated prompt.
- User and project instruction files are layered onto the base system prompt automatically.

## Tools

PyAgent has two tool layers:

1. **Built-in tools** shipped with the package.
2. **External user tools** under `~/.pyagent/tools/`.

Built-in tools include:

- `bash`
- `list_files`
- `find_files`
- `search_text`
- `read_file`
- `write_file`
- `append_file`
- `edit_file`
- `list_skills`
- `load_skills`

Tool calling is enabled by default. Disable all model tool calling for a session with:

```bash
export PYAGENT_TOOLS_ENABLED=false
```

Disable the built-in tool set while still allowing externally installed tools with:

```bash
export PYAGENT_BUILTIN_TOOLS_ENABLED=false
export PYAGENT_USER_TOOLS_ENABLED=true
```

Disable external user-tool discovery while keeping built-ins available with:

```bash
export PYAGENT_USER_TOOLS_ENABLED=false
```

Disable only the bash tool with:

```bash
export PYAGENT_BASH_ENABLED=false
```

When `PYAGENT_TOOLS_ENABLED=false`, PyAgent does not advertise tools to the model and adds a system instruction telling it not to call tools. When `PYAGENT_BUILTIN_TOOLS_ENABLED=false`, built-ins are omitted from the registry; external tools remain available if `PYAGENT_USER_TOOLS_ENABLED=true`.

`list_skills` and `load_skills` are read-only built-ins for model-driven skill use. `list_skills` returns scoped IDs like `user:python/testing.md` and `project:skills/review.md`. `load_skills` returns the requested skill file contents as a tool response only; it does not add those skills to the system prompt or persist session state.

### External user tools

User tools live under:

```text
~/.pyagent/tools/
```

Each user tool is a standalone Python file run through [`uv`](https://docs.astral.sh/uv/). Dependencies are declared inline using PEP 723 and installed into an isolated environment on first invocation, so adding a tool does not bloat the core PyAgent install. Miraculous, really.

Every user tool must implement two CLI subcommands:

- `<runner> run <script> describe` — print a JSON manifest with `name`, `description`, `parameters`, and optional `version`.
- `<runner> run <script> invoke --args-file <path>` — read JSON arguments from `<path>`, print the result to stdout, and exit non-zero with stderr on failure.

By default, `<runner>` is `uv`. Override it with `PYAGENT_TOOL_RUNNER` if needed.

Scaffold a new tool from inside the TUI:

```text
/tools new <name>
```

Or install an existing tool from the CLI:

```bash
pyagent tools install ./my_tool.py
pyagent tools install https://example.com/my_tool.py --name my_tool.py
```

Then reload tools in the TUI:

```text
/tools reload
```

### User tool skeleton

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click"]
# ///
import json
from pathlib import Path

import click


@click.group()
def cli():
    pass


@cli.command()
def describe():
    click.echo(json.dumps({
        "name": "my_tool",
        "description": "What this tool does — sent verbatim to the model.",
        "parameters": {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        },
        "version": "1",
    }))


@cli.command()
@click.option("--args-file", required=True, type=click.Path(exists=True, path_type=Path))
def invoke(args_file):
    args = json.loads(args_file.read_text())
    click.echo(my_logic(**args))


if __name__ == "__main__":
    cli()
```

A complete reference tool lives at [`examples/tools/search_hf_datasets.py`](examples/tools/search_hf_datasets.py). Install it with:

```bash
pyagent tools install examples/tools/search_hf_datasets.py
```

Then run `/tools reload` in PyAgent. UV installs `huggingface_hub` and `datasets` for that script on first invocation.

### External tool lifecycle

- New or changed scripts: `/tools reload` rescans the directory and rebuilds the registry.
- Schema cache: stored at `~/.pyagent/tools/.cache/manifests.json`, keyed by path, mtime, and size.
- Disable a tool: `/tools disable <name>` moves it to `~/.pyagent/tools/disabled/`.
- Re-enable a tool: `/tools enable <name>`.
- Locate a script: `/tools open <name>` prints its absolute path.
- Name collisions: built-ins win when built-in tools are enabled. `/tools` reports colliding external scripts so you can rename them. If `PYAGENT_BUILTIN_TOOLS_ENABLED=false`, those built-in names are no longer reserved and an external tool may register the same name.
- Broken scripts: timeout, non-zero `describe`, and malformed JSON are listed under "Broken external tools" and skipped.
- Missing `uv`: external tools are disabled at startup with a clear banner; built-ins still work unless `PYAGENT_BUILTIN_TOOLS_ENABLED=false`.

### Trust boundary

`~/.pyagent/tools/` is user-owned. PyAgent enforces wall-clock timeouts but does not otherwise sandbox these scripts. Treat any tool you install there as code you have chosen to run.

## Managing user skills and tools from the CLI

PyAgent can install, list, and remove user-managed skills and tools under `~/.pyagent/`.

```bash
pyagent skills list
pyagent skills install ./review.md
pyagent skills install https://example.com/review.md --name review.md
pyagent skills remove review.md

pyagent tools list
pyagent tools install ./my_tool.py
pyagent tools install https://example.com/my_tool.py --name my_tool.py
pyagent tools remove my_tool.py
```

Notes:

- Skills are installed under `~/.pyagent/skills/` and must use `.md` or `.skill`.
- Tools are installed under `~/.pyagent/tools/` and must use `.py`.
- Use `--force` with `install` to overwrite an existing file.
- Installed tools are marked executable automatically.

Recommended user directory layout:

```text
~/.pyagent/
├── profiles.json                    # named model profiles
├── system_prompt.txt                # base system prompt
├── AGENTS.md                        # optional user-global agent instructions
├── skills/                          # user-global skills (*.md, *.skill)
└── tools/                           # user tools, one UV script per tool
    ├── <my_tool>.py
    ├── disabled/                    # listed in /tools but not registered
    └── .cache/manifests.json        # automatic schema cache
```

## TUI reference

### Runtime slash commands

- `/clear` — clear the conversation
- `/help` — show command help
- `/tools` — show tool status, built-ins, external tools, and broken/disabled scripts
- `/tools on` — enable model tool calling for the current session
- `/tools off` — disable model tool calling for the current session
- `/tools reload` — rescan `~/.pyagent/tools/` and rebuild the tool registry; also available as `/reload_tools`
- `/tools new <name>` — scaffold a starter UV-script tool at `~/.pyagent/tools/<name>.py`
- `/tools enable <name>` — move a script out of `~/.pyagent/tools/disabled/`
- `/tools disable <name>` — move a script into `~/.pyagent/tools/disabled/`
- `/tools open <name>` — print the absolute path to a tool script
- `/profiles` — list saved profiles, including current/default markers and auth hints
- `/profiles reload` — reload profiles from disk
- `/reload_profiles` — reload profiles from disk
- `/profile` — show the active profile
- `/profile <name>` — switch to a saved profile
- `/profile add <name> provider=<provider> model=<model> [base_url=<url>] [api_key_env=<ENV>] [api_key=<KEY>] [default=true|false] [switch=true|false] [header.<Name>=<Value>]` — create or update a profile from the TUI
- `/model` — show the active model
- `/model list` — ask the current endpoint for available models, if supported
- `/model <name>` — override the current profile's model for this session
- `/status` — show current configuration, including the agent tool-loop max-iteration setting
- `/max_iterations <n|-1>` — set the maximum tool-loop iterations for the current session; `-1` means infinite
- `/cwd` — show current working directory
- `/history` — show recent prompt history
- `/history search <text>` — search saved prompt history for matching prompts
- `/context` — show loaded instruction sources and context size
- `/skills list` — show available user and project skills plus session load state
- `/skills load <id-or-path>` — load a user or project skill into the system prompt for this session
- `/skills unload <id-or-path>` — unload a previously loaded skill from the system prompt
- `/prompt` — show the active system prompt
- `/reload_context` — reload user-global/project `AGENTS.md` files and explicitly loaded skills, reporting added/removed files
- `/logging on|off` — enable or disable session logging under `~/.pyagent/logs/`
- `/debug` — show whether the debug pane is currently on or off
- `/debug on|off` — show or hide the debug pane

Changing tool mode at runtime resets the current conversation so the updated system prompt is applied cleanly.

Unknown slash commands may suggest a close match. For example, `/stats` may suggest `/status`.

### Profile creation from the TUI

Create or update profiles with `/profile add`. Quote values containing spaces.

```text
/profile add local-14b provider=ollama model=qwen2.5-coder:14b switch=true
/profile add openai-mini provider=openai model=gpt-4.1-mini api_key_env=OPENAI_API_KEY default=true
/profile add vllm-qwen provider=vllm model="Qwen/Qwen2.5-Coder-32B-Instruct" base_url=http://localhost:8000/v1 api_key_env=VLLM_API_KEY header.X-Project=PyAgent
```

### Keyboard shortcuts

- `Enter` — send the current prompt
- `Shift+Enter` — insert a newline in the prompt box
- `Ctrl+P` / `Ctrl+N` — move through prompt history
- `↑` / `↓` — scroll the chat transcript
- `PgUp` / `PgDn` — page through the chat transcript
- `Home` / `End` — jump to the top or bottom of the chat transcript
- `Ctrl+L` — clear the conversation
- `Ctrl+D` — toggle the debug pane
- `Ctrl+C` — quit the app

## Configuration reference

### Core environment variables

- `PYAGENT_PROFILE` — default profile name to select
- `PYAGENT_MODEL_PROFILES_PATH` — path to the JSON profile file, overriding `~/.pyagent/profiles.json`
- `PYAGENT_SYSTEM_PROMPT_PATH` — path to the system prompt text file, overriding `~/.pyagent/system_prompt.txt`
- `PYAGENT_REQUEST_TIMEOUT` — request timeout in seconds
- `PYAGENT_MAX_ITERATIONS` — maximum tool loop iterations per user turn; `-1` means infinite
- `PYAGENT_MAX_HISTORY_MESSAGES` — number of recent non-system messages to keep
- `PYAGENT_STREAM_BATCH_INTERVAL` — UI flush interval in seconds
- `PYAGENT_USER_DIR` — root for user-managed tools, skills, logs, and user-global `AGENTS.md`; default `~/.pyagent`. Model profiles use `PYAGENT_MODEL_PROFILES_PATH`.

### Tool environment variables

- `PYAGENT_TOOLS_ENABLED` — enable or disable all model tool calling for the session; default `true`
- `PYAGENT_BUILTIN_TOOLS_ENABLED` — register built-in tools (`bash`, file tools, search/edit tools); default `true`
- `PYAGENT_BASH_ENABLED` — enable or disable bash execution; default `true` (when `false`, the bash tool remains registered but returns a disabled-by-configuration error)
- `PYAGENT_BASH_READONLY_MODE` — restrict bash to read-only command prefixes
- `PYAGENT_BASH_TIMEOUT_DEFAULT` — default bash timeout in seconds
- `PYAGENT_BASH_BLOCKED_SUBSTRINGS` — comma-separated dangerous bash fragments to block
- `PYAGENT_BASH_READONLY_PREFIXES` — comma-separated allowed prefixes in read-only mode
- `PYAGENT_USER_TOOLS_ENABLED` — discover and register external tools under `~/.pyagent/tools/`; default `true`
- `PYAGENT_USER_TOOL_TIMEOUT` — wall-clock timeout in seconds for each external tool invocation; default `60`
- `PYAGENT_USER_TOOL_DESCRIBE_TIMEOUT` — wall-clock timeout for the `describe` schema fetch; default `10`
- `PYAGENT_TOOL_RUNNER` — executable used to run external tools; default `uv`

These tool environment variables apply to the TUI, single-shot CLI mode, and agents created by the HTTP API server process.

### Fallback profile environment variables

Used when no profile file exists:

- `PYAGENT_PROVIDER`
- `PYAGENT_MODEL`
- `PYAGENT_BASE_URL`
- `PYAGENT_API_KEY`
- `PYAGENT_API_KEY_ENV`

### Profile JSON fields

- `provider` — provider type: `ollama`, `openai_compatible`, `openai`, or `vllm`
- `base_url` — endpoint base URL
- `model` — model name
- `api_key` — inline API key, if needed
- `api_key_env` — environment variable containing the API key
- `headers` — optional object of extra HTTP headers
- `httpx_kwargs` — optional object of keyword arguments passed to `httpx.Client` for OpenAI-compatible profiles only
- `http_kwargs` — legacy alias for `httpx_kwargs`

## Development

Quick smoke test:

```bash
python test_agent.py
```

For non-trivial changes, run:

```bash
python -m py_compile pyagent/*.py test_agent.py
python -m unittest -v
```
