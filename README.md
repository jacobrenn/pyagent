# PyAgent

A lightweight coding agent built with Textual and a configurable multi-provider chat backend.

## Features

- **Streaming chat UI** built with Textual
- **Markdown rendering** for final assistant and tool messages, with a plain-text fallback for fenced code blocks that contain very long lines so transcript content does not get clipped
- **Tool use** for shell commands, file search/text search, file reads/writes/appends/edits, and listing files
- **Optional text-only mode** by disabling all model tool calling for a session
- **Provider support** for:
  - native **Ollama** chat endpoints
  - **OpenAI-compatible** chat endpoints such as OpenAI, vLLM, and other `/v1/chat/completions` servers
- **OpenAI Python SDK integration** for OpenAI-compatible chat completions and model listing
- **Named model profiles** stored in JSON for easy switching between endpoints and models
- **API key support** through inline values or environment-variable references
- **Conversation reset** with `Ctrl+L` or `/clear`
- **Scrollable transcript** with mouse wheel, `↑` / `↓`, or `PgUp` / `PgDn`
- **Multi-line prompt input** with `Shift+Enter`; press `Enter` to send, the input box auto-grows as you type, and the prompt area shows a helper hint
- **Prompt history** with `Ctrl+P` / `Ctrl+N`, plus `/history search <text>` from the TUI
- **Keyboard shortcuts** including `Ctrl+L` to clear the conversation, `Ctrl+D` to toggle the debug pane, and transcript scrolling with `↑` / `↓` / `PgUp` / `PgDn` / `Home` / `End`
- **Slash commands** such as `/help`, `/tools`, `/profiles`, `/profile`, `/model`, `/status`, `/cwd`, `/history`, `/context`, `/prompt`, `/reload_context`, and `/debug on|off`, with `/help` also summarizing prompt and transcript keybindings
- **Automatic project instructions** loaded from `AGENTS.md` and local skill files on startup, with `/context` and `/reload_context` for inspection and refresh
- **Persistent custom tools and skills** under `~/.pyagent/` that survive `pip install --upgrade`. Each user-managed tool is a standalone UV script (PEP 723) with click subcommands, so adding a new tool with new dependencies never touches the core install

## Requirements

- Python 3.10+
- A supported endpoint such as:
  - [Ollama](https://ollama.com)
  - OpenAI
  - vLLM or another OpenAI-compatible server
- A model with tool-calling support

## Installation

Install PyAgent via pip:

```bash
pip install pyagent-harness
```

If you are developing and want to install locally from the repo root:

```bash
python -m pip install -e .
```

If you only want the dependencies without installing the package entry point, this still works:

```bash
pip install -r requirements.txt
```

PyAgent uses the `openai` Python SDK for OpenAI-compatible profiles and keeps the native Ollama HTTP path for Ollama profiles.

## Running the TUI

After installation, run PyAgent from any directory with:

```bash
pyagent
```

You can also launch it as a module:

```bash
python -m pyagent
```

To choose a saved profile and optionally override its model for the current session:

```bash
pyagent --profile local-qwen
pyagent --profile openai-gpt4 --model gpt-4.1-mini
```

If the current working directory contains `AGENTS.md`, `*.skill`, or files under `skills/**/*.md` / `skills/**/*.skill`, PyAgent will load them into the system prompt automatically at startup. You can inspect the currently loaded sources with `/context` and refresh them while the app is running with `/reload_context`.

## Model profiles

PyAgent loads named profiles from JSON. By default it looks for:

~/.pyagent/profiles.json

You can override the location with:

- `PYAGENT_MODEL_PROFILES_PATH`

A sample file is included in the repo as `models.example.json`.

### Example profile file

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

Provider values:

- `ollama`
- `openai_compatible`
- `openai`
- `vllm`

`openai` and `vllm` are treated as OpenAI-compatible providers.

OpenAI-compatible profiles use the `openai` Python SDK with the Chat Completions API. This keeps PyAgent on `/v1/chat/completions` rather than the newer Responses API so it remains compatible with OpenAI-style servers such as OpenAI and vLLM.

### API keys

Profiles can specify either:

- `api_key` — inline secret value
- `api_key_env` — environment variable name to read at runtime

Using `api_key_env` is recommended.

For local OpenAI-compatible servers that do not require authentication, you can omit both `api_key` and `api_key_env`.

### Fallback behavior

If the profile file does not exist, PyAgent creates an implicit `default` profile from environment variables.

Useful env vars for that fallback:

- `PYAGENT_PROFILE`
- `PYAGENT_PROVIDER`
- `PYAGENT_MODEL`
- `PYAGENT_BASE_URL`
- `PYAGENT_API_KEY`
- `PYAGENT_API_KEY_ENV`

## Tool configuration

- `PYAGENT_TOOLS_ENABLED` — enable or disable all model tool calling for the session (`true` by default)
- `PYAGENT_BASH_ENABLED` — enable or disable the `bash` tool specifically (`true` by default)

When `PYAGENT_TOOLS_ENABLED=false`, PyAgent does not advertise tools to the model and adds a system instruction telling it not to call tools.

## Runtime slash commands

- `/tools` — show current tool status, built-in tools, external user tools, and any broken/disabled scripts
- `/tools on` — enable model tool calling for the current session
- `/tools off` — disable model tool calling for the current session
- `/tools reload` — re-scan `~/.pyagent/tools/` and rebuild the tool registry (also available as `/reload_tools`)
- `/tools new <name>` — scaffold a starter UV-script tool at `~/.pyagent/tools/<name>.py`
- `/tools enable <name>` — move a script out of `~/.pyagent/tools/disabled/`
- `/tools disable <name>` — move a script into `~/.pyagent/tools/disabled/`
- `/tools open <name>` — print the absolute path to a tool script

Changing tool mode at runtime resets the current conversation so the updated system prompt is applied cleanly.

- `/clear` — clear the conversation
- `/help` — show command help
- `/tools` — list tools
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
- `/max_iterations <n|-1>` — set the maximum tool-loop iterations for the current session (`-1` means infinite)
- `/cwd` — show current working directory
- `/history` — show recent prompt history
- `/history search <text>` — search saved prompt history for matching prompts
- `/context` — show loaded user-global and project instruction files and context size
- `/prompt` — show the active system prompt
- `/reload_context` — reload `~/.pyagent/AGENTS.md`, `~/.pyagent/skills/**`, and local instruction files and report added/removed files
- `/debug` — show whether the debug pane is currently on or off
- `/debug on|off` — show or hide the debug pane

Unknown slash commands may suggest a close match, for example `/stats` may suggest `/status`.

## Keyboard shortcuts

- `Enter` — send the current prompt
- `Shift+Enter` — insert a newline in the prompt box
- `Ctrl+P` / `Ctrl+N` — move through prompt history
- `↑` / `↓` — scroll the chat transcript
- `PgUp` / `PgDn` — page through the chat transcript
- `Home` / `End` — jump to the top or bottom of the chat transcript
- `Ctrl+L` — clear the conversation
- `Ctrl+D` — toggle the debug pane
- `Ctrl+C` — quit the app

### Profile creation from the TUI

Profile creation and updates are available through `/profile add`.
Values containing spaces should be quoted.

Examples:

```text
/profile add local-14b provider=ollama model=qwen2.5-coder:14b switch=true
/profile add openai-mini provider=openai model=gpt-4.1-mini api_key_env=OPENAI_API_KEY default=true
/profile add vllm-qwen provider=vllm model="Qwen/Qwen2.5-Coder-32B-Instruct" base_url=http://localhost:8000/v1 api_key_env=VLLM_API_KEY header.X-Project=PyAgent
```

## Configuration

Environment variables:

- `PYAGENT_PROFILE` — default profile name to select
- `PYAGENT_MODEL_PROFILES_PATH` — path to the JSON profile file, overriding the default `~/.pyagent/profiles.json` location
- `PYAGENT_SYSTEM_PROMPT_PATH` — path to the system prompt text file, overriding the default `~/.pyagent/system_prompt.txt` location
- `PYAGENT_REQUEST_TIMEOUT` — request timeout in seconds
- `PYAGENT_MAX_ITERATIONS` — maximum tool loop iterations per user turn (`-1` means infinite)
- `PYAGENT_MAX_HISTORY_MESSAGES` — number of recent non-system messages to keep
- `PYAGENT_STREAM_BATCH_INTERVAL` — UI flush interval in seconds
- `PYAGENT_BASH_ENABLED` — enable or disable the bash tool
- `PYAGENT_BASH_READONLY_MODE` — restrict bash to read-only command prefixes
- `PYAGENT_BASH_TIMEOUT_DEFAULT` — default bash timeout in seconds
- `PYAGENT_BASH_BLOCKED_SUBSTRINGS` — comma-separated dangerous bash fragments to block
- `PYAGENT_BASH_READONLY_PREFIXES` — comma-separated allowed prefixes in read-only mode
- `PYAGENT_USER_DIR` — root for user-managed tools, skills, and `profiles.json` (default `~/.pyagent`)
- `PYAGENT_USER_TOOLS_ENABLED` — discover and register external tools under `~/.pyagent/tools/` (`true` by default)
- `PYAGENT_USER_TOOL_TIMEOUT` — wall-clock timeout in seconds for each external tool invocation (default `60`)
- `PYAGENT_USER_TOOL_DESCRIBE_TIMEOUT` — wall-clock timeout for the `describe` schema fetch (default `10`)
- `PYAGENT_TOOL_RUNNER` — executable used to run external tools (defaults to `uv`; advanced override)

Fallback profile env vars when no profile file exists:

- `PYAGENT_PROVIDER`
- `PYAGENT_MODEL`
- `PYAGENT_BASE_URL`
- `PYAGENT_API_KEY`
- `PYAGENT_API_KEY_ENV`

## Custom system prompt

PyAgent stores the active system prompt in a text file. By default that file is:

```text
~/.pyagent/system_prompt.txt
```

On first run, PyAgent creates that file automatically if it does not already exist.

You can override the location with:

- `PYAGENT_SYSTEM_PROMPT_PATH`

Examples:

```bash
export PYAGENT_SYSTEM_PROMPT_PATH="$HOME/.config/pyagent/my_prompt.txt"
pyagent
```

Or edit the default prompt file directly:

```bash
mkdir -p ~/.pyagent
$EDITOR ~/.pyagent/system_prompt.txt
```

A few useful notes:

- `/prompt` shows the currently active system prompt inside the TUI.
- The system prompt is loaded when the conversation is initialized or reset, so after editing the file you should use `/clear` to start a fresh conversation with the updated prompt.
- Project and user instruction files (`AGENTS.md`, `skills/**`, `*.skill`) are layered onto the base system prompt automatically.

## Custom tools and skills

Anything you add for yourself — custom tools, custom skills, custom `AGENTS.md` instructions — should live under `~/.pyagent/` so a `pip install --upgrade` of PyAgent does not wipe it out. Built-in tools (`bash`, `list_files`, `find_files`, `search_text`, `read_file`, `write_file`, `append_file`, `edit_file`) stay inside the package; user tools layer on top.

### Layout

```text
~/.pyagent/
├── profiles.json                      # named model profiles (existing)
├── AGENTS.md                        # optional user-global agent instructions
├── skills/                          # user-global skills (*.md, *.skill)
└── tools/                           # user tools (one UV script per tool)
    ├── <my_tool>.py
    ├── disabled/                    # listed in /tools but not registered
    └── .cache/manifests.json        # auto schema cache (path+mtime+size keyed)
```

### Custom tools (UV scripts with click subcommands)

Each user tool is a single self-contained Python file. PyAgent runs it through [`uv`](https://docs.astral.sh/uv/) so its dependencies are declared inline (PEP 723) and installed into an isolated venv on first invocation. The core PyAgent install never grows when you add a new tool.

Every tool must implement two CLI subcommands:

- `<runner> run <script> describe` — print a JSON manifest with `name`, `description`, `parameters` (a JSON-Schema-shaped object), and an optional `version`. By default `<runner>` is `uv`. The output is cached by path + mtime + size, so subsequent startups skip the subprocess.
- `<runner> run <script> invoke --args-file <path>` — read the tool arguments as a JSON object from `<path>`, print the result to stdout, and exit non-zero with an error on stderr if anything goes wrong. By default `<runner>` is `uv`.

Use `/tools new <name>` from inside PyAgent to scaffold a starter file, or write one by hand. The built-in scaffold and examples use `uv`, which is the recommended runner. Skeleton:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "huggingface_hub", "datasets"]
# ///
import json
import sys
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

### Reference example: `search_hf_datasets`

`examples/tools/search_hf_datasets.py` is a fully fleshed-out reference tool (Hugging Face dataset search) using the same contract. To install it for yourself:

```bash
mkdir -p ~/.pyagent/tools
cp examples/tools/search_hf_datasets.py ~/.pyagent/tools/
```

Then inside PyAgent run `/tools reload`. UV will install `huggingface_hub` and `datasets` on first invocation, into the script's own venv — your PyAgent install stays lean.

### Lifecycle

- New / changed scripts: `/tools reload` re-scans the directory and rebuilds the registry. The schema cache invalidates automatically when the file's path, mtime, or size changes.
- Temporarily turn a tool off: `/tools disable <name>` moves it to `~/.pyagent/tools/disabled/` (still listed in `/tools`, not registered).
- Re-enable: `/tools enable <name>`.
- Locate a script: `/tools open <name>` prints the absolute path.
- Name collisions: built-ins always win. If your script's `name` collides with a built-in, `/tools` shows a warning row with the colliding script path so you can rename it.
- Bad scripts (timeout, non-zero `describe`, malformed JSON) are listed under "Broken external tools" and skipped; healthy tools keep loading.
- Missing `uv`: external tools are disabled at startup with a clear banner; built-ins continue to work.

### Custom skills and `AGENTS.md`

`~/.pyagent/AGENTS.md`, `~/.pyagent/skills/**/*.md`, and `~/.pyagent/skills/**/*.skill` are loaded into the system prompt at startup as **user-global** instructions, layered before any project-specific `AGENTS.md` or `skills/` files in the current working directory. `/context` lists each source with its scope, and `/reload_context` re-scans both layers.

### Trust boundary

`~/.pyagent/tools/` is user-owned. PyAgent enforces wall-clock timeouts (`PYAGENT_USER_TOOL_TIMEOUT`, `PYAGENT_USER_TOOL_DESCRIBE_TIMEOUT`) but does not otherwise sandbox these scripts. Treat any tool you drop into `~/.pyagent/tools/` the same as you would any code you choose to run.

## Quick CLI smoke test

```bash
python test_agent.py
```

## Development test commands

For non-trivial changes, run:

```bash
python -m py_compile pyagent/*.py test_agent.py
python -m unittest -v
```
