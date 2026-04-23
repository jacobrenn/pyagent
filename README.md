# PyAgent

A lightweight coding agent built with Textual and a configurable multi-provider chat backend.

## Features

- **Streaming chat UI** built with Textual
- **Markdown rendering** for final assistant and tool messages
- **Tool use** for shell commands, file search/text search, file reads/writes/appends/edits, listing files, and calculation
- **Provider support** for:
  - native **Ollama** chat endpoints
  - **OpenAI-compatible** chat endpoints such as OpenAI, vLLM, and other `/v1/chat/completions` servers
- **Named model profiles** stored in JSON for easy switching between endpoints and models
- **API key support** through inline values or environment-variable references
- **Conversation reset** with `Ctrl+L` or `/clear`
- **Scrollable transcript** with mouse wheel, `↑` / `↓`, or `PgUp` / `PgDn`
- **Multi-line prompt input** with `Shift+Enter`; press `Enter` to send, the input box auto-grows as you type, and the prompt area shows a helper hint
- **Prompt history** with `Ctrl+P` / `Ctrl+N`
- **Slash commands** such as `/help`, `/tools`, `/profiles`, `/profile`, `/model`, `/status`, `/cwd`, `/history`, `/prompt`, `/reload_context`, and `/debug on|off`
- **Automatic project instructions** loaded from `AGENTS.md` and local skill files on startup

## Requirements

- Python 3.10+
- A supported endpoint such as:
  - [Ollama](https://ollama.com)
  - OpenAI
  - vLLM or another OpenAI-compatible server
- A model with tool-calling support

## Installation

Install PyAgent locally from the repo root:

```bash
python -m pip install -e .
```

If you only want the dependencies without installing the package entry point, this still works:

```bash
pip install -r requirements.txt
```

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

If the current working directory contains `AGENTS.md`, `*.skill`, or files under `skills/**/*.md` / `skills/**/*.skill`, PyAgent will load them into the system prompt automatically at startup. You can refresh them while the app is running with `/reload_context`.

## Model profiles

PyAgent loads named profiles from JSON. By default it looks for:

```text
~/pyagent/models.json
```

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

### API keys

Profiles can specify either:

- `api_key` — inline secret value
- `api_key_env` — environment variable name to read at runtime

Using `api_key_env` is recommended.

### Fallback behavior

If the profile file does not exist, PyAgent creates an implicit `default` profile from environment variables.

Useful env vars for that fallback:

- `PYAGENT_PROFILE`
- `PYAGENT_PROVIDER`
- `PYAGENT_MODEL`
- `PYAGENT_BASE_URL`
- `PYAGENT_API_KEY`
- `PYAGENT_API_KEY_ENV`

## Runtime slash commands

- `/clear` — clear the conversation
- `/help` — show command help
- `/tools` — list tools
- `/profiles` — list saved profiles
- `/profiles reload` — reload profiles from disk
- `/reload_profiles` — reload profiles from disk
- `/profile` — show the active profile
- `/profile <name>` — switch to a saved profile
- `/profile add <name> provider=<provider> model=<model> [base_url=<url>] [api_key_env=<ENV>] [api_key=<KEY>] [default=true|false] [switch=true|false] [header.<Name>=<Value>]` — create or update a profile from the TUI
- `/model` — show the active model
- `/model list` — ask the current endpoint for available models, if supported
- `/model <name>` — override the current profile's model for this session
- `/status` — show current configuration
- `/cwd` — show current working directory
- `/history` — show recent prompt history
- `/prompt` — show the active system prompt
- `/reload_context` — reload `AGENTS.md` and local skill files
- `/debug on|off` — show or hide the debug pane

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
- `PYAGENT_MODEL_PROFILES_PATH` — path to the JSON profile file, overriding the default `~/pyagent/models.json` location
- `PYAGENT_REQUEST_TIMEOUT` — request timeout in seconds
- `PYAGENT_MAX_ITERATIONS` — maximum tool loop iterations per user turn
- `PYAGENT_MAX_HISTORY_MESSAGES` — number of recent non-system messages to keep
- `PYAGENT_STREAM_BATCH_INTERVAL` — UI flush interval in seconds
- `PYAGENT_BASH_ENABLED` — enable or disable the bash tool
- `PYAGENT_BASH_READONLY_MODE` — restrict bash to read-only command prefixes
- `PYAGENT_BASH_TIMEOUT_DEFAULT` — default bash timeout in seconds
- `PYAGENT_BASH_BLOCKED_SUBSTRINGS` — comma-separated dangerous bash fragments to block
- `PYAGENT_BASH_READONLY_PREFIXES` — comma-separated allowed prefixes in read-only mode

Fallback profile env vars when no profile file exists:

- `PYAGENT_PROVIDER`
- `PYAGENT_MODEL`
- `PYAGENT_BASE_URL`
- `PYAGENT_API_KEY`
- `PYAGENT_API_KEY_ENV`

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
