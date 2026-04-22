# PyAgent

A lightweight local coding agent built with Ollama and Textual.

## What's improved

- Faster UI streaming: assistant output is streamed as plain text, then rendered as Markdown once complete.
- More reliable Ollama client with connection reuse and clearer error handling.
- Structured agent events for cleaner UI integration and future extensibility.
- Configurable runtime via environment variables.
- Safer, more informative tool execution with output truncation.
- Better TUI ergonomics: status bar, clear-chat shortcut, tool activity messages.

## Features

- **Streaming chat UI** built with Textual
- **Markdown rendering** for final assistant and tool messages
- **Tool use** for shell commands, file search/text search, file reads/writes/appends/edits, listing files, and calculation
- **Local model support** through Ollama
- **Conversation reset** with `Ctrl+L` or `/clear`
- **Scrollable transcript** with mouse wheel, `↑` / `↓`, or `PgUp` / `PgDn`
- **Multi-line prompt input** with `Shift+Enter`; press `Enter` to send, the input box auto-grows as you type, and the prompt area shows a helper hint
- **Prompt history** with `Ctrl+P` / `Ctrl+N`
- **Slash commands** such as `/help`, `/tools`, `/model`, `/model list`, `/status`, `/cwd`, `/history`, `/prompt`, `/reload_context`, and `/debug on|off`
- **Automatic project instructions** loaded from `AGENTS.md` and local skill files on startup

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com)
- A tool-capable model, such as `llama3.1`, `qwen2.5`, or another Ollama chat model

## Installation

```bash
pip install -r requirements.txt
ollama pull llama3.1
```

## Running the TUI

`main.py` is executable, so you can run the app directly:

```bash
./main.py
```

To override the configured Ollama model for a session:

```bash
./main.py --model llama3.1
```

Note: on most shells, running a script from the current directory requires `./main.py`. If you want to run it as just `main.py`, make sure the repository directory is on your `PATH`.

If the current working directory contains `AGENTS.md`, `*.skill`, or files under `skills/**/*.md` / `skills/**/*.skill`, PyAgent will load them into the system prompt automatically at startup. You can refresh them while the app is running with `/reload_context`.

## Quick CLI smoke test

```bash
python test_agent.py
```

## Configuration

You can configure the app with environment variables:

- `PYAGENT_MODEL` — default Ollama model name
- `PYAGENT_BASE_URL` — Ollama base URL, default `http://localhost:11434`
- `PYAGENT_REQUEST_TIMEOUT` — request timeout in seconds
- `PYAGENT_MAX_ITERATIONS` — maximum tool loop iterations per user turn
- `PYAGENT_MAX_HISTORY_MESSAGES` — number of recent non-system messages to keep
- `PYAGENT_STREAM_BATCH_INTERVAL` — UI flush interval in seconds
- `PYAGENT_BASH_ENABLED` — enable or disable the bash tool
- `PYAGENT_BASH_READONLY_MODE` — restrict bash to read-only command prefixes
- `PYAGENT_BASH_TIMEOUT_DEFAULT` — default bash timeout in seconds
- `PYAGENT_BASH_BLOCKED_SUBSTRINGS` — comma-separated dangerous bash fragments to block
- `PYAGENT_BASH_READONLY_PREFIXES` — comma-separated allowed prefixes in read-only mode

Example:

```bash
PYAGENT_MODEL=llama3.1 PYAGENT_BASH_READONLY_MODE=true python main.py
python main.py --model qwen2.5-coder
```

At runtime, you can inspect or change models from the TUI:

- `/model` — show the current model and usage
- `/model list` — list installed Ollama models
- `/model llama3.1` — switch to a different model without clearing the conversation
