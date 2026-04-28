# AGENTS.md

Instructions for agents working in this repository.

## Project purpose

This repo implements **PyAgent**, a lightweight coding agent built with:

- **Textual** for the TUI
- a Python agent loop that streams model output, executes tools, and feeds tool results back to the model
- **multiple model backends**, including native Ollama and OpenAI-compatible endpoints such as OpenAI and vLLM
- **named model profiles** stored in JSON so endpoints and models can be switched easily

The project is intentionally lightweight. Prefer small, compatible changes over heavy abstractions unless a refactor clearly improves maintainability.

## Architecture overview

Core files:

- `pyagent/main.py` — launches the TUI
- `pyagent/ui.py` — Textual app, transcript rendering, prompt input, slash commands, prompt history/search, context inspection, scroll behavior
- `pyagent/agent.py` — agent loop, tool-call handling, provider switching, fallback behavior after tool results
- `pyagent/tools.py` — tool registry plus built-in tools
- `pyagent/external_tools.py` — discovery, schema cache, and subprocess execution for user-managed tools under `~/.pyagent/tools/`
- `pyagent/scaffold.py` + `pyagent/templates/tool_template.py` — `/tools new` scaffolding
- `pyagent/user_runtime.py` — single source of truth for `~/.pyagent/` paths and runner availability
- `pyagent/config.py` — environment-driven runtime config and system prompt
- `pyagent/model_profiles.py` — loads saved model profiles from JSON and env fallback
- `pyagent/llm_client.py` — provider-specific clients and streaming normalization
- `pyagent/project_context.py` — loads user-global (`~/.pyagent/AGENTS.md`, `~/.pyagent/skills/**`) and project-local (`AGENTS.md`, `skills/**`, `*.skill`) instructions into the system prompt
- `examples/tools/search_hf_datasets.py` — reference UV-script tool users can copy into `~/.pyagent/tools/`
- `test_agent.py` — unit tests

## Development priorities

When making changes, prefer these goals:

1. Keep the TUI responsive and simple.
2. Preserve streaming behavior.
3. Prefer dedicated tools over shell-based workarounds.
4. Keep tool behavior explicit and testable.
5. Keep provider-specific behavior contained in the client/profile layer where practical.

## Tooling expectations inside this repo

### Prefer purpose-built tools

If implementing agent-facing capabilities, prefer:

- search tools over shell grep
- file editing tools over raw shell edits
- config-driven behavior over hard-coded toggles

### Built-in vs user-managed tools

Built-in tools live in `pyagent/tools.py`. They are stable, in-process, and ship with the core install.

User-supplied tools belong under `~/.pyagent/tools/` as standalone UV scripts (PEP 723 + click) and should **not** be added to `pyagent/tools.py`. Discovery and execution live in `pyagent/external_tools.py`. The contract is `uv run <script> describe` and `uv run <script> invoke --args-file <path>`. See `README.md` ("Custom tools and skills") and `skills/tools.md` for the full layout and contract. Built-ins always win on name collisions; `/tools` reports the conflict.

### Shell safety

`bash` is intentionally restricted by config.
Do not weaken shell safety casually.
If you change shell policy, update:

- `pyagent/config.py`
- `pyagent/tools.py`
- `README.md`
- tests in `test_agent.py`

## UI conventions

In `pyagent/ui.py`:

- Keep transcript rendering compact.
- Avoid overly noisy status text.
- Keep prompt controls discoverable in the prompt subtitle.
- Preserve current keybindings unless there is a strong reason to change them.
- Be careful with Textual layout defaults; `height: auto` matters for expanding message widgets.

If touching scroll behavior, test manually in the running TUI.

## Agent loop conventions

In `pyagent/agent.py`:

- The model may stop after a tool call with an incomplete answer.
- Preserve the existing fallback behavior that appends tool output when needed.
- If changing message history or system prompt composition, make sure project context still gets applied after reset.
- Keep provider-specific request/stream formatting out of the core loop when possible.

## Model/profile conventions

- Keep profile storage simple and file-based.
- Prefer JSON over new dependencies.
- Prefer `api_key_env` over inline secrets in examples and docs.
- Make profile switching explicit in the UI rather than relying on hidden env-only state.
- If changing profile semantics, update `README.md`, `AGENTS.md`, and relevant skills docs.

## Project-context loading

This repo auto-loads agent instructions from two layered sources:

- **User-global** (loaded first):
  - `~/.pyagent/AGENTS.md`
  - `~/.pyagent/skills/**/*.md`
  - `~/.pyagent/skills/**/*.skill`
- **Project-local** (loaded next, layered on top):
  - `AGENTS.md`
  - `*.skill`
  - `skills/**/*.md`
  - `skills/**/*.skill`

If you change that behavior:

- keep limits on prompt/context size (per-section budgets must stay)
- keep startup loading transparent in the UI
- preserve `/reload_context` and `/context`, including the global-vs-project labelling
- update tests and docs

## Testing requirements

For any non-trivial change, run:

```bash
python -m py_compile pyagent/*.py test_agent.py
python -m unittest -v
```

If you change only a subset of files, still prefer running the full test suite unless the user explicitly wants a minimal pass.

## Documentation requirements

If behavior changes for users, update `README.md`.

Examples:

- new slash commands
- changed slash-command output or suggestions
- new tools or tool semantics (built-in or external)
- new env vars (including `PYAGENT_USER_DIR`, `PYAGENT_USER_TOOLS_ENABLED`, `PYAGENT_USER_TOOL_TIMEOUT`, `PYAGENT_USER_TOOL_DESCRIBE_TIMEOUT`, `PYAGENT_TOOL_RUNNER`)
- changed keybindings
- project-context loading behavior or user-global skills
- external tool contract / UV-script schema
- provider/profile configuration
- profile reload or in-TUI profile editing behavior

## Coding style

- Keep functions small and direct.
- Prefer clear helper methods over clever inline logic.
- Match existing style and typing patterns.
- Avoid introducing dependencies unless clearly justified.
- Add tests for bug fixes and new behavior.

## Good next-step areas

Changes in these areas are generally welcome if requested:

- model profile UX
- provider extensibility
- better search/edit primitives
- safer shell execution
- improved slash commands and history
- cleaner UI rendering
- stronger tests
- better project-context loading
- richer user-extension layer (template improvements, additional reference tools under `examples/tools/`)

## Avoid

- large framework-style rewrites without clear need
- weakening shell safety by default
- breaking current slash commands / keybindings silently
- replacing dedicated tools with shell-only logic
- removing tests to make refactors easier
- adding user-managed tools to `pyagent/tools.py` instead of the persistent `~/.pyagent/tools/` layer
- pulling heavy optional dependencies into the core install when they belong in a per-tool PEP 723 block
