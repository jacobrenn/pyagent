# Tools skill

Use this skill when changing `pyagent/tools.py`, tool registration, or tool behavior exposed to the model.

## Goals

- Prefer dedicated tools over shell workarounds.
- Keep tools predictable and easy for models to use.
- Keep tool output concise but informative.
- Keep tool schemas portable across supported providers.

## Where tools live

There are two distinct surfaces:

1. **Built-in tools** — implemented in `pyagent/tools.py` and registered through `create_default_tool_registry`. These are stable, fast, in-process, and ship with the core install.
2. **External user tools** — UV scripts under `~/.pyagent/tools/`. Discovered by `pyagent/external_tools.py` at startup, registered alongside built-ins, and invoked via `uv run <script> invoke --args-file <tmp>` with a wall-clock timeout. Each script declares its own dependencies via PEP 723, so adding a new external tool never grows the core install.

User-supplied tools should live under `~/.pyagent/tools/`. **Do not** add new tools to `pyagent/tools.py` unless they are core capabilities that everyone running PyAgent should have. The persistent layout is documented in `README.md` under "Custom tools and skills" and the contract is:

- `uv run <script> describe` prints the JSON manifest (`name`, `description`, `parameters`, optional `version`).
- `uv run <script> invoke --args-file <path>` reads JSON arguments from `<path>`, prints the result to stdout, and exits non-zero with stderr on failure.

Use `pyagent/templates/tool_template.py` (rendered by `/tools new <name>` or `pyagent.scaffold.create_user_tool`) as the canonical starter; `examples/tools/search_hf_datasets.py` is a fully fleshed-out reference example.

## Current tool categories

Built-in tools include:

- shell execution (`bash`)
- file listing/search (`list_files`, `find_files`, `search_text`)
- file reading/writing/appending/editing
- arithmetic

## Tool design guidance

### Prefer explicit schemas

Tool descriptions and parameter schemas should be clear and model-friendly.
Names should be simple and action-oriented.

### Prefer deterministic behavior

Avoid ambiguous behavior that depends on hidden state.
If a tool can fail, return a clear error string.

### Keep dedicated tools strong

Before expanding shell usage, ask whether the capability should instead be a first-class tool.
Examples:

- searching text should use `search_text`
- filename discovery should use `find_files`
- file edits should use `edit_file`

### Provider portability matters

Tool definitions are sent to both native Ollama and OpenAI-compatible endpoints.
When changing tool schemas:

- keep them plain JSON-schema-like objects
- avoid provider-specific assumptions in tool metadata
- preserve stable argument names unless a breaking change is intended

## Shell safety guidance

Be conservative with `bash`.
If changing shell behavior:

- preserve or improve safety checks
- keep config-driven policy in `pyagent/config.py`
- document new env vars in `README.md`
- add tests in `test_agent.py`

Do not casually relax blocked-command policy.

## Output guidance

Tool outputs should:

- be readable in the transcript
- be truncated when necessary
- still contain enough detail for the agent to continue reasoning

## If you add a new built-in tool

Also update:

- `README.md` feature/config docs if relevant
- slash-command tooling output if needed
- tests in `test_agent.py`

## If a user wants to add a custom tool

Direct them to the user-extension layer rather than the core package:

1. `mkdir -p ~/.pyagent/tools`
2. `cp examples/tools/search_hf_datasets.py ~/.pyagent/tools/` (or run `/tools new <name>` from the TUI to scaffold a starter)
3. Edit the script. Declare dependencies in the PEP 723 `# /// script` block.
4. Inside PyAgent, run `/tools reload` to register the new tool.

Collisions with built-ins are resolved in the built-in's favor; `/tools` flags the conflict so the user can rename their script. Broken scripts (timeout, non-zero `describe`, malformed JSON) are listed under "Broken external tools" and skipped.

## Preferred future direction

If working on tool architecture, good directions include:

- modular tool organization
- clearer registry composition
- more dedicated search/edit tools
- structured tool metadata/results
