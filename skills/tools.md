# Tools skill

Use this skill when changing `tools.py`, tool registration, or tool behavior exposed to the model.

## Goals

- Prefer dedicated tools over shell workarounds.
- Keep tools predictable and easy for models to use.
- Keep tool output concise but informative.
- Keep tool schemas portable across supported providers.

## Current tool categories

The repo currently includes tools for:

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
- keep config-driven policy in `config.py`
- document new env vars in `README.md`
- add tests in `test_agent.py`

Do not casually relax blocked-command policy.

## Output guidance

Tool outputs should:

- be readable in the transcript
- be truncated when necessary
- still contain enough detail for the agent to continue reasoning

## If you add a new tool

Also update:

- `README.md` feature/config docs if relevant
- slash-command tooling output if needed
- tests in `test_agent.py`

## Preferred future direction

If working on tool architecture, good directions include:

- modular tool organization
- clearer registry composition
- more dedicated search/edit tools
- structured tool metadata/results
